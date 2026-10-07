"""
nvidia_llm_bridge.py
--------------------
Drop-in async replacement for firebase_llm_bridge.py.

Instead of pushing tasks to Firebase and waiting for a Colab worker,
this module calls the NVIDIA NIM API directly using the OpenAI-compatible
client — exactly as demonstrated in test.py.

Model: configured by NVIDIA_MODEL
Endpoint: https://integrate.api.nvidia.com/v1

Usage
-----
    from nvidia_llm_bridge import NvidiaBridge

    bridge = NvidiaBridge()

    # Plain text
    text = await bridge.call_text(system_prompt="...", user_prompt="...")

    # JSON (returns parsed dict)
    data = await bridge.call_json(system_prompt="...", user_prompt="...", images_b64=[...])

    # Vision (returns plain text)
    text = await bridge.call_vision(system_prompt="...", user_prompt="...", images_b64=[...])
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import config

# ── OpenAI client (sync) is run in a thread so it doesn't block the event loop ──
# pyrefly: ignore [missing-import]
from openai import OpenAI


# ---------------------------------------------------------------------------
# Module-level singleton client (created once, reused for all calls)
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=config.NVIDIA_BASE_URL,
            api_key=config.NVIDIA_API_KEY,
            timeout=float(os.getenv("NVIDIA_REQUEST_TIMEOUT_SEC", "200")),
            max_retries=0,
        )
    return _client


# ---------------------------------------------------------------------------
# Low-level sync call (runs inside asyncio.to_thread)
# ---------------------------------------------------------------------------

def _call_sync(
    *,
    system_prompt: str,
    user_prompt: str,
    images_b64: List[str],
    max_tokens: int,
    timeout: float = 200.0,
    reasoning_effort: Optional[str] = None,
    stream: bool = False,
    model_name: Optional[str] = None,
) -> str:
    """
    Build an OpenAI-compatible chat request and return the raw text response.
    Images are embedded as data-URL inline content (same approach as test.py).
    """
    client = _get_client()
    selected_model = model_name or config.NVIDIA_MODEL
    if "llama-3.2" in selected_model.casefold() and len(images_b64) > 1:
        import base64, io
        from PIL import Image, ImageDraw
        panels = []
        for value in images_b64:
            with Image.open(io.BytesIO(base64.b64decode(value))) as image:
                panel = image.convert("RGB")
                panel.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                panels.append(panel)
        canvas = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels) + 32), "white")
        draw = ImageDraw.Draw(canvas)
        offset = 0
        for index, panel in enumerate(panels):
            draw.text((offset + 8, 8), f"IMAGE {index + 1}", fill="black")
            canvas.paste(panel, (offset, 32))
            offset += panel.width
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=90)
        images_b64 = [base64.b64encode(buffer.getvalue()).decode("ascii")]
        user_prompt += "\nAttached image combines the original images in numbered panels from left to right."
        print(f"[NVIDIA-IMAGE] Combined {len(panels)} images into one numbered image for Llama.")

    # ── Build message content ──────────────────────────────────────────────
    user_content: List[Dict[str, Any]] = []

    # Attach images first (vision models prefer images before the text)
    for b64 in images_b64:
        if b64:
            import base64
            header = base64.b64decode(b64)[:12]
            mime = "image/png" if header.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

    # Then the text prompt
    user_content.append({"type": "text", "text": user_prompt})

    # Llama models benefit from instructions anchored at the end of the user prompt,
    # ensuring the model maintains focus on JSON output even after lengthy screen payloads.
    selected_model = model_name or config.NVIDIA_MODEL
    if "llama" in selected_model.casefold() and system_prompt:
        tail_instruction = (
            f"\n\n[OUTPUT INSTRUCTION]\n{system_prompt}\n"
            "You MUST respond ONLY with a single JSON object. "
            "Do NOT include conversational prose, explanations, or multiple code blocks. Output raw JSON directly."
        )
        user_content[-1]["text"] = user_prompt + tail_instruction
        if images_b64:
            system_prompt = ""

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    # ── Call NVIDIA NIM ───────────────────────────────────────────────────
    request_model = model_name or config.NVIDIA_MODEL
    options = {}
    if "glm-5.3-flash" in request_model.casefold():
        # NVIDIA documents max as the default; simple screen decisions need low.
        import os
        effort = reasoning_effort or os.getenv("NVIDIA_REASONING_EFFORT", "low")
        options["extra_body"] = {"reasoning_effort": effort if effort in {"low", "high", "max"} else "low"}
    completion = client.chat.completions.create(
        model=request_model,
        messages=messages,
        temperature=0.2 if "llama-3.2" in request_model.casefold() else 1.00,
        top_p=1.0 if "llama-3.2" in request_model.casefold() else 0.01,
        max_tokens=max_tokens,
        stream=stream,
        timeout=timeout,
        **options,
    )

    if not stream:
        return (completion.choices[0].message.content or "").strip()
    import time
    started = time.monotonic()
    chunks = []
    first = True
    try:
        for event in completion:
            if first:
                print(f"[NVIDIA-STREAM] First response event received; model={request_model}.")
                first = False
            if not event.choices:
                continue
            choice = event.choices[0]
            text = getattr(choice.delta, "content", None)
            if text:
                chunks.append(text)
            if getattr(choice, "finish_reason", None) == "length":
                raise ValueError("Model response truncated at its token budget; incomplete JSON will not be accepted")
        print(f"[NVIDIA-STREAM] Response finished in {time.monotonic()-started:.1f}s; output characters={sum(map(len, chunks))}.")
        return "".join(chunks).strip()
    finally:
        close = getattr(completion, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# Public async bridge class  (same interface as FirebaseLLMBridge)
# ---------------------------------------------------------------------------

def _parse_json_response(raw):
    """Accept one complete JSON object, optionally surrounded by model prose."""
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("Model returned empty output")

    # 1. If markdown code fences exist, extract the JSON object from the code fence.
    # When the model reasons step-by-step and provides candidate/revised JSON blocks,
    # the last valid JSON dictionary represents the concluded answer.
    import re
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    matches = fence_pattern.findall(cleaned)
    if matches:
        for block in reversed(matches):
            b_strip = block.strip()
            start = b_strip.find("{")
            if start >= 0:
                try:
                    val, _ = json.JSONDecoder().raw_decode(b_strip[start:])
                    if isinstance(val, dict):
                        return val
                except (json.JSONDecodeError, ValueError):
                    pass

    # 2. Direct JSON or JSON surrounded by prose without code fences.
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Model response contains no JSON object")
    value, end = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("Model response must contain a JSON object")
    trailing = cleaned[start + end:]
    while "{" in trailing:
        extra_start = trailing.index("{")
        try:
            extra, extra_end = json.JSONDecoder().raw_decode(trailing[extra_start:])
        except json.JSONDecodeError:
            break
        if extra != value:
            raise ValueError("Model response contains conflicting JSON objects")
        trailing = trailing[extra_start + extra_end:]
    return value


class NvidiaBridge:
    """
    Async LLM bridge that calls NVIDIA NIM directly.

    All methods are coroutines and can be awaited from any async context.
    The underlying OpenAI client is synchronous but is always dispatched
    via asyncio.to_thread so it never blocks the event loop.
    """

    def __init__(
        self,
        *,
        max_tokens_text: int = 512,
        max_tokens_json: int = 4096,
        max_tokens_vision: int = 1024,
        reasoning_effort: Optional[str] = None,
        stream: bool = False,
        model_name: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._stream = stream
        self._reasoning_effort = reasoning_effort
        self._max_text   = max_tokens_text
        self._max_json   = max_tokens_json
        self._max_vision = max_tokens_vision

    # ------------------------------------------------------------------
    # call_text  — plain text in, plain text out
    # ------------------------------------------------------------------

    async def call_text(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: float = 200.0,
    ) -> str:
        """Plain text → text call."""
        return await asyncio.to_thread(
            _call_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=[],
            max_tokens=self._max_text,
            timeout=timeout,
            reasoning_effort=self._reasoning_effort,
            stream=self._stream,
            model_name=self._model_name,
        )

    # ------------------------------------------------------------------
    # call_vision  — text + images in, plain text out
    # ------------------------------------------------------------------

    async def call_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: List[str],
        timeout: float = 200.0,
    ) -> str:
        """Multimodal (text + images) call, returns plain text."""
        return await asyncio.to_thread(
            _call_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64,
            max_tokens=self._max_vision,
            timeout=timeout,
            reasoning_effort=self._reasoning_effort,
            stream=self._stream,
            model_name=self._model_name,
        )

    # ------------------------------------------------------------------
    # call_json  — text (+ optional images) in, parsed dict out
    # ------------------------------------------------------------------

    async def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: Optional[List[str]] = None,
        timeout: float = 200.0,
    ) -> Dict[str, Any]:
        """
        Call the model and parse the response as JSON.

        Strips optional markdown code fences (```json … ```) before parsing,
        matching the behaviour of FirebaseLLMBridge.call_json.
        """
        raw = await asyncio.to_thread(
            _call_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64 or [],
            max_tokens=self._max_json,
            timeout=timeout,
            reasoning_effort=self._reasoning_effort,
            stream=self._stream,
            model_name=self._model_name,
        )

        try:
            return _parse_json_response(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            # Preserve the response for a caller's bounded formatting recovery.
            exc.response_text = raw
            raise
