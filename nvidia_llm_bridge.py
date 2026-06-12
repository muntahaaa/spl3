"""
nvidia_llm_bridge.py
--------------------
Drop-in async replacement for firebase_llm_bridge.py.

Instead of pushing tasks to Firebase and waiting for a Colab worker,
this module calls the NVIDIA NIM API directly using the OpenAI-compatible
client — exactly as demonstrated in test.py.

Model: nvidia/llama-3.1-nemotron-nano-vl-8b-v1
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
) -> str:
    """
    Build an OpenAI-compatible chat request and return the raw text response.
    Images are embedded as data-URL inline content (same approach as test.py).
    """
    client = _get_client()

    # ── Build message content ──────────────────────────────────────────────
    user_content: List[Dict[str, Any]] = []

    # Attach images first (vision models prefer images before the text)
    for b64 in images_b64:
        if b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    # Then the text prompt
    user_content.append({"type": "text", "text": user_prompt})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    # ── Call NVIDIA NIM ───────────────────────────────────────────────────
    completion = client.chat.completions.create(
        model=config.NVIDIA_MODEL,
        messages=messages,
        temperature=1.00,
        top_p=0.01,
        max_tokens=max_tokens,
        stream=False,
    )

    return (completion.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Public async bridge class  (same interface as FirebaseLLMBridge)
# ---------------------------------------------------------------------------

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
        max_tokens_json: int = 1024,
        max_tokens_vision: int = 1024,
    ) -> None:
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
        timeout: float = 300.0,        # kept for API compat; not used directly
    ) -> str:
        """Plain text → text call."""
        return await asyncio.to_thread(
            _call_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=[],
            max_tokens=self._max_text,
        )

    # ------------------------------------------------------------------
    # call_vision  — text + images in, plain text out
    # ------------------------------------------------------------------

    async def call_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: List[str],
        timeout: float = 300.0,
    ) -> str:
        """Multimodal (text + images) call, returns plain text."""
        return await asyncio.to_thread(
            _call_sync,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64,
            max_tokens=self._max_vision,
        )

    # ------------------------------------------------------------------
    # call_json  — text (+ optional images) in, parsed dict out
    # ------------------------------------------------------------------

    async def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: Optional[List[str]] = None,
        timeout: float = 300.0,
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
        )

        cleaned = raw.strip()
        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        if not cleaned:
            raise ValueError(f"Model returned empty output. Raw: {raw!r}")

        return json.loads(cleaned)
