"""
firebase_llm_bridge.py
----------------------
Drop-in async helper that replaces direct Gemini API calls with a
Firebase Realtime Database request/response round-trip.

Flow
----
  1. Caller pushes a task dict to  /llm_tasks/{task_id}
  2. Colab worker picks it up, runs Qwen2.5-VL-7B-Instruct, writes
     result to /llm_results/{task_id}
  3. This module polls for the result, returns it, then deletes both
     the task and result nodes so nothing lingers in Firebase.

Usage
-----
    from firebase_llm_bridge import FirebaseLLMBridge

    bridge = FirebaseLLMBridge(
        firebase_url="https://<project>.firebaseio.com",
        firebase_secret="<db-secret-or-service-account-token>",
    )
    result_text = await bridge.call(
        task_type="text",          # "text" | "vision" | "json"
        system_prompt="You are …",
        user_prompt="Analyse …",
        images_b64=[],             # list of base64 strings (optional)
        timeout=300,               # seconds to wait for Colab
    )
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional
from google.oauth2 import service_account
import google.auth.transport.requests

import aiohttp


# ---------------------------------------------------------------------------
# Tiny async Firebase REST client (no SDK needed on the codebase side)
# ---------------------------------------------------------------------------

class _FirebaseREST:
    """Minimal async wrapper around Firebase Realtime DB REST API."""

    def __init__(self, base_url: str, secret: str) -> None:
        # base_url like "https://myproject-default-rtdb.firebaseio.com"
        self._base = base_url.rstrip("/")
        self._auth = secret
        self._creds = service_account.Credentials.from_service_account_file(
            "appagent-chain-firebase-adminsdk-fbsvc-8465429f1d.json",
            scopes=["https://www.googleapis.com/auth/firebase.database",
                "https://www.googleapis.com/auth/userinfo.email"]
)

    def _url(self, path: str) -> str:
        req = google.auth.transport.requests.Request()
        self._creds.refresh(req)
        return f"{self._base}/{path.lstrip('/')}.json?access_token={self._creds.token}"

    async def put(self, path: str, data: Any) -> Any:
        async with aiohttp.ClientSession() as s:
            async with s.put(self._url(path), json=data) as r:
                r.raise_for_status()
                return await r.json()

    async def get(self, path: str) -> Any:
        async with aiohttp.ClientSession() as r_session:
            async with r_session.get(self._url(path)) as r:
                r.raise_for_status()
                return await r.json()

    async def delete(self, path: str) -> None:
        async with aiohttp.ClientSession() as s:
            async with s.delete(self._url(path)) as r:
                r.raise_for_status()


# ---------------------------------------------------------------------------
# Public bridge class
# ---------------------------------------------------------------------------

class FirebaseLLMBridge:
    """
    Sends LLM requests to Firebase and retrieves results produced by
    the Colab Qwen2.5-VL worker.
    """

    TASKS_PATH = "llm_tasks"
    RESULTS_PATH = "llm_results"

    def __init__(self, firebase_url: str, firebase_secret: str) -> None:
        self._fb = _FirebaseREST(firebase_url, firebase_secret)

    # ------------------------------------------------------------------
    # Core async call
    # ------------------------------------------------------------------

    async def call(
        self,
        *,
        task_type: str = "text",          # "text" | "vision" | "json"
        system_prompt: str = "",
        user_prompt: str = "",
        images_b64: Optional[List[str]] = None,   # raw base64 strings
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> str:
        """
        Submit an inference task and block until the result arrives.

        Returns the model's text output as a plain string.
        Raises TimeoutError if the Colab worker doesn't respond in time.
        Raises RuntimeError on model-side errors.
        """
        task_id = str(uuid.uuid4())
        print(f"  [bridge] Submitting task {task_id[:8]} type={task_type} "
          f"images={len(images_b64 or [])} prompt_len={len(user_prompt)}")

        task_payload: Dict[str, Any] = {
            "task_id": task_id,
            "task_type": task_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "images_b64": images_b64 or [],
            "created_at": time.time(),
            "status": "pending",
        }

        # 1. Push task
        await self._fb.put(f"{self.TASKS_PATH}/{task_id}", task_payload)
        print(f"  [bridge] Task {task_id[:8]} pushed to Firebase, polling for result...")
        # 2. Poll for result
        deadline = time.time() + timeout
        polls = 0
        while time.time() < deadline:
            await asyncio.sleep(poll_interval)
            polls += 1
            if polls % 15 == 0:  # print every 30 seconds
                elapsed = polls * poll_interval
                print(f"  [bridge] Still waiting for {task_id[:8]}... ({elapsed:.0f}s elapsed)")

            result_node = await self._fb.get(f"{self.RESULTS_PATH}/{task_id}")
            if result_node is None:
                continue

            print(f"  [bridge] Result received for {task_id[:8]} after {polls * poll_interval:.0f}s")
            await asyncio.gather(
                self._fb.delete(f"{self.TASKS_PATH}/{task_id}"),
                self._fb.delete(f"{self.RESULTS_PATH}/{task_id}"),
                return_exceptions=True,
            )

            if result_node.get("status") == "error":
                raise RuntimeError(
                    f"Colab worker error for task {task_id}: "
                    f"{result_node.get('error', 'unknown')}"
                )

            return result_node.get("output", "")

        await self._fb.delete(f"{self.TASKS_PATH}/{task_id}")
        raise TimeoutError(
            f"No response from Colab worker within {timeout}s (task_id={task_id})"
        )

    # ------------------------------------------------------------------
    # Convenience wrappers matching Gemini usage patterns
    # ------------------------------------------------------------------

    async def call_text(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: float = 300.0,
    ) -> str:
        """Plain text → text call."""
        return await self.call(
            task_type="text",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout=timeout,
        )

    async def call_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: List[str],
        timeout: float = 300.0,
    ) -> str:
        """Multimodal (text + images) call."""
        return await self.call(
            task_type="vision",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64,
            timeout=timeout,
        )

    async def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images_b64: Optional[List[str]] = None,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        raw = await self.call(
            task_type="json",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images_b64=images_b64 or [],
            timeout=timeout,
        )

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]                      # drop leading ```
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]                  # drop optional "json"
            cleaned = cleaned.strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        if not cleaned:
            raise ValueError(f"Model returned empty output. Raw response was: {raw!r}")

        return json.loads(cleaned)