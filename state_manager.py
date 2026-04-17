"""
state_manager.py  –  Thread-safe shared session state
======================================================

BUG-3 FIX
──────────
insert_parsed_result() now accepts THREE parameters instead of two:
    labeled_image_path        → local path of labeled PNG
    parsed_content_json_path  → local path of downloaded JSON
    parsed_content_json_url   → original Cloudinary / cloud URL of the JSON

It stores:
    state["current_page_json"]      = local temp path  (used for live coord lookups)
    state["current_page_json_url"]  = cloud URL        (stored in history_steps as source_json)

This means:
  • element_number_to_coords() can open the file immediately (local path)
  • single_human_explor() records the portable cloud URL in history_steps
  • json2db() has both the local path (fast) and the URL (fallback)
"""

import threading
from typing import Optional


class SessionManager:
    """Holds the single active exploration session."""

    def __init__(self):
        self._lock            = threading.Lock()
        self._state: Optional[dict] = None
        self.user_log_storage: list = []
        self.user_page_storage: list = []

    # ── state access ──────────────────────────────────────────────────────────

    def get_state(self) -> Optional[dict]:
        with self._lock:
            return self._state

    def set_state(self, state: dict) -> None:
        with self._lock:
            self._state = state

    def clear(self) -> None:
        with self._lock:
            self._state            = None
            self.user_log_storage  = []
            self.user_page_storage = []

    # ── parsed-result insertion ───────────────────────────────────────────────

    def insert_parsed_result(
        self,
        labeled_image_path: str,
        parsed_content_json_path: str,
        parsed_content_json_url: str = "",   # BUG-3 FIX: new parameter
    ) -> dict:
        """
        Called after the user supplies cloud URLs and the files have been
        downloaded locally.

        Parameters
        ──────────
        labeled_image_path
            Local path to the downloaded labeled PNG.
        parsed_content_json_path
            Local path to the downloaded elements JSON.
        parsed_content_json_url
            Original cloud URL of the elements JSON.  This is stored in
            state["current_page_json_url"] and later written into
            history_steps["source_json"] for portability.

        Returns a summary dict used as the API response body.
        """
        with self._lock:
            if self._state is None:
                raise RuntimeError("No active session. Initialize a device first.")

            prev_json = self._state.get("current_page_json")

            # Local path → used by element_number_to_coords() immediately
            self._state["current_page_json"] = parsed_content_json_path

            # BUG-3 FIX: also store the cloud URL so history_steps can reference it
            self._state["current_page_json_url"] = parsed_content_json_url or parsed_content_json_path

            # Add labeled image to gallery (avoid duplicates)
            if labeled_image_path and labeled_image_path not in self._state["page_history"]:
                self._state["page_history"].append(labeled_image_path)
                self.user_page_storage = list(self._state["page_history"])

            return {
                "status":       "ok",
                "step":         self._state.get("step", 0),
                "screenshot":   self._state.get("current_page_screenshot"),
                "previous_json": prev_json,
                "new_json":     parsed_content_json_path,
                "new_json_url": self._state["current_page_json_url"],
                "labeled_image": labeled_image_path,
            }

    # ── convenience helpers ───────────────────────────────────────────────────

    def is_ready_for_action(self) -> bool:
        """True when the current screenshot has been parsed."""
        with self._lock:
            if self._state is None:
                return False
            return self._state.get("current_page_json") is not None

    def pending_screenshot(self) -> Optional[str]:
        """Returns the screenshot path awaiting a parsed result, or None."""
        with self._lock:
            if self._state is None:
                return None
            if self._state.get("current_page_json") is None:
                return self._state.get("current_page_screenshot")
            return None


# ── module-level singleton ────────────────────────────────────────────────────
session = SessionManager()