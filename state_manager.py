# ─────────────────────────────────────────────
#  state_manager.py  –  Thread-safe shared state
# ─────────────────────────────────────────────
"""
A single SessionManager instance is imported wherever the live
session state is needed (Gradio UI callbacks, FastAPI routes, etc.).
"""

import threading
from typing import Optional


class SessionManager:
    """Holds the single active exploration session."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state: Optional[dict] = None
        self.user_log_storage: list = []
        self.user_page_storage: list = []          # labeled image paths for the gallery

    # ── state access ──────────────────────────

    def get_state(self) -> Optional[dict]:
        with self._lock:
            return self._state

    def set_state(self, state: dict) -> None:
        with self._lock:
            self._state = state

    def clear(self) -> None:
        with self._lock:
            self._state = None
            self.user_log_storage = []
            self.user_page_storage = []

    # ── parsed-result insertion ────────────────

    def insert_parsed_result(
        self,
        labeled_image_path: str,
        parsed_content_json_path: str,
    ) -> dict:
        """
        Called by the FastAPI endpoint after the user has run their own
        parsing tool.  Updates the live state so the next action can
        resolve element coordinates.

        Returns a summary dict (used as the API response body).
        """
        with self._lock:
            if self._state is None:
                raise RuntimeError("No active session.  Initialize a device first.")

            prev_json = self._state.get("current_page_json")

            # Update current page pointers
            self._state["current_page_json"] = parsed_content_json_path

            # Add labeled image to gallery history (avoid duplicates)
            if labeled_image_path and labeled_image_path not in self._state["page_history"]:
                self._state["page_history"].append(labeled_image_path)
                self.user_page_storage = list(self._state["page_history"])

            return {
                "status": "ok",
                "step": self._state.get("step", 0),
                "screenshot": self._state.get("current_page_screenshot"),
                "previous_json": prev_json,
                "new_json": parsed_content_json_path,
                "labeled_image": labeled_image_path,
            }

    # ── convenience helpers ───────────────────

    def is_ready_for_action(self) -> bool:
        """True when the current screenshot has been parsed."""
        with self._lock:
            if self._state is None:
                return False
            return self._state.get("current_page_json") is not None

    def pending_screenshot(self) -> Optional[str]:
        """Returns the screenshot path that still awaits a parsed result, or None."""
        with self._lock:
            if self._state is None:
                return None
            if self._state.get("current_page_json") is None:
                return self._state.get("current_page_screenshot")
            return None


# ── module-level singleton ─────────────────────
session = SessionManager()
