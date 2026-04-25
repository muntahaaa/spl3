import asyncio
import time
from collections import deque

# Free-tier protection: cap LLM requests to 4 per minute.
MAX_REQUESTS_PER_MINUTE = 4
_WINDOW_SECONDS = 60.0
_timestamps = deque()
_lock = asyncio.Lock()


async def wait_for_llm_slot(max_requests_per_minute: int = MAX_REQUESTS_PER_MINUTE) -> None:
    """Block until a request slot is available under the per-minute limit."""
    window = _WINDOW_SECONDS

    while True:
        async with _lock:
            now = time.monotonic()

            # Drop timestamps outside the rolling window.
            while _timestamps and now - _timestamps[0] >= window:
                _timestamps.popleft()

            if len(_timestamps) < max_requests_per_minute:
                _timestamps.append(now)
                return

            # Need to wait until the oldest request leaves the window.
            sleep_for = window - (now - _timestamps[0]) + 0.01

        await asyncio.sleep(max(0.01, sleep_for))
