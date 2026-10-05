"""Bounded waiting and progress for remote reads/inference (never device actions)."""
import math
import os
import threading
import time


def timeout_setting(name, default):
    try:
        value = float(os.getenv(name, str(default)))
        return value if math.isfinite(value) and value > 0 else float(default)
    except ValueError:
        return float(default)


def run_with_progress(label, operation, *, timeout, log=print, interval=5):
    """Discard a late read result after timeout; device actions must stay synchronous."""
    done = threading.Event()
    outcome = {}
    def worker():
        try:
            outcome['value'] = operation()
        except Exception as exc:
            outcome['error'] = exc
        finally:
            done.set()
    started = time.monotonic()
    log(f"[WAIT-START] {label}; deadline={timeout:.1f}s.")
    threading.Thread(target=worker, name=f"deployment-{label}", daemon=True).start()
    while True:
        remaining = timeout - (time.monotonic()-started)
        if remaining <= 0:
            log(f"[TIMEOUT] {label} exceeded {timeout:.1f}s; late result will not be used for actions.")
            raise TimeoutError(f"{label} exceeded {timeout:.1f}s")
        if done.wait(min(interval, remaining)):
            elapsed = time.monotonic()-started
            if 'error' in outcome:
                log(f"[WAIT-END] {label} failed after {elapsed:.1f}s: {outcome['error']}")
                raise outcome['error']
            log(f"[WAIT-END] {label} completed in {elapsed:.1f}s.")
            return outcome.get('value')
        log(f"[WAIT] {label} is still running; elapsed={time.monotonic()-started:.1f}s, remaining={max(0, timeout-(time.monotonic()-started)):.1f}s.")
