"""Run-scoped user assistance and device locks for deployment."""
import threading
import uuid

_registry_lock = threading.RLock()
_requests = {}
_device_locks = {}


def device_lock(device):
    """Share one execution lock per device across UI workers."""
    with _registry_lock:
        return _device_locks.setdefault(str(device), threading.Lock())


def deployment_active():
    with _registry_lock:
        return any(lock.locked() for lock in _device_locks.values())


class Assistance:
    def __init__(self):
        self.token = uuid.uuid4().hex
        self._condition = threading.Condition()
        self._pending = None
        self._answer = None
        self._closed = False
        with _registry_lock:
            _requests[self.token] = self

    @property
    def pending(self):
        with self._condition:
            return dict(self._pending) if self._pending is not None else None

    def request(self, context):
        with self._condition:
            if self._closed:
                return {"skip": True, "info": ""}
            self._pending = dict(context)
            self._answer = None
            while self._answer is None and not self._closed:
                self._condition.wait()
            answer = self._answer or {"skip": True, "info": ""}
            self._pending = None
            self._answer = None
            return answer

    def answer(self, info="", skip=False):
        info = str(info or "").strip()
        with self._condition:
            if self._closed or self._pending is None or self._answer is not None:
                return "This assistance request is no longer active."
            if not skip and not info:
                return "Provide more information before continuing, or skip the task."
            self._answer = {"skip": bool(skip), "info": info}
            self._condition.notify_all()
            return "Task skipped." if skip else "Information received; execution will continue."

    def close(self):
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        with _registry_lock:
            _requests.pop(self.token, None)


def answer_request(token, info="", skip=False):
    with _registry_lock:
        request = _requests.get(token)
    if request is None:
        return "This assistance request is no longer active."
    return request.answer(info, skip)
