"""Track and remove only images produced by one successful deployment run."""
from pathlib import Path
import threading


class DeploymentImages:
    def __init__(self, log=print, workspace=None):
        root = Path(workspace or Path.cwd()).resolve()
        self.roots = [(root / "log/screenshots/deployment").resolve(),
                      (root / "labeled_image/img").resolve(),
                      (root / "labeled_image/json_labeled_data").resolve()]
        self.paths = set()
        self.removed = 0
        self.errors = []
        self.completed = False
        self.lock = threading.Lock()
        self.log = log

    def track_parser_message(self, message):
        for prefix in ("[CLIENT] Image saved", "[CLIENT] JSON saved"):
            if message.startswith(prefix):
                value = message[len(prefix):].strip()
                # Parser emits a Unicode arrow; strip the separator, not path characters.
                value = value.lstrip(chr(0x2192) + "?->: ").strip()
                if value:
                    self.track(value)
                return

    def track(self, value):
        path = Path(value).resolve()
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".json", ".xml"} or not any(path.is_relative_to(root) for root in self.roots):
            self.log(f"[CLEANUP] Ignoring image outside deployment output directories: {path}")
            return
        with self.lock:
            self.paths.add(path)
            if self.completed:
                self._remove(path)  # Late parser output after a read timeout.

    def _remove(self, path):
        try:
            if path.is_file():
                path.unlink()
                self.removed += 1
                self.log(f"[CLEANUP] Deleted execution image: {path}")
        except OSError as exc:
            self.errors.append(str(path))
            self.log(f"[CLEANUP] Could not delete {path}: {exc}")

    def finish(self, success):
        with self.lock:
            if success:
                self.completed = True
                for path in sorted(self.paths):
                    self._remove(path)
            else:
                self.log(f"[CLEANUP] Task not completed; retaining {len(self.paths)} execution image(s).")
            return {"deleted": self.removed, "errors": list(self.errors), "retained_on_failure": not success}
