import firebase_admin
from firebase_admin import credentials, db
import base64
import uuid
import time
import json
import io
import os
import dotenv
from pathlib import Path
from PIL import Image, ImageGrab  # ImageGrab for screenshot


# ── Firebase init (deferred until first use) ─────────────────────────────────
dotenv.load_dotenv()

_firebase_initialized = False

def _ensure_firebase_initialized():
    """Initialize Firebase only once, on first use."""
    global _firebase_initialized
    if _firebase_initialized:
        return
    if firebase_admin._apps:
        _firebase_initialized = True
        return
    
    try:
        cred_path = Path(__file__).resolve().parent / "omniparser-queue-firebase-adminsdk-fbsvc-f46bd6f7ca.json"
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://omniparser-queue-default-rtdb.firebaseio.com/'
        })
        _firebase_initialized = True
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Firebase: {exc}")
# ── Screenshot → base64 ──────────────────────────────────────────────────────
def take_screenshot_base64():
    """Capture screen and return as base64 PNG string."""
    screenshot = ImageGrab.grab()          # full screen
    buf = io.BytesIO()
    screenshot.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ── Submit task to Firebase ──────────────────────────────────────────────────
def submit_task(image_b64: str, log_callback=None) -> str:
    """Write a pending task to Firebase. Returns the task_id."""
    task_id = str(uuid.uuid4())
    _ensure_firebase_initialized()
    db.reference(f'tasks/{task_id}').set({
        'status': 'pending',
        'image': image_b64,
        'created_at': time.time()
    })
    (log_callback or print)(f"[CLIENT] Task submitted: {task_id}")
    return task_id

# ── Poll for result ──────────────────────────────────────────────────────────
def wait_for_result(task_id: str, timeout: int = 120, poll_interval: float = 2.0, log_callback=None):
    """
    Poll Firebase until the result is ready or timeout is reached.
    Returns result dict or None on timeout.
    """
    deadline = time.time() + timeout
    next_report = time.time() + 10
    (log_callback or print)(f"[CLIENT] Waiting for result (timeout={timeout}s)...")
    _ensure_firebase_initialized()

    while time.time() < deadline:
        result = db.reference(f'results/{task_id}').get()
        if result and result.get('status') == 'done':
            (log_callback or print)(f"[CLIENT] Result received!")
            return result
        if result and result.get('status') in ('error', 'failed', 'cancelled'):
            (log_callback or print)(f"[PARSER] Worker stopped with status={result.get('status')}: {result.get('error', result.get('message', 'no details'))}")
            return None
        if time.time() >= next_report:
            (log_callback or print)(f"[PARSER] Job {task_id}: status={(result or {}).get('status', 'pending')}; remaining timeout={max(0, int(deadline-time.time()))}s")
            next_report = time.time() + 10
        time.sleep(poll_interval)

    (log_callback or print)(f"[CLIENT] Timed out after {timeout}s")
    return None

# ── Display result ───────────────────────────────────────────────────────────
def display_result(result: dict, task_id: str, log_callback=None) -> str:
    """
    Decode annotated image + save to disk. Returns path to saved JSON.
    
    Args:
        result: dict with 'annotated_image' (base64) and 'elements' (list)
        task_id: unique task identifier for naming output files
        
    Returns:
        str: absolute path to saved JSON file
    """
    img_dir = os.path.join("labeled_image", "img")
    json_dir = os.path.join("labeled_image", "json_labeled_data")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # ── Build timestamped base name for outputs ─────────────────────────────-
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int(time.time() * 1000) % 1000
    base_name = f"{ts}_{ms:03d}"

    # ── Save annotated image ─────────────────────────────────────────────────
    img_b64 = result.get('annotated_image')
    if img_b64:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
            img_path = os.path.join(img_dir, f"{base_name}.png")
            img.save(img_path)
            (log_callback or print)(f"[CLIENT] Image saved → {img_path}")
        except Exception as exc:
            (log_callback or print)(f"[CLIENT] Warning: Image save failed: {exc}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    elements = result.get('elements', [])
    json_path = os.path.join(json_dir, f"{base_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(elements, f, indent=2, ensure_ascii=False)
    (log_callback or print)(f"[CLIENT] JSON saved  → {json_path}")
    (log_callback or print)(f"[CLIENT] {len(elements)} elements found")
    
    return os.path.abspath(json_path)

# ── Main flow (for standalone testing) ──────────────────────────────────────
def run(image_b64: str = None, log_callback=None) -> str:
    """
    End-to-end workflow: submit image → wait for result → save outputs.
    
    Args:
        image_b64: base64 encoded image. If None, captures screenshot.
        
    Returns:
        str: absolute path to saved JSON file, or empty string on error.
    """
    # Use provided image or capture from screen
    if image_b64 is None:
        (log_callback or print)("[CLIENT] Taking screenshot...")
        image_b64 = take_screenshot_base64()

    task_id = submit_task(image_b64, log_callback=log_callback)
    result = wait_for_result(task_id, log_callback=log_callback)

    if result:
        json_path = display_result(result, task_id, log_callback=log_callback)
        # Clean up task from Firebase
        _ensure_firebase_initialized()
        db.reference(f'tasks/{task_id}').delete()
        return json_path
    else:
        (log_callback or print)("[CLIENT] No result received.")
        return ""


if __name__ == '__main__':
    run()