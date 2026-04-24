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
def submit_task(image_b64: str) -> str:
    """Write a pending task to Firebase. Returns the task_id."""
    task_id = str(uuid.uuid4())
    _ensure_firebase_initialized()
    db.reference(f'tasks/{task_id}').set({
        'status': 'pending',
        'image': image_b64,
        'created_at': time.time()
    })
    print(f"[CLIENT] Task submitted: {task_id}")
    return task_id

# ── Poll for result ──────────────────────────────────────────────────────────
def wait_for_result(task_id: str, timeout: int = 120, poll_interval: float = 2.0):
    """
    Poll Firebase until the result is ready or timeout is reached.
    Returns result dict or None on timeout.
    """
    deadline = time.time() + timeout
    print(f"[CLIENT] Waiting for result (timeout={timeout}s)...")
    _ensure_firebase_initialized()

    while time.time() < deadline:
        result = db.reference(f'results/{task_id}').get()
        if result and result.get('status') == 'done':
            print(f"[CLIENT] Result received!")
            return result
        time.sleep(poll_interval)

    print(f"[CLIENT] Timed out after {timeout}s")
    return None

# ── Display result ───────────────────────────────────────────────────────────
def display_result(result: dict, task_id: str) -> str:
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

    # ── Save annotated image ─────────────────────────────────────────────────
    img_b64 = result.get('annotated_image')
    if img_b64:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
            img_path = os.path.join(img_dir, f"{task_id}.png")
            img.save(img_path)
            print(f"[CLIENT] Image saved → {img_path}")
        except Exception as exc:
            print(f"[CLIENT] Warning: Image save failed: {exc}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    elements = result.get('elements', [])
    json_path = os.path.join(json_dir, f"{task_id}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(elements, f, indent=2, ensure_ascii=False)
    print(f"[CLIENT] JSON saved  → {json_path}")
    print(f"[CLIENT] {len(elements)} elements found")
    
    return os.path.abspath(json_path)

# ── Main flow (for standalone testing) ──────────────────────────────────────
def run(image_b64: str = None) -> str:
    """
    End-to-end workflow: submit image → wait for result → save outputs.
    
    Args:
        image_b64: base64 encoded image. If None, captures screenshot.
        
    Returns:
        str: absolute path to saved JSON file, or empty string on error.
    """
    # Use provided image or capture from screen
    if image_b64 is None:
        print("[CLIENT] Taking screenshot...")
        image_b64 = take_screenshot_base64()

    task_id = submit_task(image_b64)
    result = wait_for_result(task_id)

    if result:
        json_path = display_result(result, task_id)
        # Clean up task from Firebase
        _ensure_firebase_initialized()
        db.reference(f'tasks/{task_id}').delete()
        return json_path
    else:
        print("[CLIENT] No result received.")
        return ""


if __name__ == '__main__':
    run()