import requests, base64, io
from PIL import Image

BASE_URL = 'https://martin-hyetological-inherently.ngrok-free.dev'  # replace with your URL

with open('log/screenshots/human_exploration/human_exploration_step1_20260327_234139.png', 'rb') as f:
    resp = requests.post(
        f'{BASE_URL}/parse',
        files={'file': ('log/screenshots/human_exploration/human_exploration_step1_20260327_234139.png', f, 'image/png')},
        timeout=120
    )

data = resp.json()
print(f"{data['element_count']} elements found in {data['elapsed_seconds']}s")

# Save annotated image
img = Image.open(io.BytesIO(base64.b64decode(data['annotated_image'])))
img.save('annotated.png')

# Print elements
import json
print(json.dumps(data['elements'], indent=2))