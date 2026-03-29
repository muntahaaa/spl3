# Human Explorer

A 3-step pipeline for recording, parsing, and storing human UI-exploration
sessions on Android devices.

```
┌──────────────────────────────────────────────────────────────┐
│  STEP 1  –  Human Exploration  (Gradio UI + ADB)             │
│  • Initialise device & task                                   │
│  • Perform actions  →  screenshot saved automatically         │
│  • Insert parsed result via  POST /api/insert_parsed_result   │
│  • Repeat until task complete                                 │
├──────────────────────────────────────────────────────────────┤
│  STEP 2  –  Save State  (Gradio "Stop & save" button)         │
│  • Serialises State → ./log/json_state/state_<ts>.json        │
├──────────────────────────────────────────────────────────────┤
│  STEP 3  –  Push to Databases  (Gradio "Store to DB" tab)     │
│  • Reads JSON  →  Neo4j (graph) + Pinecone (vectors)          │
└──────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
human_explorer/
├── main.py              ← entry point (FastAPI + Gradio)
├── config.py            ← DB credentials, paths
├── state_manager.py     ← thread-safe shared session state
├── api_routes.py        ← REST endpoints
├── ui.py                ← Gradio layout
├── explor_human.py      ← ADB action + screenshot logic  (Step 1)
├── data/
│   ├── State.py         ← TypedDict definition
│   ├── data_storage.py  ← state2json + json2db  (Steps 2 & 3)
│   ├── graph_db.py      ← Neo4j adapter
│   └── vector_db.py     ← Pinecone adapter
├── tool/
│   ├── adb_tools.py     ← ADB wrappers (no parsing tool)
│   └── img_tool.py      ← element crop + feature-extraction client
├── log/
│   ├── screenshots/     ← raw + labeled PNGs
│   └── json_state/      ← serialised session JSON files
└── requirements.txt
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 + | Tested on 3.11 |
| ADB (Android Debug Bridge) | `sudo apt install adb` / install Android SDK |
| Android device or emulator | USB debugging enabled |
| Neo4j (local or remote) | Free Community Edition works |
| Pinecone account | Free tier is enough for testing |
| Feature-extraction service | Your own CPU-based ResNet50 REST service on port 8001 |

---

## Installation

```bash
# 1. Clone / copy the project
cd human_explorer

# 2. Create a virtual environment (CPU-only)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies (CPU only – no CUDA needed)
pip install -r requirements.txt

# 4. Configure credentials
#    Open config.py and fill in:
#      Neo4j_URI, Neo4j_AUTH
#      PINECONE_API_KEY
#      Feature_URI   (your ResNet50 service base URL)
```

---

## Running the project

```bash
python main.py
```

This starts **one** Uvicorn server on port **7860** that serves both:

| URL | Purpose |
|---|---|
| `http://localhost:7860/` | Gradio UI |
| `http://localhost:7860/api/` | REST API |
| `http://localhost:7860/docs` | Swagger / interactive API docs |

---

## Step-by-step usage

### STEP 1 – Human exploration

**In the Gradio UI:**

1. Open `http://localhost:7860`
2. Go to **① Initialization** tab
   - Click **Refresh devices** → select your device
   - Enter a task description → click **Initialize**
3. Go to **② Exploration** tab
   - Click **▶ Start session** — the first screenshot is taken and saved
   - Check the log for the screenshot path

**After each screenshot (including the initial one):**

4. Run your own parsing tool on the screenshot PNG
5. Call the API to insert the parsed result (see below)
6. Now choose an action in the UI and click **⚡ Perform action**
7. Repeat steps 4–6 for every subsequent screenshot

---

### Inserting a parsed result (API)

After your parsing tool produces two files, POST them to the API:

**Endpoint:**  `POST http://localhost:7860/api/insert_parsed_result`

**Request body (JSON):**

```json
{
  "labeled_image_path": "./log/screenshots/human_exploration/processed/labeled_human_exploration_step_1_20240101_120000.png",
  "parsed_content_json_path": "./log/screenshots/human_exploration/processed/human_exploration_step_1_20240101_120000.json"
}
```

**curl example:**

```bash
curl -X POST http://localhost:7860/api/insert_parsed_result \
     -H "Content-Type: application/json" \
     -d '{
       "labeled_image_path": "./log/screenshots/human_exploration/processed/labeled_human_exploration_step_1_20240101_120000.png",
       "parsed_content_json_path": "./log/screenshots/human_exploration/processed/human_exploration_step_1_20240101_120000.json"
     }'
```

**Python example:**

```python
import requests

requests.post(
    "http://localhost:7860/api/insert_parsed_result",
    json={
        "labeled_image_path": "./log/screenshots/human_exploration/processed/labeled_human_exploration_step_1_20240101_120000.png",
        "parsed_content_json_path": "./log/screenshots/human_exploration/processed/human_exploration_step_1_20240101_120000.json",
    },
)
```

---

### Parsed elements JSON format

Your parsing tool must produce a JSON file that is an **array** of element
objects.  Each object must have these fields:

```json
[
  {
    "ID":      1,
    "bbox":    [0.05, 0.10, 0.90, 0.18],
    "type":    "text",
    "content": "Welcome Screen"
  },
  {
    "ID":      2,
    "bbox":    [0.15, 0.45, 0.85, 0.55],
    "type":    "button",
    "content": "Sign In"
  },
  {
    "ID":      3,
    "bbox":    [0.10, 0.60, 0.90, 0.70],
    "type":    "input",
    "content": "Username field"
  }
]
```

| Field | Type | Description |
|---|---|---|
| `ID` | integer | Unique element identifier within this screen |
| `bbox` | `[x1, y1, x2, y2]` | Relative coordinates (0.0 – 1.0). Top-left = (x1, y1), bottom-right = (x2, y2) |
| `type` | string | Element type: `button`, `text`, `input`, `image`, `icon`, `checkbox`, `list_item`, … |
| `content` | string | Visible text or description of the element |

> **bbox note:**  values are *relative* to the screen dimensions.  
> Example: a button occupying the middle 80 % of the screen at 45 % height  
> → `[0.10, 0.44, 0.90, 0.50]`

---

### Checking session status

```bash
curl http://localhost:7860/api/session/status
```

Response:

```json
{
  "has_session":         true,
  "step":                3,
  "device":              "emulator-5554",
  "task":                "Navigate to Settings",
  "parsed_result_ready": false,
  "pending_screenshot":  "./log/screenshots/human_exploration/human_exploration_step_4_20240101_120000.png",
  "history_count":       3
}
```

`parsed_result_ready: false` means you must POST the parsed result before
performing an element-based action.

---

### STEP 2 – Save session to JSON

In the Gradio **② Exploration** tab, click **🛑 Stop & save to JSON**.

The file is written to `./log/json_state/state_<timestamp>.json`.

---

### STEP 3 – Push to databases

In the Gradio **③ Store to Neo4j + Pinecone** tab:

1. Paste the path to your saved JSON state file
2. Click **🚀 Store to databases**

This reads the JSON and creates:
- **Neo4j nodes:**  `Page`  and  `Element`
- **Neo4j relationships:**  `(Page)-[:HAS_ELEMENT]->(Element)`  and  `(Element)-[:LEADS_TO]->(Page)`
- **Pinecone vectors:**  one ResNet50 embedding per tapped element

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No devices found` | Check USB cable, enable USB debugging, run `adb devices` in terminal |
| `RuntimeError: No active session` | Call **Initialize** in the UI before inserting parsed results |
| `JSON file not found` | The path in `parsed_content_json_path` must be accessible from the machine running the server |
| Neo4j connection error | Ensure Neo4j is running and `Neo4j_URI` / `Neo4j_AUTH` in `config.py` are correct |
| Pinecone error | Check `PINECONE_API_KEY` in `config.py` |
| Feature service unreachable | Start your ResNet50 service and update `Feature_URI` in `config.py` |
