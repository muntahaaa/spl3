# Android App Task Explorer

An end-to-end pipeline for recording, parsing, storing, and reasoning over
human UI-exploration sessions on Android devices.

```
┌──────────────────────────────────────────────────────────────┐
│  STEP 1  –  Exploration (Gradio UI + ADB + OmniParser)       │
│  • Initialise device & task                                   │
│  • Perform actions → screenshot saved automatically           │
│  • OmniParser runs automatically per screenshot               │
│  • Labeled image + elements JSON saved locally                │
├──────────────────────────────────────────────────────────────┤
│  STEP 2  –  Save State  (Gradio "Stop & save" button)         │
│  • Serialises State → ./log/json_state/state_<ts>.json        │
├──────────────────────────────────────────────────────────────┤
│  STEP 3  –  Push to Databases  (Gradio "Store to DB" tab)     │
│  • Reads JSON → Neo4j (graph) + Pinecone (vectors)            │
├──────────────────────────────────────────────────────────────┤
│  STEP 4  –  Chain Processing  (Gradio tab)                    │
│  • chain_understand: triplet reasoning + description updates  │
│  • chain_evolve: optional high-level action synthesis         │
├──────────────────────────────────────────────────────────────┤
│  STEP 5  –  Action Execution (Deployment engine)              │
│  • structured execution + reactive fallback                   │
└──────────────────────────────────────────────────────────────┘
```

---

## Project structure

```
spl3/
├── main.py                ← entry point (FastAPI + Gradio)
├── config.py              ← DB credentials, paths
├── firebase_llm_bridge.py ← Firebase RTDB async LLM bridge
├── chain_understand.py    ← triplet reasoning + graph enrichment
├── chain_evolve.py        ← high-level action synthesis
├── deployment.py          ← execution workflow runner
├── state_manager.py       ← thread-safe shared session state
├── ui.py                  ← Gradio layout
├── api/
│   ├── api_routes.py      ← REST endpoints
│   └── chain_routes.py    ← chain job endpoints
├── chain/
│   ├── chain_models.py    ← job response models
│   ├── chain_service.py   ← async chain workers
│   └── task_store.py      ← in-memory job store
├── OmniParser/
│   └── client.py          ← Firebase queue client (auto-parse)
│   └── omniparser-queue.ipynb ← OmniParser worker notebook
├── explor_human.py        ← ADB action + screenshot logic
├── explore_auto.py        ← automated exploration
├── data/
│   ├── State.py           ← TypedDict definition
│   ├── data_storage.py    ← state2json + json2db
│   ├── graph_db.py        ← Neo4j adapter
│   └── vector_db.py       ← Pinecone adapter
├── tool/
│   ├── adb_tools.py       ← ADB wrappers
│   └── img_tool.py        ← element crop + feature-extraction client
├── log/
│   ├── screenshots/       ← raw screenshots
│   └── json_state/        ← serialised session JSON files
├── labeled_image/
│   ├── img/               ← OmniParser labeled PNGs
│   └── json_labeled_data/ ← OmniParser elements JSON
├── qwen_firebase_worker.ipynb ← Qwen2.5-VL Firebase worker (Colab)
├── verify_pipeline.py     ← end-to-end pipeline validation
├── context_chain.md       ← chain_understand/chain_evolve design notes
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
#      CHAIN_FIREBASE_URL
#      FIREBASE_SECRET  (or service-account-based access token flow)
```

---

## Chain reasoning and evolution context

### Overview

ChainEvolve is a graph-driven multimodal UI reasoning system that observes UI interaction flows, understands their semantic meaning with an LLM, and converts repeated low-level UI operations into reusable high-level task abstractions.

It combines:

- Neo4j for structured UI memory
- Firebase as an asynchronous inference bridge
- Multimodal LLM reasoning using screenshots + textual UI metadata
- Graph-based workflow abstraction and procedural memory generation

### Core idea

Instead of storing raw clicks and page transitions only, the system learns:

- what each interaction means
- what the user is trying to accomplish
- how UI states change
- how multiple low-level steps form reusable workflows

Example high-level actions:

```text
Create Alarm
Login to Application
Send Email
Search Contact
```

### System architecture

```text
UI Interaction Recording
  ↓
Graph Storage (Neo4j)
  ↓
Multimodal LLM Reasoning
  ↓
Semantic Graph Enrichment
  ↓
Workflow Abstraction
  ↓
Reusable High-Level Actions
```

### Graph structure

```text
(Page)-[:HAS_ELEMENT]->(Element)-[:LEADS_TO]->(Page)
```

### Chain retrieval

The system retrieves a navigation chain starting from a page:

```python
chain = db.get_chain_from_start(start_page_id)
```

A chain consists of triplets:

```python
{
  source_page,
  element,
  target_page,
  action
}
```

### Multimodal UI understanding (chain_understand.py)

Inputs:

- Source page description
- Target page description
- UI element metadata
- Screenshots of pages

Processing:

- screenshots are loaded, resized, converted to base64, and sent to the LLM
- the LLM receives both visual context and structured textual context

The LLM generates:

- context
- user intent
- state change
- task relation
- enhanced descriptions

### Firebase-based inference bridge flow

```text
Main Application
      ↓
Firebase Task Queue
      ↓
Colab/Qwen-VL Runtime (T4 GPU supported)
      ↓
Inference Result
      ↓
Neo4j Graph Update
```

### Graph enrichment

After reasoning, the system updates Neo4j nodes with richer semantic descriptions and merges overlapping page descriptions for coherence.

### Workflow abstraction (chain_evolve.py)

The system evaluates whether a chain is reusable and, if so, produces a high-level action node with preconditions, element sequence, and template pattern for reuse.

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

### STEP 1 – Exploration (OmniParser auto-parse)

**In the Gradio UI:**

1. Open `http://localhost:7860`
2. Go to **① Initialization** tab
   - Click **Refresh devices** → select your device
   - Enter a task description → click **Initialize**
3. Go to **② Exploration** tab
  - Click **▶ Start session** — the first screenshot is taken 
  - Start **omniparser-queue.ipynb** into a T4 GPU supported environment
  - First screenshot is sent to OmniParser
  - The labeled image + elements JSON are saved automatically
4. Choose an action and click **⚡ Perform action**
  - A new screenshot is captured and automatically parsed again
5. Repeat until the task is complete

---

### OmniParser outputs

Each screenshot is parsed automatically. Outputs are stored in:

- Labeled images: `./labeled_image/img/<task_id>.png`
- Elements JSON: `./labeled_image/json_labeled_data/<task_id>.json`

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

`parsed_result_ready: false` means a parsed result is missing for the last
captured screenshot.

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
- **Pinecone vectors:**  ResNet50 embeddings for pages and elements

### STEP 4 – Chain processing (background jobs)

In the **④ Chain Processing** tab:

1. Provide the start page ID from Neo4j
2. Click **🧠 Start chain_understand** or **🚀 Start chain_evolve**
3. Copy the **Job ID** and click **🔄 Poll status** to track progress

Jobs are tracked in an in-memory store and return status + results when done.

---

### STEP 5 – Action execution strategy (deployment engine)

The deployment engine executes high-level tasks on Android devices using a hybrid strategy:

```text
User Task
  ↓
Task Matching
  ↓
Screen Capture + UI Parsing
  ↓
Element Matching
  ↓
Shortcut Discovery
  ↓
Execution Template Generation
  ↓
Action Execution
  ↓
Task Completion Verification
```

If any structured step fails, it falls back to a reactive LLM-driven UI agent:

```text
Fallback → Reactive LLM-driven UI agent
```

Key execution modes:

- **Structured execution**: matches tasks to known high-level actions, uses shortcuts/templates, and executes deterministic steps (faster, reliable, lower LLM cost).
- **Reactive fallback**: captures the screen, sends screenshot + UI JSON to the LLM, and executes the next atomic action until completion.

Core execution components:

- `FirebaseLLMBridge`: async LLM calls via Firebase RTDB
- `Neo4jDatabase`: actions, shortcuts, page flows, UI metadata
- `OmniParser`: screen parsing and element extraction
- `ADB Tools`: tap/swipe/text/back device actions
- `LangGraph Workflow`: orchestration state machine

---

## Notebook execution (GPU)

Run these notebooks with a **T4 GPU** enabled runtime:

- `qwen_firebase_worker.ipynb`
- `OmniParser/omniparser-queue.ipynb`

---

## Component stack

| Component | Tool/Technology |
|---|---|
| Frontend | React (JavaScript) |
| Backend | FastAPI (Python) |
| ML Framework | Hugging Face Transformers |
| NLP Framework | spaCy, TextBlob, Hugging Face Tokenizers |
| Bias Mitigation Method | Word Replacement, Fairness Constraints, Attention Mechanism |
| ML Model | BERT, RoBERTa (for text classification) |
| DL Model | T5, XLNet, DeBERTa |
| Dataset | Custom Dataset from Bengali News Sources |
| Preprocessing Tool | NLTK, spaCy |
| Version Control | Git, GitHub |
| Database | PostgreSQL |
| Environment | Visual studio/ Jupyter notebook |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No devices found` | Check USB cable, enable USB debugging, run `adb devices` in terminal |
| `RuntimeError: No active session` | Call **Initialize** in the UI before starting exploration |
| `JSON file not found` | Ensure the OmniParser output JSON exists under `./labeled_image/json_labeled_data/` |
| Neo4j connection error | Ensure Neo4j is running and `Neo4j_URI` / `Neo4j_AUTH` in `config.py` are correct |
| Pinecone error | Check `PINECONE_API_KEY` in `config.py` |
| Feature service unreachable | Start your ResNet50 service and update `Feature_URI` in `config.py` |
