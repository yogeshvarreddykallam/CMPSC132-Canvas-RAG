# CMPSC 132 Canvas RAG

A fully local Retrieval-Augmented Generation (RAG) pipeline that indexes the CMPSC 132 course corpus from Canvas and lets you query it through a CLI or a Gradio web UI — no external API calls required.

## Overview

| Part | Module | Description |
|------|--------|-------------|
| 0 | `scrape_canvas.py` | Scrapes Canvas pages & files via the REST API into a structured tree of PDFs + metadata |
| 1 | `build_index.py`, `query.py` | Chunks + embeds the corpus locally; retrieves top-K passages to verify the pipeline |
| 2 | `chat.py`, `retriever.py` | Multi-turn RAG chat over the indexed corpus using a local LLM via Ollama |
| 3 | `app.py` | Gradio web UI over the same pipeline |

## Tech Stack

- **Embeddings:** `sentence-transformers` (local)
- **Vector Store:** FAISS (`IndexFlatIP` with cosine similarity)
- **LLM:** Any Ollama-compatible model (default: `llama3.1:8b`)
- **Web UI:** Gradio
- **PDF rendering:** WeasyPrint

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
# macOS only — if weasyprint fails:
brew install pango cairo libffi

# 3. Configure environment
cp .env.example .env
# Edit .env: set CANVAS_TOKEN and CANVAS_COURSE_ID
```

**Finding your Course ID:** It's the number in the Canvas URL — e.g., `https://psu.instructure.com/courses/2345678` → `2345678`.

**Generating a Canvas token:** Canvas → Account → Settings → Approved Integrations → **+ New Access Token**.

## Usage

```bash
# Smoke-test with one module
python scrape_canvas.py --limit 1

# Full corpus scrape
python scrape_canvas.py

# Build the FAISS index
python build_index.py

# Run a test query
python query.py "What is the late policy?"

# Start multi-turn CLI chat
python chat.py

# Launch Gradio web UI (http://127.0.0.1:7860)
python app.py
```

## Output Layout

```
canvas_data/
├── 00_academic_integrity_in_eecs/
│   ├── 01_what_is_academic_integrity_ai.pdf
│   └── module_index.md
├── 01_module_0_getting_started_week_1/
│   ├── 01_about_your_instructor.pdf
│   └── assignments/
│       └── 08_lab_0_environment_set_up.pdf
...
index/
├── faiss.index
├── chunks.jsonl
└── meta.json
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CANVAS_TOKEN` | — | Canvas API access token (required) |
| `CANVAS_COURSE_ID` | — | Canvas course ID (required) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model to use for generation |
| `TOP_K` | `5` | Number of passages to retrieve |
| `MAX_HISTORY_TURNS` | `6` | Conversation history window |
