# CMPSC 132 Canvas RAG

A small RAG pipeline over the CMPSC 132 course corpus.

- **Part 0** — scrape Canvas content via the REST API into a structured tree of PDFs + metadata (`scrape_canvas.py`).
- **Part 1** — chunk + embed the corpus locally; retrieve top-K passages for a test query to verify the pipeline before touching an LLM (`build_index.py`, `query.py`).
- **Part 2** — multi-turn RAG chat over the indexed corpus, running a local LLM via Ollama (`chat.py`, shared `retriever.py`).
- **Part 3** — Gradio web UI over the same pipeline (`app.py`). FastAPI + Docker still TBD.

## Setup

1. Create and activate a virtualenv:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install deps:

   ```bash
   pip install -r requirements.txt
   ```

   weasyprint on macOS also needs system libs — if install fails:

   ```bash
   brew install pango cairo libffi
   ```

3. Generate a Canvas access token:
   Canvas → Account → Settings → scroll to "Approved Integrations" → **+ New Access Token**. Give it a name, optional expiry, then copy the token once (you can't see it again).

4. Copy `.env.example` to `.env` and fill in the token + course ID:

   ```bash
   cp .env.example .env
   # then edit .env
   ```

   The course ID is the number in the course URL: `https://psu.instructure.com/courses/2345678` → `2345678`.

## Run

Smoke-test against one module first:

```bash
python scrape_canvas.py --limit 1
```

Full scrape:

```bash
python scrape_canvas.py
```

## Output layout

```
canvas_data/
├── 00_academic_integrity_in_eecs/
│   ├── 01_what_is_academic_integrity_ai.pdf
│   ├── 02_why_is_academic_integrity_important.pdf
│   ├── ...
│   └── module_index.md
├── 01_module_0_getting_started_week_1/
│   ├── 01_about_your_instructor.pdf
│   ├── assignments/
│   │   └── 08_lab_0_environment_set_up.pdf
│   └── module_index.md
...
```

- **Pages** → rendered HTML → PDF via weasyprint.
- **Files** → downloaded verbatim (PDFs stay PDFs, `.py` stays `.py`).
- **External URLs** → recorded in `module_index.md` (no scraping of external sites).
- **Assignments / Quizzes / Discussions** → description rendered to PDF + `.json` metadata sidecar.
- **Sub Headers** → section dividers in `module_index.md`.

## Part 1 — chunk, embed, sanity-check retrieval

Once you have `canvas_data/` populated, build a local embedding index:

```bash
python build_index.py
```

This walks `canvas_data/`, extracts text from every PDF (per page) and every `.py` file, chunks with overlap, embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2` (first run downloads ~90 MB), and writes:

```
index/
├── embeddings.npy    # float32, row-normalized
├── chunks.jsonl      # source path / module / page / chunk text
└── meta.json         # model name, dim, count, timestamp
```

Then poke at it with the retrieval CLI (no LLM yet — this is a sanity check):

```bash
python query.py "what is a linked list"
python query.py --k 10 "how do you balance an AVL tree"
python query.py --full "explain the quicksort partition step"
```

Each hit shows rank, cosine score, source file, page, and a preview of the chunk. If the top-5 for a question look obviously relevant, the index is healthy and we move to Part 2.

Useful flags:

- `--chunk-size` / `--overlap` on `build_index.py` — tune chunk length (default 1000 chars / 150 overlap).
- `--k` on `query.py` — how many hits to return.
- `--full` on `query.py` — dump full chunks instead of previews.

Once the top-K hits for a handful of CMPSC 132 questions look obviously relevant, Part 1 is done.

## Part 2 — multi-turn RAG chat (Ollama)

Part 2 adds a chat CLI that wraps retrieval + a local LLM.

### One-time Ollama setup

```bash
brew install ollama
ollama serve &                 # run the daemon (or use the menu-bar app)
ollama pull llama3.1:8b        # ~4.7 GB; pick any model you like
```

Any chat-capable Ollama model works — `llama3.1:8b`, `qwen2.5:7b-instruct`, `mistral:7b`. Set it in `.env`:

```
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
TOP_K=5
MAX_HISTORY_TURNS=6
```

### Run the chat

```bash
python chat.py
```

Each turn, the script retrieves the top-K chunks for your question, injects them as "Course material" into the current user message, and streams the assistant's reply. Sources are listed after each answer. Chat history keeps only raw questions + replies — the retrieved context is fresh per turn, so it doesn't pollute future prompts.

Slash commands at the prompt:

```
/help        list commands
/exit /quit  leave
/reset       clear chat history
/k N         change retrieval top-K
/sources     toggle showing sources after each answer
/history     print current history
```

### Sanity checks

Try a few CMPSC 132 questions after launching:

```
> what is a linked list
> how do you implement insert in a BST
> explain the difference between __str__ and __repr__
> show me a recursive factorial function from the notes
```

If the answers reference specific source files from the corpus (with `[1]`, `[2]` citations matching the listed Sources block), the pipeline is healthy.

## Part 3 — Gradio web UI

`app.py` wraps the same `Retriever` + `stream_ollama` logic behind a browser
chat. No RAG logic is duplicated — it imports `SYSTEM_PROMPT`,
`render_user_turn`, and `stream_ollama` from `chat.py`, so prompt tweaks made
for the CLI apply to the UI too.

Install Part 3 dep:

```bash
pip install -r requirements.txt   # adds gradio
```

Make sure Ollama is running and the index has been built (Parts 1 + 2), then:

```bash
python app.py
```

Open http://127.0.0.1:7860.

The UI mirrors the CLI's feature set:

- **Chat panel** — streaming replies, with a copy button per message.
- **Top-K slider** — equivalent of `/k N` in the CLI (1–15).
- **Show sources** — toggle the sources panel on/off (equivalent of `/sources`).
- **Clear chat** — wipes chat history (equivalent of `/reset`).
- **Sources panel** — below the chat, shows the retrieved chunks + scores for
  the most recent answer only (not persisted across turns, so the main chat
  stays readable).

History is stored per browser session (Gradio `State`); reloading the page
starts fresh. All env vars used by `chat.py` (`OLLAMA_URL`, `OLLAMA_MODEL`,
`TOP_K`, `MAX_HISTORY_TURNS`, `OLLAMA_NUM_CTX`) apply here too.
