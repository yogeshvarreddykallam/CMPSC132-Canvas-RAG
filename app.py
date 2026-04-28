"""Gradio web UI over the same Retriever and Ollama pipeline that powers
chat.py. This module is a thin presentation layer; all retrieval and
generation logic is imported from chat.py and retriever.py so that any
change to the system prompt, context-injection function, or Ollama client
takes effect uniformly across the CLI and the web UI.

Launch:
    python app.py
    # then open http://127.0.0.1:7860
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from chat import SYSTEM_PROMPT, render_user_turn, stream_ollama
from retriever import Hit, Retriever


load_dotenv()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_TOP_K = _env_int("TOP_K", 5)
MAX_HISTORY_TURNS = _env_int("MAX_HISTORY_TURNS", 6)
NUM_CTX = _env_int("OLLAMA_NUM_CTX", 8192)


INDEX_DIR = Path("index").resolve()
RETRIEVER = Retriever(INDEX_DIR)


def _format_sources_md(hits: list[Hit]) -> str:
    """Render a markdown sources block for display under the chat."""
    if not hits:
        return "_No sources retrieved for the last question._"
    lines = ["### Sources", ""]
    for i, hit in enumerate(hits, start=1):
        lines.append(f"**[{i}]** `{hit.source_label}` &nbsp;&nbsp; _score {hit.score:.3f}_")
    return "\n".join(lines)


def _trim_history(messages: list[dict]) -> list[dict]:
    """Keep at most MAX_HISTORY_TURNS user/assistant pairs."""
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(messages) > max_msgs:
        return messages[-max_msgs:]
    return messages


def respond(
    user_message: str,
    chat_messages: list[dict],
    top_k: int,
    show_sources: bool,
):
    """Stream a RAG response. Yields updates for (textbox, chatbot, sources)."""
    user_message = (user_message or "").strip()
    if not user_message:
        yield "", chat_messages, gr.update()
        return

    chat_messages = list(chat_messages) + [
        {"role": "user", "content": user_message}
    ]
    yield "", chat_messages, gr.update()

    hits = RETRIEVER.search(user_message, k=int(top_k))
    sources_md = _format_sources_md(hits) if show_sources else ""

    past = _trim_history(chat_messages[:-1])
    current_user_content = render_user_turn(user_message, hits)
    llm_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *past,
        {"role": "user", "content": current_user_content},
    ]

    chat_messages = chat_messages + [{"role": "assistant", "content": ""}]
    try:
        answer = ""
        for piece in stream_ollama(OLLAMA_URL, OLLAMA_MODEL, llm_messages, num_ctx=NUM_CTX):
            answer += piece
            chat_messages[-1]["content"] = answer
            yield "", chat_messages, sources_md
    except RuntimeError as e:
        chat_messages[-1]["content"] = f"**[error]** {e}"
        yield "", chat_messages, sources_md
        return

    if not chat_messages[-1]["content"].strip():
        chat_messages[-1]["content"] = "_[empty response from Ollama]_"
        yield "", chat_messages, sources_md


def clear_chat():
    return [], ""


HEADER_MD = f"""
# CMPSC 132 RAG Chat

Retrieval over **{len(RETRIEVER.chunks)} chunks** (`{RETRIEVER.meta['model']}`)
&middot; LLM **{OLLAMA_MODEL}** &middot; history up to **{MAX_HISTORY_TURNS}** turns
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CMPSC 132 RAG") as demo:
        gr.Markdown(HEADER_MD)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=520, label="Chat")
                with gr.Row():
                    textbox = gr.Textbox(
                        placeholder="Ask a question about CMPSC 132...",
                        show_label=False,
                        scale=8,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                sources_md = gr.Markdown(
                    value="_Sources for the most recent answer will appear here._",
                    label="Sources",
                )

            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Controls")
                top_k = gr.Slider(
                    label="Top-K (retrieved chunks)",
                    minimum=1, maximum=15, step=1, value=DEFAULT_TOP_K,
                )
                show_sources = gr.Checkbox(label="Show sources", value=True)
                clear_btn = gr.Button("Clear chat", variant="secondary")
                gr.Markdown(
                    "---\n"
                    "**Tips**\n"
                    "- Ask questions grounded in the course corpus.\n"
                    "- The assistant will say so if it doesn't have the answer.\n"
                    "- Sources panel shows the chunks used for the most recent reply."
                )

        submit_inputs = [textbox, chatbot, top_k, show_sources]
        submit_outputs = [textbox, chatbot, sources_md]

        textbox.submit(respond, inputs=submit_inputs, outputs=submit_outputs)
        send_btn.click(respond, inputs=submit_inputs, outputs=submit_outputs)
        clear_btn.click(clear_chat, inputs=None, outputs=[chatbot, sources_md])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
