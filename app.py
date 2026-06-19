import gradio as gr

from rag import client, collection, index_profile, rewrite_query, EMBED_MODEL, CHAT_MODEL

DIGITAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&display=swap');

:root {
    --bg-0: #030c0c;
    --bg-1: #071414;
    --bg-2: #0a1c1c;
    --cyan:  #00f5ff;
    --green: #00ff88;
    --dim:   #007a5a;
    --muted: #2e5a52;
    --text:  #b8f0e0;
    --border: #0d3030;
    --glow-c: 0 0 8px #00f5ff, 0 0 20px rgba(0,245,255,.25);
    --glow-g: 0 0 8px #00ff88, 0 0 20px rgba(0,255,136,.2);
}

/* ── base ────────────────────────────── */
body, .gradio-container, .app {
    background: var(--bg-0) !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    color: var(--text) !important;
}

/* scanlines */
.gradio-container::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,245,255,.018) 2px,
        rgba(0,245,255,.018) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── chat window ────────────────────── */
.chatbot, #chatbot, .chatbot > .wrap {
    background: var(--bg-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    box-shadow: inset 0 0 40px rgba(0,245,255,.04), var(--glow-c) !important;
}

/* ── message bubbles ────────────────── */
/* Gradio 4.x */
.message.user, .user > .message {
    background: rgba(0,160,96,.15) !important;
    border: 1px solid var(--dim) !important;
    border-radius: 2px 14px 14px 14px !important;
    color: var(--green) !important;
    box-shadow: var(--glow-g) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
.message.bot, .bot > .message, .message.assistant {
    background: rgba(0,40,70,.45) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px 2px 14px 14px !important;
    color: var(--text) !important;
    box-shadow: 0 0 10px rgba(0,245,255,.08) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* Gradio 5.x bubble wrappers */
.bubble-wrap.user .message,
[data-testid="user"] .message {
    background: rgba(0,160,96,.15) !important;
    border: 1px solid var(--dim) !important;
    border-radius: 2px 14px 14px 14px !important;
    color: var(--green) !important;
    box-shadow: var(--glow-g) !important;
}
.bubble-wrap.bot .message,
[data-testid="bot"] .message {
    background: rgba(0,40,70,.45) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px 2px 14px 14px !important;
    color: var(--text) !important;
    box-shadow: 0 0 10px rgba(0,245,255,.08) !important;
}

/* ── input box ──────────────────────── */
textarea, .input-container textarea, #chat-input textarea {
    background: var(--bg-2) !important;
    border: 1px solid var(--dim) !important;
    border-radius: 4px !important;
    color: var(--green) !important;
    caret-color: var(--cyan);
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9em !important;
}
textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: var(--glow-c) !important;
    outline: none !important;
}
textarea::placeholder { color: var(--muted) !important; }

/* ── buttons ────────────────────────── */
button.primary, button[variant="primary"] {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: 0.75em !important;
    box-shadow: var(--glow-c) !important;
    transition: all .2s ease !important;
    border-radius: 3px !important;
}
button.primary:hover, button[variant="primary"]:hover {
    background: rgba(0,245,255,.1) !important;
    box-shadow: 0 0 18px var(--cyan), 0 0 36px rgba(0,245,255,.4) !important;
}
button.secondary, button[variant="secondary"] {
    background: transparent !important;
    border: 1px solid var(--muted) !important;
    color: var(--muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px;
    border-radius: 3px !important;
    transition: all .2s ease !important;
}
button.secondary:hover, button[variant="secondary"]:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}

/* ── examples ───────────────────────── */
.examples table td, .examples-holder .label, .example {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8em !important;
    border-radius: 3px !important;
    transition: all .15s ease !important;
}
.example:hover {
    border-color: var(--dim) !important;
    color: var(--green) !important;
    box-shadow: var(--glow-g) !important;
}

/* ── scrollbar ──────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-0); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--dim); }

/* ── misc ───────────────────────────── */
.block, .panel, .gr-block { background: var(--bg-1) !important; border-color: var(--border) !important; }
.prose, .prose p, .prose li { color: var(--muted) !important; font-family: 'Share Tech Mono', monospace !important; }
footer, .footer { display: none !important; }
"""

HEADER_HTML = """
<div style="
    font-family:'Orbitron',monospace;
    text-align:center;
    padding:24px 0 16px;
    border-bottom:1px solid #0d3030;
    margin-bottom:8px;
">
    <div style="
        font-size:2em;
        font-weight:700;
        color:#00f5ff;
        letter-spacing:6px;
        text-shadow:0 0 12px #00f5ff, 0 0 28px rgba(0,245,255,.5);
    ">OSAS.AI</div>
    <div style="
        font-family:'Share Tech Mono',monospace;
        color:#2e5a52;
        font-size:.72em;
        letter-spacing:4px;
        margin-top:6px;
    ">[ PORTFOLIO INTELLIGENCE SYSTEM ]</div>
    <div style="
        font-family:'Share Tech Mono',monospace;
        color:#007a5a;
        font-size:.62em;
        letter-spacing:2px;
        margin-top:4px;
    ">RAG v1.0 &nbsp;·&nbsp; GPT-4o-mini &nbsp;·&nbsp; ChromaDB</div>
</div>
"""


def chat_fn(message: str, history: list) -> str:
    # Convert Gradio tuple history → OpenAI messages
    openai_history = []
    for user_msg, assistant_msg in history:
        openai_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            openai_history.append({"role": "assistant", "content": assistant_msg})

    search_query = rewrite_query(message, openai_history)

    q_response = client.embeddings.create(input=[search_query], model=EMBED_MODEL)
    q_embedding = q_response.data[0].embedding

    results = collection.query(query_embeddings=[q_embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant representing this person's portfolio. "
                "Answer questions about them using only the context below. "
                "Be conversational and friendly. "
                "If the answer isn't in the context, say you don't have that information.\n\n"
                f"Context:\n{context}"
            ),
        }
    ]
    messages.extend(openai_history)
    messages.append({"role": "user", "content": message})

    completion = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return completion.choices[0].message.content


with gr.Blocks(css=DIGITAL_CSS, title="OSAS.AI — Portfolio Intelligence") as demo:
    gr.HTML(HEADER_HTML)
    gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(
            height=500,
            placeholder=(
                "<div style='color:#0d3030;font-family:Share Tech Mono,monospace;"
                "text-align:center;padding-top:40px;letter-spacing:2px'>"
                "[ AWAITING QUERY... ]</div>"
            ),
            show_label=False,
        ),
        textbox=gr.Textbox(
            placeholder="> Ask about Osas — skills, projects, experience...",
            container=False,
            scale=7,
            show_label=False,
            lines=1,
        ),
        submit_btn="SEND ▶",
        clear_btn="CLR ✕",
        examples=[
            "What are Osas's main technical skills?",
            "Tell me about the RAG workshop project.",
            "What backend technologies does Osas work with?",
            "What is Osas's experience with Java SDKs?",
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    index_profile()
    demo.launch()
