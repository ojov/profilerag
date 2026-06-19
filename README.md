# Portfolio RAG Chatbot

A minimal RAG (Retrieval-Augmented Generation) chatbot that answers questions about you, built from a single markdown profile. Uses FastAPI, ChromaDB, Gradio, and the OpenAI API — no LangChain, no LlamaIndex, just the raw pipeline so every step is visible.

## How it works

Three pipelines:

**Indexing (runs once, on startup)**
`profile.md` → chunked by heading → embedded → stored in ChromaDB

**Query rewriting (runs on every follow-up)**
conversation history + new question → LLM rewrites into a standalone search query
(without this, "tell me more about that" retrieves garbage from the vector store)

**Querying (runs on every request)**
rewritten query → embedded → top-3 chunks retrieved → injected into prompt → LLM generates grounded answer

---

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) — fast Python package/project manager
- An OpenAI API key ([platform.openai.com/api-keys](https://platform.openai.com/api-keys))

Install `uv` if you don't have it:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
```

---

## 1. Project setup

```bash
mkdir portfolio-rag
cd portfolio-rag
uv init --no-readme
```

This creates `pyproject.toml` and a stub `main.py`.

Add dependencies — `uv` creates a `.venv` and installs everything:

```bash
uv add fastapi uvicorn chromadb openai python-dotenv gradio httpx
```

From now on, run any Python command with `uv run` (no manual venv activation needed):

```bash
uv run python -c "import chromadb; print('chroma ok')"
```

If that prints `chroma ok`, you're set.

### Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

### .gitignore

```
.venv/
.env
__pycache__/
*.pyc
```

---

## 2. Write your knowledge base — `profile.md`

This is the source content the chatbot retrieves from. Use `#`/`##` headings — they become natural chunk boundaries — and write in full sentences rather than keyword lists, since embedding models work better on natural prose.

```markdown
# About Me
I'm [Your Name], a [role] based in [location].
I work on [what you do]...

# Skills
[Your skills, written as sentences, not just a list]

# Projects

## [Project Name]
[What you built, the stack, the outcome]

# Experience
[Where you've worked, what you did]

# Contact
GitHub: github.com/yourusername
LinkedIn: linkedin.com/in/yourusername
Email: your.email@example.com
```

The richer and more specific this file is, the better the chatbot's answers will be.

---

## 3. Build the RAG pipeline — `rag.py`

This file holds chunking, embedding, indexing, and query-rewriting logic — kept separate from the FastAPI app so it's reusable and easier to read.

```python
import re
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(timeout=30.0, max_retries=3)
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("portfolio")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def chunk_markdown(path: str, chunk_size: int = 250) -> list[str]:
    text = open(path, encoding="utf-8").read()
    sections = re.split(r"\n(?=#{1,3}\s)", text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        words = section.split()
        if len(words) <= chunk_size:
            chunks.append(section)
        else:
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def index_profile(path: str = "profile.md"):
    chunks = chunk_markdown(path)
    response = client.embeddings.create(input=chunks, model=EMBED_MODEL)
    embeddings = [item.embedding for item in response.data]
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB.")


def rewrite_query(question: str, history: list[dict]) -> str:
    """
    Rewrite a follow-up question into a standalone search query.
    Without this, vague questions like "tell me more about that" embed
    poorly and return irrelevant chunks from the vector store.
    """
    if not history:
        return question

    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"""Given this conversation history:
{history_text}

Rewrite the user's latest question into a short, standalone search query
that makes sense without the conversation history. Reply with ONLY the
rewritten query, nothing else.""",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    index_profile()
    results = collection.get()
    print(f"\nCollection now has {len(results['ids'])} documents.")
    print("First document preview:", results["documents"][0][:100])
```

**Test chunking and indexing standalone, before bringing FastAPI into it:**

```bash
uv run python rag.py
```

Expected output:
```
Indexed 9 chunks into ChromaDB.

Collection now has 9 documents.
First document preview: # About Me ...
```

---

## 4. Build the Gradio UI — `app.py`

This file defines the chat interface. It calls the `/chat` FastAPI endpoint and converts Gradio's tuple history format into the OpenAI message list format.

```python
import httpx
import gradio as gr

CHAT_ENDPOINT = "http://localhost:8000/chat"


def chat_fn(message: str, history: list) -> str:
    openai_history = []
    for user_msg, assistant_msg in history:
        openai_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            openai_history.append({"role": "assistant", "content": assistant_msg})

    response = httpx.post(
        CHAT_ENDPOINT,
        json={"message": message, "history": openai_history},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["reply"]


with gr.Blocks(title="Portfolio Intelligence") as demo:
    gr.ChatInterface(
        fn=chat_fn,
        examples=[
            "What are your main technical skills?",
            "Tell me about your projects.",
            "What is your work experience?",
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()
```

---

## 5. Build the API — `main.py`

This file wires together the FastAPI backend and the Gradio UI. The Gradio app is mounted at `/` so visiting the server in a browser opens the chat interface directly.

```python
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import client, collection, index_profile, rewrite_query, EMBED_MODEL, CHAT_MODEL
from app import demo

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"/"assistant", "content": "..."}]


@app.post("/chat")
def chat(query: Query):
    # Rewrite follow-up questions into standalone queries for retrieval
    search_query = rewrite_query(query.message, query.history)

    q_response = client.embeddings.create(input=[search_query], model=EMBED_MODEL)
    q_embedding = q_response.data[0].embedding

    results = collection.query(query_embeddings=[q_embedding], n_results=3)
    context = "\n\n".join(results["documents"][0])

    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful assistant representing this person's portfolio.
Answer questions about them using only the context below. Be conversational and friendly.
If the answer isn't in the context, say you don't have that information.

Context:
{context}""",
        }
    ]
    messages.extend(query.history)
    messages.append({"role": "user", "content": query.message})

    completion = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return {"reply": completion.choices[0].message.content}


@app.on_event("startup")
def startup_event():
    index_profile()


# Mount Gradio at the root so the chat UI is the landing page
app = gr.mount_gradio_app(app, demo, path="/")
```

### Run it

```bash
uv run uvicorn main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
Indexed 9 chunks into ChromaDB.
INFO:     Application startup complete.
```

### Use it

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) — the Gradio chat UI loads directly.

The raw API is also available. To test via Swagger UI:

```
http://127.0.0.1:8000/docs
```

Example request:
```json
{
  "message": "What projects has this person worked on?",
  "history": []
}
```

For a follow-up with history:
```json
{
  "message": "Tell me more about that.",
  "history": [
    {"role": "user", "content": "What projects has this person worked on?"},
    {"role": "assistant", "content": "They worked on..."}
  ]
}
```

---

## Project structure

```
portfolio-rag/
├── profile.md          # your knowledge base
├── rag.py              # chunking, embedding, indexing, query rewriting
├── app.py              # Gradio chat UI
├── main.py             # FastAPI app + /chat endpoint + Gradio mount
├── pyproject.toml      # uv-managed dependencies
├── uv.lock
├── .env                # OPENAI_API_KEY (gitignored)
└── .gitignore
```

---

## Troubleshooting

**`openai.APIConnectionError: Connection error.`**
First confirm it's not a network/SSL issue at the system level:
```bash
curl -v https://api.openai.com/v1/models -H "Authorization: Bearer YOUR_KEY"
```
If `curl` succeeds but Python still fails, update the client libraries:
```bash
uv add --upgrade openai httpx
```
The client in `rag.py` already has explicit timeout/retry settings (`timeout=30.0, max_retries=3`), which helps on flaky connections.

**Auth errors on startup**
Make sure `.env` is in the same directory you're running `uv run` from, and that `load_dotenv()` runs before `OpenAI()` is instantiated.

**Gradio UI not loading at `/`**
The `gr.mount_gradio_app(app, demo, path="/")` call must happen *after* all FastAPI routes are defined (it reassigns `app`). Keep it as the last line in `main.py`.

---

## Notes & limitations

- **In-memory vector store.** ChromaDB's default client keeps everything in RAM. Restarting the server re-indexes `profile.md` from scratch (a few cents in embedding calls, but not instant).
- **Single document.** This indexes one markdown file. Scaling to multiple documents just means looping `chunk_markdown` over a folder.
- **No streaming.** Responses arrive all at once. Adding `stream=True` to the completions call and returning a `StreamingResponse` from FastAPI would give a typing effect.

## What's next

- Index multiple files (CV, blog posts, project READMEs)
- Add metadata filtering (tag chunks by topic, filter at query time)
- Stream responses (`stream=True`) for a typing effect in the UI
- Persist the ChromaDB collection to disk so re-indexing isn't needed on every restart
- Deploy to Railway or Render and connect to a real portfolio site
- Trigger re-indexing from a webhook or n8n workflow when `profile.md` changes
