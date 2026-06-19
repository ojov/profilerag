# Portfolio RAG Chatbot

A minimal RAG (Retrieval-Augmented Generation) chatbot that answers questions about you, built from a single markdown profile. Uses FastAPI, ChromaDB, and the OpenAI API — no LangChain, no LlamaIndex, just the raw pipeline so every step is visible.

## How it works

Two pipelines:

**Indexing (runs once, on startup)**
`profile.md` → chunked by heading → embedded → stored in ChromaDB

**Querying (runs on every request)**
user question → embedded → top-3 chunks retrieved → injected into prompt → LLM generates grounded answer

---

## Prerequisites

- Python 3.10+
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
uv add fastapi uvicorn chromadb openai python-dotenv
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

This file holds chunking, embedding, and indexing logic — kept separate from the FastAPI app so it's reusable and easier to read.

```python
import os
import re

from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(timeout=30.0, max_retries=3)  # reads OPENAI_API_KEY from environment
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("portfolio")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


# ── 1. CHUNKING ──────────────────────────────────────────────
def chunk_markdown(path: str, chunk_size: int = 250) -> list[str]:
    """
    Split a markdown file into chunks.
    Strategy: split on top-level headings first (so each section
    stays together), then further split long sections by word count.
    """
    text = open(path, encoding="utf-8").read()

    # Split on lines starting with '#' (headings), keeping the heading
    # attached to its section
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
            # Section too long — split further by word count
            for i in range(0, len(words), chunk_size):
                chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


# ── 2. EMBED & INDEX ─────────────────────────────────────────
def index_profile(path: str = "profile.md"):
    chunks = chunk_markdown(path)

    response = client.embeddings.create(
        input=chunks,
        model=EMBED_MODEL
    )
    embeddings = [item.embedding for item in response.data]

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB.")


if __name__ == "__main__":
    index_profile()

    # Sanity check: pull everything back out
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

This confirms two things independently: that your markdown is being split sensibly, and that OpenAI's embedding API is reachable and returning vectors.

---

## 4. Build the API — `main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import client, collection, index_profile, EMBED_MODEL, CHAT_MODEL

# ── APP SETUP ─────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    message: str


# ── QUERY ENDPOINT ───────────────────────────────────────────
@app.post("/chat")
def chat(query: Query):
    # Embed the incoming question
    q_response = client.embeddings.create(
        input=[query.message],
        model=EMBED_MODEL
    )
    q_embedding = q_response.data[0].embedding

    # Retrieve top 3 most relevant chunks
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3
    )
    context = "\n\n".join(results["documents"][0])

    # Generate a grounded response
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"""You are a helpful assistant representing this person's portfolio.
Answer questions about them using only the context below. Be conversational and friendly.
If the answer isn't in the context, say you don't have that information.

Context:
{context}"""
            },
            {"role": "user", "content": query.message}
        ]
    )

    return {"reply": completion.choices[0].message.content}


# ── INDEX ON STARTUP ─────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    index_profile()
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

### Test it

Open the auto-generated Swagger UI:

```
http://127.0.0.1:8000/docs
```

Click **POST /chat → Try it out**, enter:

```json
{
  "message": "What projects has this person worked on?"
}
```

Click **Execute** — you should get back a generated answer grounded in your `profile.md`.

---

## Project structure

```
portfolio-rag/
├── profile.md          # your knowledge base
├── rag.py              # chunking, embedding, indexing
├── main.py             # FastAPI app + /chat endpoint
├── pyproject.toml       # uv-managed dependencies
├── uv.lock
├── .env                 # OPENAI_API_KEY (gitignored)
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
Also add explicit timeout/retry settings to the client in `rag.py` (already included above):
```python
client = OpenAI(timeout=30.0, max_retries=3)
```

**Auth errors on startup**
Make sure `.env` is in the same directory you're running `uv run` from, and that `load_dotenv()` runs before `OpenAI()` is instantiated.

---

## Notes & limitations

- **No memory.** Each `/chat` call is stateless — the bot has no record of earlier messages in the conversation. Adding memory means passing conversation history from the client with each request (see "What's next" below).
- **In-memory vector store.** ChromaDB's default client keeps everything in RAM. Restarting the server re-indexes `profile.md` from scratch (cheap — a few cents in embedding calls — but not instant).
- **Single document.** This indexes one markdown file. Scaling to multiple documents just means looping `chunk_markdown` over a folder.

## What's next

- Add conversation memory (pass `history` from the frontend)
- Index multiple files (CV, blog posts, project READMEs)
- Add metadata filtering (tag chunks by topic, filter at query time)
- Stream responses (`stream=True`) for a typing effect
- Deploy to Railway and connect to a real portfolio site
- Trigger re-indexing from an n8n workflow when `profile.md` changes