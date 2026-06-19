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