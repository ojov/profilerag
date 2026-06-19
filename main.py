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