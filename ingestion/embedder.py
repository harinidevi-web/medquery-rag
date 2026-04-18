from dotenv import load_dotenv
load_dotenv()

import os
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
import json
from pathlib import Path

CHROMA_PATH = "./chroma_db"
PARENTS_STORE_PATH = "./documents/parents.json"

# Cohere free tier — no disk space, 1000 calls/month free
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-english-v3.0",
)

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def store_children(children: list[Document], collection_name: str = "medquery_children"):
    """Embed and store child chunks in ChromaDB."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=cohere_ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids, texts, metadatas = [], [], []
    for doc in children:
        ids.append(doc.metadata["chunk_id"])
        texts.append(doc.page_content)
        metadatas.append(doc.metadata)

    # Upsert in batches of 96 (Cohere's max batch size)
    batch_size = 96
    total = len(ids)
    for i in range(0, total, batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )
        print(f"  Embedded {min(i+batch_size, total)}/{total} chunks...", end="\r")

    print(f"\n  Stored {total} child chunks in ChromaDB.")


def store_parents(parents: list[Document]):
    """Parents stored as JSON — retrieved by parent_id when a child matches."""
    Path(PARENTS_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if Path(PARENTS_STORE_PATH).exists():
        with open(PARENTS_STORE_PATH) as f:
            existing = json.load(f)

    for doc in parents:
        pid = doc.metadata["parent_id"]
        existing[pid] = {
            "content":  doc.page_content,
            "metadata": doc.metadata,
        }

    with open(PARENTS_STORE_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"  Stored {len(parents)} parent chunks to disk.")


def get_parent_by_id(parent_id: str) -> dict | None:
    if not Path(PARENTS_STORE_PATH).exists():
        return None
    with open(PARENTS_STORE_PATH) as f:
        store = json.load(f)
    return store.get(parent_id)