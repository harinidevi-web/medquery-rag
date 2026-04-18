import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()

ef = embedding_functions.CohereEmbeddingFunction(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-english-v3.0"
)

client = chromadb.PersistentClient(path="./chroma_db")

def vector_search(query: str, n_results: int = 20) -> list[dict]:
    """
    Search ChromaDB using semantic vector similarity.
    Returns list of dicts with content, metadata, and rank score.
    """
    col = client.get_collection("medquery_children", embedding_function=ef)
    results = col.query(query_texts=[query], n_results=n_results)

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "content":  doc,
            "metadata": meta,
            "score":    1 - distance,  # cosine distance → similarity
        })

    return hits