import os
import cohere
from dotenv import load_dotenv
load_dotenv()

co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

def rerank(query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    if not hits:
        return []

    documents = [hit["content"] for hit in hits]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_n,
    )

    reranked = []
    for item in response.results:
        hit = hits[item.index].copy()
        hit["rerank_score"] = item.relevance_score
        reranked.append(hit)

    return reranked