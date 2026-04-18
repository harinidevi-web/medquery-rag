import json
from pathlib import Path
from retrieval.vector_retriever import vector_search
from retrieval.bm25_retriever   import bm25_search
from retrieval.fusion           import reciprocal_rank_fusion
from retrieval.reranker         import rerank

PARENTS_PATH = "./documents/parents.json"

def _get_parent(parent_id: str) -> str:
    """Look up the full parent chunk by ID."""
    with open(PARENTS_PATH) as f:
        store = json.load(f)
    entry = store.get(parent_id)
    return entry["content"] if entry else ""


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Full hybrid retrieval pipeline:
    1. Vector search  — semantic similarity (children)
    2. BM25 search    — keyword matching (parents)
    3. RRF fusion     — combine both ranked lists
    4. Rerank         — cross-encoder precision pass
    5. Parent lookup  — swap children for full parent context
    """
    print(f"\nQuery: {query}")

    # Step 1 + 2 — run both retrievers
    vector_hits = vector_search(query, n_results=20)
    bm25_hits   = bm25_search(query,   n_results=20)
    print(f"  Vector hits: {len(vector_hits)} | BM25 hits: {len(bm25_hits)}")

    # Step 3 — fuse
    fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
    print(f"  After RRF fusion: {len(fused)} unique chunks")

    # Step 4 — rerank top 20 fused results
    reranked = rerank(query, fused[:20], top_n=top_k)
    print(f"  After reranking: {len(reranked)} chunks")

    # Step 5 — swap child chunks for their parent (full context)
    final = []
    for hit in reranked:
        parent_id = hit["metadata"].get("parent_id")
        parent_content = _get_parent(parent_id) if parent_id else hit["content"]
        final.append({
            "content":       parent_content,
            "metadata":      hit["metadata"],
            "rerank_score":  hit["rerank_score"],
            "citation": {
                "source":   hit["metadata"].get("source_doc"),
                "section":  hit["metadata"].get("section_heading"),
                "page":     hit["metadata"].get("page_num"),
                "version":  hit["metadata"].get("guideline_version"),
            }
        })

    return final