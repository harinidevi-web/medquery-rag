import json
from pathlib import Path
from rank_bm25 import BM25Okapi

PARENTS_PATH = "./documents/parents.json"

def _load_corpus() -> tuple[list[str], list[dict]]:
    """Load all parent chunks as the BM25 corpus."""
    with open(PARENTS_PATH) as f:
        store = json.load(f)

    texts, metas = [], []
    for pid, entry in store.items():
        texts.append(entry["content"])
        metas.append({**entry["metadata"], "parent_id": pid})

    return texts, metas


def bm25_search(query: str, n_results: int = 20) -> list[dict]:
    """
    Keyword search over parent chunks using BM25.
    Returns list of dicts with content, metadata, and score.
    """
    texts, metas = _load_corpus()

    # Tokenise — lowercase, split on whitespace
    tokenised_corpus = [t.lower().split() for t in texts]
    tokenised_query  = query.lower().split()

    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(tokenised_query)

    # Pair scores with documents and sort
    ranked = sorted(
        zip(scores, texts, metas),
        key=lambda x: x[0],
        reverse=True
    )[:n_results]

    hits = []
    for score, text, meta in ranked:
        hits.append({
            "content":  text,
            "metadata": meta,
            "score":    float(score),
        })

    return hits