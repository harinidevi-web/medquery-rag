def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum of 1 / (rank + k) across all lists.
    k=60 is the standard constant — dampens the effect of top ranks.

    Why RRF over weighted sum:
    - No need to normalise scores across different scales (BM25 vs cosine)
    - Robust to outliers — a rank-1 result in one list doesn't dominate
    - Consistently outperforms more complex fusion methods in practice
    """
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            # Use content as the deduplication key
            key = hit["content"][:100]
            if key not in scores:
                scores[key] = 0.0
                docs[key]   = hit
            scores[key] += 1.0 / (rank + k)

    # Sort by fused score descending
    fused = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    return [
        {**docs[key], "rrf_score": scores[key]}
        for key in fused
    ]