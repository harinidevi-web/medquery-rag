from retrieval.pipeline import retrieve

results = retrieve("first line treatment for gonorrhoea")

for i, hit in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Score:   {hit['rerank_score']:.4f}")
    print(f"Source:  {hit['citation']['source']}")
    print(f"Page:    {hit['citation']['page']}")
    print(f"Section: {hit['citation']['section']}")
    print(f"Content: {hit['content'][:300]}")