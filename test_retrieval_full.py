from retrieval.pipeline import retrieve
import json

TEST_QUERIES = [
    # --- STI queries (should hit STI-Guidelines-2021) ---
    {
        "query": "first line treatment for gonorrhoea",
        "expect_source": "STI-Guidelines-2021",
        "expect_keyword": ["ceftriaxone", "gonorrhea", "gonorrhoea"],
    },
    {
        "query": "recommended treatment for chlamydia in adults",
        "expect_source": "STI-Guidelines-2021",
        "expect_keyword": ["doxycycline", "azithromycin", "chlamydia"],
    },
    {
        "query": "syphilis treatment during pregnancy",
        "expect_source": "STI-Guidelines-2021",
        "expect_keyword": ["penicillin", "syphilis", "pregnancy"],
    },
    {
        "query": "HIV pre-exposure prophylaxis dosage",
        "expect_source": "STI-Guidelines-2021",
        "expect_keyword": ["PrEP", "tenofovir", "prophylaxis"],
    },

    # --- Hypertension queries (should hit NICE or WHO) ---
    {
        "query": "blood pressure target for hypertension in adults",
        "expect_source": "NICE-NG136-hypertension",
        "expect_keyword": ["140", "mmHg", "blood pressure"],
    },
    {
        "query": "first line antihypertensive drug treatment",
        "expect_source": "NICE-NG136-hypertension",
        "expect_keyword": ["ACE", "calcium", "thiazide", "amlodipine"],
    },

    # --- Diabetes queries (should hit NICE-NG28) ---
    {
        "query": "HbA1c target for type 2 diabetes",
        "expect_source": "NICE-NG28-diabetes",
        "expect_keyword": ["HbA1c", "mmol", "diabetes"],
    },
    {
        "query": "metformin dosage for type 2 diabetes",
        "expect_source": "NICE-NG28-diabetes",
        "expect_keyword": ["metformin", "diabetes", "dose"],
    },

    # --- Cross-document query (tests that fusion picks the right source) ---
    {
        "query": "cardiovascular risk assessment",
        "expect_source": None,  # could be hypertension or diabetes doc
        "expect_keyword": ["cardiovascular", "risk"],
    },
]


def run_tests():
    passed = 0
    failed = 0
    results_log = []

    print("=" * 60)
    print("MEDQUERY RAG — RETRIEVAL TEST SUITE")
    print("=" * 60)

    for i, test in enumerate(TEST_QUERIES, 1):
        query    = test["query"]
        expected_source  = test["expect_source"]
        expected_keywords = test["expect_keyword"]

        print(f"\nTest {i}: {query}")

        try:
            hits = retrieve(query, top_k=5)

            if not hits:
                print(f"  FAIL — no results returned")
                failed += 1
                continue

            top_hit      = hits[0]
            top_content  = top_hit["content"].lower()
            top_source   = top_hit["citation"]["source"]
            top_score    = top_hit["rerank_score"]

            # Check 1 — rerank score is meaningful
            score_ok = top_score > 0.30
            # Check 2 — correct source document
            source_ok = (expected_source is None) or (top_source == expected_source)
            # Check 3 — at least one expected keyword in top result
            keyword_ok = any(
                kw.lower() in top_content
                for kw in expected_keywords
            )

            test_passed = score_ok and source_ok and keyword_ok

            status = "PASS" if test_passed else "FAIL"
            print(f"  {status}")
            print(f"  Score:   {top_score:.4f} {'OK' if score_ok else 'LOW'}")
            print(f"  Source:  {top_source} {'OK' if source_ok else f'WRONG (expected {expected_source})'}")
            print(f"  Keyword: {'found' if keyword_ok else 'NOT FOUND — expected one of ' + str(expected_keywords)}")
            print(f"  Content: {top_hit['content'][:200]}")

            if test_passed:
                passed += 1
            else:
                failed += 1

            results_log.append({
                "query":       query,
                "status":      status,
                "score":       top_score,
                "source":      top_source,
                "keyword_ok":  keyword_ok,
            })

        except Exception as e:
            print(f"  ERROR — {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed / {failed} failed / {len(TEST_QUERIES)} total")
    print(f"PASS RATE: {round(passed/len(TEST_QUERIES)*100)}%")
    print("=" * 60)

    # Save results to JSON for your README
    with open("test_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print("\nResults saved to test_results.json")


if __name__ == "__main__":
    run_tests()