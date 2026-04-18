from generation.pipeline import answer

TEST_QUERIES = [
    # STI Guidelines
    "What is the recommended treatment for chlamydia in adults?",
    "How should syphilis be treated during pregnancy?",
    "What are the diagnostic criteria for pelvic inflammatory disease?",
    "What is the recommended HIV PrEP regimen?",
    "How is herpes simplex virus infection managed?",

    # NICE Hypertension
    "What blood pressure threshold requires drug treatment in adults?",
    "Which antihypertensive drugs are recommended for Black African patients?",
    "How should hypertension be monitored after starting treatment?",
    "What lifestyle changes are recommended for hypertension?",

    # NICE Diabetes
    "What is the first line drug treatment for type 2 diabetes?",
    "When should insulin be initiated in type 2 diabetes?",
    "What are the criteria for diagnosing type 2 diabetes?",
    "How should metformin dose be adjusted for kidney disease?",

    # WHO Hypertension
    "What does WHO recommend for hypertension screening frequency?",
    "What cardiovascular risk threshold triggers treatment in WHO guidelines?",

    # Cross-document — tests source routing
    "What is the cardiovascular risk management approach for diabetic patients with hypertension?",
]

print("=" * 70)
print("MEDQUERY RAG — EXTENDED GENERATION TEST")
print("=" * 70)

passed = 0
failed = 0

for i, q in enumerate(TEST_QUERIES, 1):
    print(f"\nTest {i}/{len(TEST_QUERIES)}: {q}")
    print("-" * 70)
    try:
        result = answer(q)
        print(f"Answer:\n{result['answer']}")
        print(f"\nCitations:")
        seen = set()
        for c in result['citations']:
            key = f"{c['source']} | Page {c['page']}"
            if key not in seen:
                print(f"  - {c['source']}, Page {c['page']}")
                seen.add(key)
        print(f"\nLatency: {result['latency_sec']}s | "
              f"Model: {result['model']} | "
              f"Chunks used: {result['chunks_used']}")
        passed += 1
    except Exception as e:
        print(f"ERROR: {e}")
        failed += 1
    print("=" * 70)

print(f"\nSUMMARY: {passed} succeeded / {failed} failed / {len(TEST_QUERIES)} total")