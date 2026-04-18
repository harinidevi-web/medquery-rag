from retrieval.pipeline import retrieve
from generation.prompt_loader import load_prompt_config, build_prompt
from generation.generator import generate_answer
import time

def answer(
    question: str,
    prompt_version: str = "v1",
    top_k: int = 5,
) -> dict:
    """
    Full end-to-end pipeline:
    question → retrieval → prompt → LLM → answer + citations
    """
    start = time.time()

    # Step 1 — retrieve relevant chunks
    hits = retrieve(question, top_k=top_k)

    # Step 2 — load versioned prompt config
    config = load_prompt_config(version=prompt_version)

    # Step 3 — build prompt from chunks
    system_prompt, user_prompt = build_prompt(question, hits, config)

    # Step 4 — generate answer
    answer_text = generate_answer(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )

    latency = round(time.time() - start, 2)

    # Step 5 — return structured response
    return {
        "question":       question,
        "answer":         answer_text,
        "citations":      [hit["citation"] for hit in hits],
        "prompt_version": prompt_version,
        "model":          config["model"],
        "latency_sec":    latency,
        "chunks_used":    len(hits),
    }