import yaml
from pathlib import Path

PROMPTS_DIR = Path("./prompts")

def load_prompt_config(version: str = "v1") -> dict:
    """
    Load a versioned prompt config from YAML.
    This is how you A/B test prompts without touching code.
    """
    config_path = PROMPTS_DIR / version / "system.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Prompt config not found: {config_path}. "
            f"Available versions: {[d.name for d in PROMPTS_DIR.iterdir()]}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def build_prompt(
    question: str,
    hits: list[dict],
    config: dict,
) -> tuple[str, str]:
    """
    Build system + user prompt from retrieved chunks.
    Returns (system_prompt, user_prompt).
    """
    # Format each chunk with its citation
    context_blocks = []
    for i, hit in enumerate(hits, 1):
        citation = hit.get("citation", {})
        block = (
            f"[{i}] Source: {citation.get('source', 'Unknown')}, "
            f"Page {citation.get('page', '?')}, "
            f"Section: {citation.get('section', 'General')}\n"
            f"{hit['content']}"
        )
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = config["system_prompt"]
    user_prompt = config["user_template"].format(
        question=question,
        context=context,
    )

    return system_prompt, user_prompt