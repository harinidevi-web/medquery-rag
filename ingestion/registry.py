import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = "./documents/registry.json"

def load_registry() -> dict:
    if Path(REGISTRY_PATH).exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {}

def save_registry(registry: dict):
    Path(REGISTRY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def register_document(
    source_name: str,
    version: str,
    pdf_path: str,
    chunk_count: int,
    parent_count: int,
):
    registry = load_registry()
    registry[source_name] = {
        "version":      version,
        "ingested_at":  datetime.now(timezone.utc).isoformat(),
        "pdf_path":     pdf_path,
        "chunk_count":  chunk_count,
        "parent_count": parent_count,
    }
    save_registry(registry)
    print(f"  Registered '{source_name}' v{version} — {chunk_count} chunks.")

def is_already_ingested(source_name: str, version: str) -> bool:
    registry = load_registry()
    entry = registry.get(source_name)
    return entry is not None and entry.get("version") == version