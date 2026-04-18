from ingestion.loader import load_pdf
from ingestion.chunker import chunk_pages
from ingestion.embedder import store_children, store_parents
from ingestion.registry import register_document, is_already_ingested

def ingest_document(
    pdf_path: str,
    source_name: str,
    guideline_version: str,
    force: bool = False,
):
    if not force and is_already_ingested(source_name, guideline_version):
        print(f"Skipping '{source_name}' v{guideline_version} — already ingested.")
        return

    print(f"\nIngesting: {source_name} v{guideline_version}")

    print("  Loading PDF...")
    pages = load_pdf(pdf_path)
    print(f"  Loaded {len(pages)} pages.")

    print("  Chunking...")
    parents, children = chunk_pages(pages, source_name, guideline_version)
    print(f"  Created {len(parents)} parents, {len(children)} children.")

    print("  Embedding + storing children...")
    store_children(children)

    print("  Storing parents...")
    store_parents(parents)

    register_document(
        source_name=source_name,
        version=guideline_version,
        pdf_path=pdf_path,
        chunk_count=len(children),
        parent_count=len(parents),
    )

    print(f"Done: {source_name}\n")