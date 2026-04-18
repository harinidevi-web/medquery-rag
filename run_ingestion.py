from ingestion.pipeline import ingest_document
from ingestion.sources import GUIDELINE_SOURCES

if __name__ == "__main__":
    for source in GUIDELINE_SOURCES:
        ingest_document(
            pdf_path=f"data/raw/{source['name']}.pdf",
            source_name=source["name"],
            guideline_version=source["version"],
        )