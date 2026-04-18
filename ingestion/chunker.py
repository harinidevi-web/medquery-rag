from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ingestion.loader import RawPage, table_to_text
import re, hashlib

CHILD_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n\n", "\n", ". ", " "],
)

PARENT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", r"\n\d+\.\d+", "\n\n", "\n"],
)


def extract_section_heading(text: str) -> str:
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+(\.\d+)*\s+\w", line):
            return line[:80]
        if line.startswith("#"):
            return line.lstrip("#").strip()[:80]
        if line.isupper() and len(line) > 5:
            return line[:80]
    return "General"


def make_chunk_id(text: str, page_num: int, index: int = 0) -> str:
    content = f"{page_num}:{index}:{text[:100]}"  # fixed — index now included
    return hashlib.md5(content.encode()).hexdigest()[:10]


def chunk_pages(
    pages: list[RawPage],
    source_name: str,
    guideline_version: str,
) -> tuple[list[Document], list[Document]]:

    parents: list[Document] = []
    children: list[Document] = []
    global_index = 0  # THIS was missing — must be before the loop

    for page in pages:
        # --- Handle tables as atomic chunks ---
        for table in page.tables:
            table_text = table_to_text(table)
            if len(table_text.strip()) < 20:
                continue

            parent_id = make_chunk_id(table_text, page.page_num, global_index)
            global_index += 1
            meta = {
                "source_doc":        source_name,
                "guideline_version": guideline_version,
                "page_num":          page.page_num,
                "section_heading":   "Table",
                "chunk_type":        "table",
                "parent_id":         parent_id,
                "chunk_id":          parent_id,
            }
            doc = Document(page_content=table_text, metadata=meta)
            parents.append(doc)
            children.append(doc)

        # --- Handle prose with parent-child splitting ---
        if not page.text:
            continue

        parent_docs = PARENT_SPLITTER.create_documents(
            [page.text],
            metadatas=[{
                "source_doc":        source_name,
                "guideline_version": guideline_version,
                "page_num":          page.page_num,
                "chunk_type":        "prose",
            }]
        )

        for parent_doc in parent_docs:
            parent_id = make_chunk_id(parent_doc.page_content, page.page_num, global_index)
            global_index += 1
            parent_doc.metadata["parent_id"] = parent_id
            parent_doc.metadata["chunk_id"] = parent_id
            parent_doc.metadata["section_heading"] = extract_section_heading(
                parent_doc.page_content
            )
            parents.append(parent_doc)

            child_docs = CHILD_SPLITTER.create_documents(
                [parent_doc.page_content],
                metadatas=[{**parent_doc.metadata, "chunk_type": "child"}]
            )
            for i, child in enumerate(child_docs):
                child.metadata["chunk_id"] = make_chunk_id(
                    child.page_content, page.page_num, global_index + i
                ) + f"_{i}"
                child.metadata["parent_id"] = parent_id
                children.append(child)

    return parents, children