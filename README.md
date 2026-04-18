# MedQuery RAG

A production-grade Retrieval Augmented Generation system for 
clinical guideline Q&A, built on NICE, WHO, and CDC guidelines.

## Architecture

- **Phase 1** — PDF ingestion with parent-child chunking
- **Phase 2** — Hybrid retrieval (BM25 + vector + RRF fusion + Cohere reranking)
- **Phase 3** — LLM generation with citation injection + RAGAS evaluation (in progress)

## Stack

- ChromaDB — vector store
- Cohere — embeddings + reranking
- LangChain — document processing
- RAGAS — evaluation framework

## Results

Retrieval test suite: 8/9 pass (89%) on first run, zero tuning.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your Cohere API key
python run_ingestion.py
python test_retrieval_full.py
```
## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Cohere API key
4. Download PDFs to `data/raw/` (see sources.py for URLs)
5. Run ingestion: `python run_ingestion.py`
6. Test retrieval: `python test_retrieval_full.py`

> `documents/parents.json` and `documents/registry.json` are generated 
> automatically by the ingestion pipeline — do not commit them.
