from dotenv import load_dotenv
load_dotenv()

import os
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.CohereEmbeddingFunction(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name="embed-english-v3.0"
)

client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("medquery_children", embedding_function=ef)

results = col.query(
    query_texts=["first line treatment for gonorrhoea"],
    n_results=3
)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"Page {meta['page_num']} | {meta['section_heading']}")
    print(doc[:150])
    print()