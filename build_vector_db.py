"""
build_vector_db.py

One-time (rerun-when-source-docs-change) ingestion script:
1. Load .txt files from data/
2. Split them into overlapping chunks
3. Embed each chunk using Amazon Titan Text Embeddings V2 (via Bedrock)
4. Store chunks + embeddings in a local, persistent ChromaDB collection
"""

import os
import json
import boto3
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm

DATA_DIR = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "space_apps_basics"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 180

bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def embed_text(text):
    """Call Bedrock's Titan Text Embeddings V2 model and return the embedding vector."""
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": 512, "normalize": True}),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def load_documents(data_dir):
    """Return a list of (filename, content) for every .pdf file in data_dir."""
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(data_dir, filename)
            content = pymupdf4llm.to_markdown(path)
            docs.append((filename, content))
    return docs


def main():
    print(f"Loading documents from ./{DATA_DIR} ...")
    docs = load_documents(DATA_DIR)
    print(f"Found {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Start fresh each run so re-running this script doesn't create duplicate chunks
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    chunk_id = 0
    for filename, content in docs:
        chunks = splitter.split_text(content)
        print(f"  {filename}: {len(chunks)} chunk(s)")
        for chunk in chunks:
            embedding = embed_text(chunk)
            collection.add(
                ids=[f"chunk_{chunk_id}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": filename}],
            )
            chunk_id += 1

    print(f"\nDone. {chunk_id} chunks embedded and stored in ./{CHROMA_PATH}")


if __name__ == "__main__":
    main()
