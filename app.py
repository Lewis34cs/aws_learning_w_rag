"""
app.py

A basic RAG (Retrieval-Augmented Generation) Streamlit app:
1. Take a question from the user
2. Embed the question with Titan Text Embeddings V2
3. Retrieve the most relevant chunks from the local ChromaDB vector store
4. Feed those chunks + the question to a Bedrock LLM (via langchain-aws) for the final answer
5. Display the answer, plus the source chunks it was based on
"""

import json
import boto3
import chromadb
import streamlit as st
from langchain_aws import ChatBedrockConverse

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "space_apps_basics"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
LLM_MODEL_ID = "amazon.nova-micro-v1:0"
REGION = "us-east-1"
TOP_K = 5

bedrock = boto3.client("bedrock-runtime", region_name=REGION)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
llm = ChatBedrockConverse(model=LLM_MODEL_ID, region_name=REGION)


def embed_text(text):
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": 512, "normalize": True}),
    )
    return json.loads(response["body"].read())["embedding"]


def retrieve(question, k=TOP_K):
    collection = chroma_client.get_collection(COLLECTION_NAME)
    query_embedding = embed_text(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return list(zip(chunks, sources))


def generate_answer(question, retrieved):
    context = "\n\n".join(chunk for chunk, _ in retrieved)
    prompt = (
        "Answer the question using only the context below. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = llm.invoke(prompt)
    return response.content


st.title("Space Apps basics RAG Demo")
st.write("Ask a question about Space Apps, answered using the docs in data/.")

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Retrieving context and generating an answer..."):
        retrieved = retrieve(question)
        answer = generate_answer(question, retrieved)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Sources used"):
        for chunk, source in retrieved:
            st.markdown(f"**{source}**")
            st.write(chunk)
            st.divider()
