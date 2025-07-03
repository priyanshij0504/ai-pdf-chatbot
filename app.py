import streamlit as st
from utils import (
    process_pdf,
    store_embeddings_in_pinecone,
    search_similar_chunks,
    generate_answer
)
import os
from pinecone import Pinecone

# 🔐 Set API keys (put your keys here)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_7EdYGU_SBWsjsJnVyaY5XYvcdt96tBao69aJkTpxtykKZTnfNk8VB5m6GJVPgyAiK5vQwb"
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"

# Initialize Pinecone
pc = Pinecone(api_key="pcsk_7EdYGU_SBWsjsJnVyaY5XYvcdt96tBao69aJkTpxtykKZTnfNk8VB5m6GJVPgyAiK5vQwb")
index = pc.Index("langchainpinecone")

# Streamlit UI
st.title("AI PDF Chatbot")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for file in uploaded_files:
        chunks = process_pdf(file)
        all_chunks.extend(chunks)

    store_embeddings_in_pinecone(all_chunks, index)
    st.success(f"Uploaded and processed {len(uploaded_files)} PDFs.")

query = st.text_input("Ask a question based on uploaded PDFs:")

if query:
    top_chunks = search_similar_chunks(query, index)

    if not top_chunks:
        st.warning("Sorry, I couldn't find relevant information in the uploaded PDFs.")
    else:
        answer = generate_answer(query, top_chunks)
        st.markdown("### Answer:")
        st.write(answer)

        with st.expander("Source Context"):
            for doc in top_chunks:
                st.write(doc.page_content)

