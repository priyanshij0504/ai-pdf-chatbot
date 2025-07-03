from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline
import tempfile
import os

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load and split PDF into chunks
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = splitter.split_documents(docs)
    return chunks

# Store embeddings in Pinecone
def store_embeddings_in_pinecone(chunks, index):
    texts = [doc.page_content for doc in chunks]
    vectors = embeddings.embed_documents(texts)

    pine_vectors = [
        {"id": f"doc-{i}", "values": vectors[i], "metadata": {"text": texts[i]}}
        for i in range(len(vectors))
    ]

    index.upsert(vectors=pine_vectors)

# Search for relevant chunks from Pinecone
def search_similar_chunks(query, index, k=3):
    query_vector = embeddings.embed_query(query)
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)

    return [
        Document(page_content=match["metadata"]["text"], metadata={"score": match["score"]})
        for match in results["matches"]
    ]

# Generate answer using FLAN-T5
def generate_answer(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
    llm = HuggingFacePipeline(pipeline=qa_pipeline)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""

Context:
{context}

Question:
{question}

Answer:"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"context": context, "question": question})
    return response["text"]
