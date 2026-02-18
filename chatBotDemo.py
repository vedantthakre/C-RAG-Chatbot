import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG Demo", layout="wide")

st.title("📚 C++ RAG Chatbot")

@st.cache_resource
def load_vectorstore():

    # 1️⃣ Load document
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8")
    documents = loader.load()

    # 2️⃣ Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    finalDocuments = text_splitter.split_documents(documents)

    # 3️⃣ Embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4️⃣ Create FAISS vectorstore
    vectorstore = FAISS.from_documents(finalDocuments, embedding)

    return vectorstore


# Load vectorstore
vs = load_vectorstore()

# User input
query = st.text_input("Enter your question:")

if query:
    results = vs.similarity_search(query, k=3)

    st.write("### Retrieved Results:")
    for i, doc in enumerate(results):
        st.write(f"**Result {i+1}:**")
        st.write(doc.page_content[:300])
        st.write("---")

