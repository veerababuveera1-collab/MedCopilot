# ============================================================
# AI Research Assistant - Streamlit App (xAI Grok Backend)
# Author: Veera Babu
# ============================================================

import os
import tempfile
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI Research Assistant")
st.caption("Powered by Grok (xAI) + LangChain + FAISS")

# ============================================================
# Load API Key from secrets.toml
# ============================================================

xai_key = st.secrets["XAI_API_KEY"]
os.environ["XAI_API_KEY"] = xai_key

# ============================================================
# Initialize LLM (xAI Grok)
# ============================================================

llm = OpenAI(
    model_name="grok-2",
    temperature=0.2,
    openai_api_base="https://api.x.ai/v1",
    openai_api_key=os.getenv("XAI_API_KEY")
)

# ============================================================
# Initialize Embeddings + Vector DB
# ============================================================

embeddings = OpenAIEmbeddings(
    openai_api_base="https://api.x.ai/v1",
    openai_api_key=os.getenv("XAI_API_KEY")
)

DB_PATH = "faiss_db"

if os.path.exists(DB_PATH):
    vector_db = FAISS.load_local(DB_PATH, embeddings)
else:
    vector_db = FAISS.from_texts(["AI Research Assistant Initialized"], embeddings)
    vector_db.save_local(DB_PATH)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(),
    chain_type="stuff"
)

# ============================================================
# UI Layout
# ============================================================

col1, col2 = st.columns(2)

# ------------------------------
# Research Query Section
# ------------------------------

with col1:
    st.subheader("🔍 Research Query")
    query = st.text_area("Enter your research question:")

    if st.button("Run AI Research"):
        if query.strip() == "":
            st.warning("Please enter a research question.")
        else:
            with st.spinner("Grok AI is analyzing..."):
                answer = qa_chain.run(query)

            st.success("Research Completed")
            st.text_area("📄 AI Research Output:", answer, height=300)

# ------------------------------
# PDF Upload Section
# ------------------------------

with col2:
    st.subheader("📄 Upload Research PDF")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Indexing PDF into AI memory..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            vector_db.add_documents(chunks)
            vector_db.save_local(DB_PATH)

            st.success(f"PDF indexed successfully! Pages processed: {len(chunks)}")

# ============================================================
# Ask from PDF Knowledge Base
# ============================================================

st.divider()
st.subheader("🧠 Ask from PDF Knowledge Base")

kb_query = st.text_input("Ask a question from uploaded PDFs:")

if st.button("Ask AI"):
    if kb_query.strip() == "":
        st.warning("Enter a question.")
    else:
        with st.spinner("Searching knowledge base..."):
            kb_answer = qa_chain.run(kb_query)

        st.text_area("📘 Knowledge Base Answer:", kb_answer, height=250)

# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption("🚀 AI Research Assistant | Grok + LangChain + FAISS | Built by Veera Babu")
