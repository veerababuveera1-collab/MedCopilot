import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from pypdf import PdfReader
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="MedCopilot AI Research Assistant",
    layout="wide"
)

st.title("🧠 MedCopilot AI Research Assistant")
st.caption("Powered by OpenAI + LangChain + FAISS")

# ---------------- Load API Key ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OpenAI API key not found in Streamlit Secrets")
    st.stop()

# ---------------- Embeddings ----------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------- Session State ----------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ---------------- PDF Upload ----------------
uploaded_files = st.file_uploader(
    "Upload Medical Research PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Indexing documents..."):
        all_text = ""
        for file in uploaded_files:
            reader = PdfReader(file)
            for page in reader.pages:
                all_text += page.extract_text()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        docs = splitter.split_text(all_text)
        vector_db = FAISS.from_texts(docs, embeddings)
        st.session_state.vector_db = vector_db

        st.success("✅ Documents indexed successfully")

# ---------------- Chat Engine ----------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
    streaming=True
)

# ---------------- Query UI ----------------
query = st.text_input("Ask your medical research question")

if query:
    if not st.session_state.vector_db:
        st.warning("Upload PDFs first")
    else:
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        with st.spinner("Thinking..."):
            answer = qa_chain.run(query)

        st.markdown("### 🧾 Answer")
        st.write(answer)
