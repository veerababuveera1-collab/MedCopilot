import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from pypdf import PdfReader
import os, time

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="MedCopilot AI Research Assistant",
    layout="wide"
)

st.title("🧠 MedCopilot AI Research Assistant")
st.caption("Hospital-grade Medical Research Copilot — Powered by OpenAI + LangChain + FAISS")

# ---------------------------------------------------
# Load API Key
# ---------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.warning("🔑 OpenAI API key not configured.")
    st.info("Add your API key in Streamlit → Manage App → Secrets")
    st.code('OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"')
    st.stop()

# ---------------------------------------------------
# Initialize Embeddings
# ---------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------
# Sidebar Control Panel
# ---------------------------------------------------
st.sidebar.title("⚙ MedCopilot Control Panel")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Doctor Mode", "Research Mode"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Document Center")

# ---------------------------------------------------
# PDF Upload
# ---------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload Medical Research PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# Safe Batch Indexing (Rate Limit Protected)
# ---------------------------------------------------
if uploaded_files:
    with st.spinner("📚 Reading PDFs..."):
        all_text = ""
        for file in uploaded_files:
            reader = PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_text(all_text)

    st.info(f"📄 Total text chunks: {len(docs)}")

    # Batch embedding to avoid rate limit
    batch_size = 50
    vector_db = None

    progress = st.progress(0)
    total_batches = (len(docs) // batch_size) + 1

    with st.spinner("🔎 Creating medical knowledge index..."):
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]

            if vector_db is None:
                vector_db = FAISS.from_texts(batch, embeddings)
            else:
                vector_db.add_texts(batch)

            progress.progress(min((i + batch_size) / len(docs), 1.0))

            # throttle to avoid OpenAI rate limit
            time.sleep(0.5)

    st.session_state.vector_db = vector_db
    st.sidebar.success("✅ Medical knowledge indexed successfully")

# ---------------------------------------------------
# LLM Engine
# ---------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2
)

# ---------------------------------------------------
# Main Chat UI
# ---------------------------------------------------
st.markdown("## 💬 Medical AI Copilot")

query = st.chat_input("Ask your medical research question...")

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Handle new query
if query:
    if not st.session_state.vector_db:
        st.warning("📂 Please upload research PDFs first.")
    else:
        with st.chat_message("user"):
            st.markdown(query)

        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 4})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        with st.spinner("🧠 MedCopilot is analyzing research..."):
            answer = qa_chain.run(query)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", answer))

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.caption("⚕ MedCopilot AI — Clinical Decision Intelligence System | Designed by Veera Babu")
