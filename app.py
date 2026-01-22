import streamlit as st
import os, requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# ---------------------------------------------------
# FORCE EARLY RENDER (CRITICAL FOR STREAMLIT CLOUD)
# ---------------------------------------------------
st.write("")  # prevents healthz failure

# ---------------------------------------------------
# MAIN APP (CLOUD SAFE)
# ---------------------------------------------------
def main():

    # ---------------- Page Config ----------------
    st.set_page_config(
        page_title="MedCopilot Research OS",
        layout="wide"
    )

    st.title("🧠 MedCopilot AI Research Assistant")
    st.caption(
        "Hospital-grade Medical Research Copilot — Powered by OpenAI + LangChain + FAISS"
    )

    # ---------------- API KEY ----------------
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        st.warning("🔑 OpenAI API key not configured")
        st.info("Streamlit → Manage App → Secrets లో API key add చేయండి")
        st.code('OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"')
        st.stop()

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("⚙ MedCopilot Control Panel")

    mode = st.sidebar.selectbox(
        "Select Mode",
        ["Doctor Mode", "Research Mode"]
    )

    st.sidebar.markdown("### 📂 Document Center")

    # ---------------- INIT MODELS ----------------
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # ---------------- SESSION STATE ----------------
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------------- SAFE PDF READER ----------------
    def safe_read_pdf(uploaded_file):
        text_content = ""
        try:
            reader = PdfReader(uploaded_file, strict=False)
            for page in reader.pages:
                try:
                    text = page.extract_text()
                    if text:
                        text_content += text + "\n"
                except Exception:
                    continue
        except PdfReadError:
            st.warning(f"⚠ {uploaded_file.name} లో కొన్ని pages corrupted")
        except Exception:
            st.warning(f"⚠ {uploaded_file.name} read చేయలేకపోయాం")
        return text_content

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs([
        "🔍 Research",
        "💬 Ask Questions",
        "📄 Upload PDFs"
    ])

    # =================================================
    # 🔍 RESEARCH TAB (BUTTON-GUARDED)
    # =================================================
    with tab1:
        st.subheader("🔬 AI-Powered Research Assistant")

        research_query = st.text_input(
            "Research Query",
            placeholder="Enter your research question"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            source = st.selectbox("Sources", ["arxiv"])
        with col2:
            depth = st.selectbox("Search Depth", ["light", "moderate", "deep"])
        with col3:
            paper_count = st.slider("ArXiv Papers", 5, 30, 15)

        if st.button("🚀 Run Research"):

            if not research_query.strip():
                st.warning("⚠ Please enter a research query")
                st.stop()

            with st.spinner("🔍 Searching research papers..."):
                try:
                    url = "http://export.arxiv.org/api/query"
                    params = {
                        "search_query": f"all:{research_query}",
                        "start": 0,
                        "max_results": paper_count
                    }
                    response = requests.get(url, params=params, timeout=10)
                except Exception:
                    st.error("❌ ArXiv service unavailable")
                    st.stop()

                papers = []
                if response.status_code == 200:
                    entries = response.text.split("<entry>")[1:]
                    for entry in entries:
                        title = entry.split("<title>")[1].split("</title>")[0].strip()
                        summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
                        papers.append({"title": title, "summary": summary})

            st.success(f"📄 Found {len(papers)} papers")

            research_text = "\n\n".join(
                [p["title"] + "\n" + p["summary"] for p in papers]
            )

            with st.spinner("🧠 Synthesizing research..."):
                prompt = f"""
You are a medical research analyst.

Generate:
- Executive Summary
- Main Themes
- Key Findings
- Research Gaps
- Implications
- Recommendations

Research Content:
{research_text}
"""
                synthesis = llm.invoke(prompt).content

            st.markdown("## 📋 Research Synthesis")
            st.write(synthesis)

            st.markdown("## 📚 Papers Found")
            for p in papers:
                st.markdown(f"📄 **{p['title']}**")

    # =================================================
    # 💬 ASK QUESTIONS TAB (RAG)
    # =================================================
    with tab2:
        st.subheader("💬 Ask Medical Questions")

        query = st.text_input("Ask your medical research question")

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        if query and query.strip():
            if not st.session_state.vector_db:
                st.warning("📄 Upload PDFs first")
            else:
                with st.chat_message("user"):
                    st.markdown(query)

                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": 4}
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff"
                )

                with st.spinner("🧠 Analyzing medical knowledge..."):
                    answer = qa_chain.run(query)

                with st.chat_message("assistant"):
                    st.markdown(answer)

                st.session_state.chat_history.append(("user", query))
                st.session_state.chat_history.append(("assistant", answer))

    # =================================================
    # 📄 UPLOAD PDFs TAB (SAFE)
    # =================================================
    with tab3:
        st.subheader("📂 Upload Medical Research PDFs")

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            with st.spinner("📚 Reading PDFs safely..."):
                all_text = ""
                for file in uploaded_files:
                    all_text += safe_read_pdf(file)

            if not all_text.strip():
                st.warning("⚠ No readable text found in PDFs")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_text(all_text)

            with st.spinner("🔎 Building medical knowledge base..."):
                vector_db = FAISS.from_texts(docs, embeddings)
                st.session_state.vector_db = vector_db

            st.success("✅ Knowledge base ready")

    # ---------------- FOOTER ----------------
    st.markdown("---")
    st.caption("⚕ MedCopilot Research OS | Designed by Veera Babu")


# ---------------------------------------------------
# ENTRY POINT (CRITICAL FOR CLOUD)
# ---------------------------------------------------
if __name__ == "__main__":
    main()
