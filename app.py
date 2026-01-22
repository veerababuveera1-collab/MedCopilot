import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import tempfile

# Core libs
import numpy as np

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

import requests
from bs4 import BeautifulSoup
import arxiv
from pathlib import Path
from dotenv import load_dotenv

# Try to support Google Scholar
try:
    from scholarly import scholarly as sch
except ImportError:
    sch = None

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResearchQuery:
    """Structure for research queries"""
    query: str
    topic: str
    depth: str = "moderate"
    sources: List[str] = field(default_factory=lambda: ["arxiv", "scholar"])
    timeframe: str = "recent"


class VectorDatabase:
    """FAISS vector database for semantic search - simplified for reliability"""

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Document] = []
        self.index_path = Path("research_index")

    def create_index(self, documents: List[Document]) -> bool:
        """Create FAISS index from documents"""
        try:
            if not documents:
                logger.warning("No documents provided for indexing")
                return False

            # Create or merge with existing index
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                # Add new documents to existing index
                new_store = FAISS.from_documents(documents, self.embeddings)
                self.vector_store.merge_from(new_store)

            self.documents.extend(documents)
            logger.info(f"Indexed {len(documents)} documents. Total: {len(self.documents)}")
            return True

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False

    def save_index(self) -> bool:
        """Persist the index locally"""
        try:
            if self.vector_store:
                self.vector_store.save_local(str(self.index_path))
                logger.info("Vector database saved")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False

    def load_index(self) -> bool:
        """Load existing index if present"""
        try:
            if self.index_path.exists():
                self.vector_store = FAISS.load_local(
                    str(self.index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector database loaded")
                return True
            logger.info("No existing index found")
            return False
        except Exception as e:
            logger.warning(f"Could not load index: {e}")
            return False

    def similarity_search(self, query: str, k: int = 10) -> List[Document]:
        """Semantic search"""
        if self.vector_store:
            try:
                return self.vector_store.similarity_search(query, k=k)
            except Exception as e:
                logger.error(f"Search error: {e}")
        return []

    def clear(self):
        """Clear the index"""
        self.vector_store = None
        self.documents = []
        if self.index_path.exists():
            import shutil
            shutil.rmtree(self.index_path)
        logger.info("Vector database cleared")


class DocumentProcessor:
    """Process PDFs, web pages, and text into chunks"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Load and chunk a PDF"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Processed PDF: {len(chunks)} chunks from {pdf_path}")
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return []

    def process_pdf_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> List[Document]:
        """Process PDF from bytes (for Streamlit uploads)"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            chunks = self.process_pdf(tmp_path)

            # Add filename to metadata
            for chunk in chunks:
                chunk.metadata["filename"] = filename

            os.unlink(tmp_path)
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDF bytes: {e}")
            return []

    def process_web_content(self, urls: List[str]) -> List[Document]:
        """Fetch and chunk web pages"""
        documents = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                documents.extend(chunks)
                logger.info(f"Processed URL: {len(chunks)} chunks from {url}")
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
        return documents

    def process_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """Convert raw text into chunked Documents"""
        if not text.strip():
            return []
        doc = Document(page_content=text, metadata=metadata or {})
        return self.text_splitter.split_documents([doc])


class ArxivAgent:
    """ArXiv paper fetcher"""

    def __init__(self):
        self.client = arxiv.Client()

    def search_papers(self, query: str, max_results: int = 15) -> List[Dict]:
        """Search ArXiv for papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            papers = []
            for result in self.client.results(search):
                papers.append({
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "summary": result.summary,
                    "url": result.pdf_url,
                    "entry_id": result.entry_id,
                    "published": result.published,
                    "updated": result.updated,
                    "categories": result.categories,
                    "source": "arxiv"
                })

            logger.info(f"ArXiv: Found {len(papers)} papers for '{query}'")
            return papers

        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            return []


class ScholarAgent:
    """Google Scholar fetcher (optional)"""

    def __init__(self):
        self.enabled = sch is not None
        if not self.enabled:
            logger.warning("Google Scholar disabled (install: pip install scholarly)")

    def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search Google Scholar"""
        if not self.enabled:
            return []

        try:
            papers = []
            search_query = sch.search_pubs(query)

            for i, pub in enumerate(search_query):
                if i >= max_results:
                    break

                bib = pub.get("bib", {})
                papers.append({
                    "title": bib.get("title", "Unknown Title"),
                    "authors": bib.get("author", "Unknown"),
                    "summary": bib.get("abstract", "No abstract available"),
                    "url": pub.get("pub_url", ""),
                    "citations": pub.get("num_citations", 0),
                    "year": bib.get("pub_year", ""),
                    "venue": bib.get("venue", ""),
                    "source": "scholar"
                })

            logger.info(f"Scholar: Found {len(papers)} papers for '{query}'")
            return papers

        except Exception as e:
            logger.error(f"Scholar search error: {e}")
            return []


class WebSearchAgent:
    """Web content extraction"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    def extract_content(self, url: str, max_chars: int = 5000) -> str:
        """Extract text from a webpage"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove scripts, styles, nav, footer
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Get text
            text = soup.get_text(separator=" ", strip=True)

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = " ".join(line for line in lines if line)

            return text[:max_chars]

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""


class SynthesisAgent:
    """LLM-powered research synthesis"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.parser = StrOutputParser()

        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research analyst. Synthesize the provided research data 
into a comprehensive, well-structured analysis. Be specific, cite sources where relevant, 
and highlight key insights, patterns, and gaps in the literature."""),
            ("user", """Research Query: {query}

Research Data:
{research_data}

Please provide a comprehensive synthesis including:
1. **Executive Summary**: Key findings in 2-3 sentences
2. **Main Themes**: Major patterns and insights discovered
3. **Key Findings**: Specific important discoveries from the papers
4. **Debates & Contradictions**: Areas where sources disagree
5. **Research Gaps**: What's missing or needs more investigation
6. **Implications**: Practical applications and future directions
7. **Top Sources**: Most relevant papers with brief descriptions

Synthesis:""")
        ])

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful research assistant. Answer questions based on the 
provided context. If the answer isn't in the context, say so clearly."""),
            ("user", """Context:
{context}

Question: {question}

Answer:""")
        ])

    def synthesize(self, research_data: str, query: str) -> str:
        """Generate research synthesis"""
        try:
            chain = self.synthesis_prompt | self.llm | self.parser
            response = chain.invoke({
                "research_data": research_data[:15000],  # Limit context
                "query": query
            })
            return response
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return f"Error generating synthesis: {str(e)}"

    def answer_question(self, context: str, question: str) -> str:
        """Answer a question based on context"""
        try:
            chain = self.qa_prompt | self.llm | self.parser
            response = chain.invoke({
                "context": context[:10000],
                "question": question
            })
            return response
        except Exception as e:
            logger.error(f"QA error: {e}")
            return f"Error answering question: {str(e)}"


class MultiAgentResearchAssistant:
    """Main orchestrator for the research pipeline"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize LLM with ChatOpenAI (correct for newer LangChain)
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=4000
        )

        # Components
        self.vector_db = VectorDatabase()
        self.doc_processor = DocumentProcessor()
        self.arxiv_agent = ArxivAgent()
        self.scholar_agent = ScholarAgent()
        self.web_agent = WebSearchAgent()
        self.synthesis_agent = SynthesisAgent(self.llm)

        # Chat history for follow-up questions
        self.chat_history: List[Dict[str, str]] = []
        self.last_results: Optional[Dict] = None

        # Load existing index
        self.vector_db.load_index()

        logger.info("Multi-Agent Research Assistant initialized")

    def research(self, query: ResearchQuery) -> Dict[str, Any]:
        """
        Execute the full research pipeline:
        1. Gather sources from arxiv/scholar
        2. Process and embed documents
        3. Retrieve relevant chunks
        4. Synthesize findings
        """
        results = {
            "query": query.query,
            "timestamp": datetime.now().isoformat(),
            "sources": {
                "arxiv": [],
                "scholar": [],
                "web": [],
            },
            "synthesis": "",
            "recommendations": [],
            "error": None
        }

        try:
            # Phase 1: Gather papers
            logger.info("Phase 1: Gathering sources...")

            if "arxiv" in query.sources:
                results["sources"]["arxiv"] = self.arxiv_agent.search_papers(
                    query.query, max_results=15
                )

            if "scholar" in query.sources:
                results["sources"]["scholar"] = self.scholar_agent.search_papers(
                    query.query, max_results=10
                )

            # Phase 2: Process into documents
            logger.info("Phase 2: Processing documents...")

            new_documents = []

            for paper in results["sources"]["arxiv"]:
                doc_text = self._format_paper(paper)
                docs = self.doc_processor.process_text(doc_text, {
                    "source": "arxiv",
                    "title": paper["title"],
                    "url": paper.get("url", ""),
                    "published": str(paper.get("published", "")),
                })
                new_documents.extend(docs)

            for paper in results["sources"]["scholar"]:
                doc_text = self._format_paper(paper)
                docs = self.doc_processor.process_text(doc_text, {
                    "source": "scholar",
                    "title": paper["title"],
                    "citations": paper.get("citations", 0),
                    "year": paper.get("year", ""),
                })
                new_documents.extend(docs)

            if new_documents:
                self.vector_db.create_index(new_documents)
                self.vector_db.save_index()
                logger.info(f"Indexed {len(new_documents)} document chunks")

            # Phase 3: Retrieve relevant content
            logger.info("Phase 3: Semantic retrieval...")

            relevant_docs = self.vector_db.similarity_search(query.query, k=20)

            # Phase 4: Synthesize
            logger.info("Phase 4: Synthesizing findings...")

            research_data = self._format_retrieved_docs(relevant_docs)
            results["synthesis"] = self.synthesis_agent.synthesize(
                research_data, query.query
            )

            # Phase 5: Generate recommendations
            results["recommendations"] = self._generate_recommendations(
                results["sources"], query
            )

            # Store for follow-up questions
            self.last_results = results
            self.chat_history.append({
                "role": "user",
                "content": query.query
            })
            self.chat_history.append({
                "role": "assistant",
                "content": results["synthesis"]
            })

            logger.info("Research completed successfully")

        except Exception as e:
            logger.error(f"Research error: {e}")
            results["error"] = str(e)

        return results

    def ask_followup(self, question: str) -> str:
        """Answer a follow-up question using the current knowledge base"""
        relevant_docs = self.vector_db.similarity_search(question, k=10)

        if not relevant_docs:
            return "I don't have enough context to answer that. Please run a research query first."

        context = self._format_retrieved_docs(relevant_docs)
        answer = self.synthesis_agent.answer_question(context, question)

        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer

    def add_pdf(self, pdf_bytes: bytes, filename: str) -> int:
        """Add a PDF to the knowledge base"""
        docs = self.doc_processor.process_pdf_bytes(pdf_bytes, filename)
        if docs:
            self.vector_db.create_index(docs)
            self.vector_db.save_index()
        return len(docs)

    def _format_paper(self, paper: Dict) -> str:
        """Format paper metadata into text for embedding"""
        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors = ", ".join(authors[:5])  # Limit authors
            if len(paper.get("authors", [])) > 5:
                authors += " et al."

        parts = [
            f"Title: {paper.get('title', 'Unknown')}",
            f"Authors: {authors}",
        ]

        if paper.get("year"):
            parts.append(f"Year: {paper['year']}")
        if paper.get("published"):
            parts.append(f"Published: {paper['published']}")
        if paper.get("citations"):
            parts.append(f"Citations: {paper['citations']}")

        parts.append(f"Summary: {paper.get('summary', 'No summary available')}")

        return "\n".join(parts)

    def _format_retrieved_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for synthesis"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = f"[Source {i}]"
            if meta.get("title"):
                header += f" {meta['title']}"
            if meta.get("source"):
                header += f" ({meta['source']})"

            formatted.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted)

    def _generate_recommendations(self, sources: Dict, query: ResearchQuery) -> List[str]:
        """Generate research recommendations"""
        recommendations = []

        total = len(sources.get("arxiv", [])) + len(sources.get("scholar", []))

        if total == 0:
            recommendations.append(
                "⚠️ No papers found. Try different search terms or check your query."
            )
        elif total < 5:
            recommendations.append(
                "📚 Limited sources found. Consider broadening your search terms."
            )
        elif total > 20:
            recommendations.append(
                "📚 Rich literature available. Consider narrowing focus for deeper analysis."
            )

        # Check for recent papers
        arxiv_papers = sources.get("arxiv", [])
        if arxiv_papers:
            recent = sum(
                1 for p in arxiv_papers
                if p.get("published") and
                (datetime.now() - p["published"].replace(tzinfo=None)).days < 365
            )
            if recent < 3:
                recommendations.append(
                    "📅 Few recent papers found. This may be an established field or emerging area."
                )
            else:
                recommendations.append(
                    f"📅 Found {recent} papers from the last year - active research area."
                )

        # Scholar-specific
        scholar_papers = sources.get("scholar", [])
        if scholar_papers:
            high_cited = [p for p in scholar_papers if p.get("citations", 0) > 100]
            if high_cited:
                recommendations.append(
                    f"🌟 Found {len(high_cited)} highly-cited papers (100+ citations)."
                )

        recommendations.append(
            "💡 Use follow-up questions to explore specific aspects of the research."
        )

        return recommendations


def create_streamlit_interface():
    """Streamlit UI"""
    import streamlit as st

    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 AI-Powered Multi-Agent Research Assistant")
    st.markdown("*Discover, analyze, and synthesize research papers intelligently*")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Key handling
        api_key = os.getenv("OPENAI_API_KEY", "")

        if not api_key:
            try:
                api_key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                pass

        if not api_key:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Get your key at https://platform.openai.com/api-keys"
            )

        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API Key")
            st.stop()

        st.success("✅ API Key configured")

        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
            help="gpt-4o-mini is fastest and cheapest"
        )

        st.markdown("---")

        # Knowledge base management
        st.subheader("📚 Knowledge Base")

        if st.button("🗑️ Clear Knowledge Base"):
            if "assistant" in st.session_state:
                st.session_state.assistant.vector_db.clear()
                st.success("Knowledge base cleared!")

        st.markdown("---")
        st.markdown("**Built with:**")
        st.markdown("- LangChain 0.2+")
        st.markdown("- OpenAI GPT")
        st.markdown("- FAISS Vector DB")
        st.markdown("- ArXiv API")

    # Initialize assistant
    if "assistant" not in st.session_state or st.session_state.get("model") != model:
        st.session_state.assistant = MultiAgentResearchAssistant(api_key, model)
        st.session_state.model = model

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Research", "💬 Ask Questions", "📄 Upload PDFs"])

    with tab1:
        st.header("Research Query")

        query_text = st.text_area(
            "Enter your research question:",
            height=100,
            placeholder="e.g., What are recent advances in transformer architectures for computer vision?"
        )

        col1, col2 = st.columns(2)

        with col1:
            sources = st.multiselect(
                "Sources",
                ["arxiv", "scholar"],
                default=["arxiv"],
                help="Scholar requires 'scholarly' package"
            )

        with col2:
            depth = st.selectbox(
                "Search Depth",
                ["surface", "moderate", "deep"],
                index=1
            )

        if st.button("🚀 Start Research", type="primary", use_container_width=True):
            if not query_text:
                st.error("Please enter a research query.")
            else:
                with st.spinner("🔍 Searching and analyzing papers..."):
                    research_query = ResearchQuery(
                        query=query_text,
                        topic=query_text,
                        depth=depth,
                        sources=sources,
                    )

                    results = st.session_state.assistant.research(research_query)
                    st.session_state.results = results

        # Display results
        if "results" in st.session_state:
            results = st.session_state.results

            if results.get("error"):
                st.error(f"Error: {results['error']}")
            else:
                # Stats
                col1, col2, col3 = st.columns(3)
                col1.metric("ArXiv Papers", len(results["sources"]["arxiv"]))
                col2.metric("Scholar Papers", len(results["sources"]["scholar"]))
                col3.metric("Total Sources", 
                           len(results["sources"]["arxiv"]) + len(results["sources"]["scholar"]))

                # Recommendations
                if results["recommendations"]:
                    st.subheader("💡 Recommendations")
                    for rec in results["recommendations"]:
                        st.info(rec)

                # Synthesis
                st.subheader("📋 Research Synthesis")
                st.markdown(results["synthesis"])

                # Papers
                st.subheader("📚 Papers Found")

                arxiv_tab, scholar_tab = st.tabs([
                    f"ArXiv ({len(results['sources']['arxiv'])})",
                    f"Scholar ({len(results['sources']['scholar'])})"
                ])

                with arxiv_tab:
                    for paper in results["sources"]["arxiv"]:
                        with st.expander(f"📄 {paper['title']}"):
                            st.write(f"**Authors:** {', '.join(paper['authors'][:5])}")
                            st.write(f"**Published:** {paper['published']}")
                            st.write(f"**Categories:** {', '.join(paper['categories'])}")
                            st.write(f"**Summary:** {paper['summary'][:500]}...")
                            if paper.get("url"):
                                st.link_button("📥 PDF", paper["url"])

                with scholar_tab:
                    if not results["sources"]["scholar"]:
                        st.info("Scholar search not enabled or no results found.")
                    for paper in results["sources"]["scholar"]:
                        with st.expander(f"📄 {paper['title']}"):
                            st.write(f"**Authors:** {paper['authors']}")
                            st.write(f"**Year:** {paper.get('year', 'N/A')}")
                            st.write(f"**Citations:** {paper.get('citations', 'N/A')}")
                            st.write(f"**Summary:** {paper.get('summary', 'N/A')[:500]}")
                            if paper.get("url"):
                                st.link_button("🔗 Link", paper["url"])

    with tab2:
        st.header("Ask Follow-up Questions")
        st.markdown("Ask questions about the papers in your knowledge base.")

        question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the main limitations mentioned in these papers?"
        )

        if st.button("Ask", type="primary"):
            if question:
                with st.spinner("Thinking..."):
                    answer = st.session_state.assistant.ask_followup(question)
                    st.markdown("### Answer")
                    st.markdown(answer)
            else:
                st.warning("Please enter a question.")

        # Chat history
        if st.session_state.assistant.chat_history:
            st.markdown("---")
            st.subheader("💬 Chat History")
            for msg in st.session_state.assistant.chat_history[-10:]:
                role = "🧑" if msg["role"] == "user" else "🤖"
                with st.expander(f"{role} {msg['content'][:50]}..."):
                    st.markdown(msg["content"])

    with tab3:
        st.header("Upload PDFs")
        st.markdown("Add your own papers to the knowledge base.")

        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("📤 Process PDFs", type="primary"):
                total_chunks = 0
                progress = st.progress(0)

                for i, file in enumerate(uploaded_files):
                    with st.spinner(f"Processing {file.name}..."):
                        chunks = st.session_state.assistant.add_pdf(
                            file.read(),
                            file.name
                        )
                        total_chunks += chunks
                        progress.progress((i + 1) / len(uploaded_files))

                st.success(f"✅ Added {total_chunks} chunks from {len(uploaded_files)} PDFs")


def main():
    """Entry point"""
    try:
        import streamlit as st
        create_streamlit_interface()
    except ImportError:
        print("=" * 60)
        print("AI-Powered Multi-Agent Research Assistant")
        print("=" * 60)
        print("\nTo run the web interface:")
        print("  pip install streamlit")
        print("  streamlit run ai_research_assistant.py")
        print("\nRunning CLI demo...")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Enter your OpenAI API key: ")

        assistant = MultiAgentResearchAssistant(api_key)

        query = ResearchQuery(
            query="transformer architecture optimization techniques",
            topic="machine learning",
            depth="moderate",
            sources=["arxiv"]
        )

        print(f"\nSearching for: {query.query}")
        results = assistant.research(query)

        print("\n" + "=" * 60)
        print("SYNTHESIS")
        print("=" * 60)
        print(results["synthesis"])

        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        for rec in results["recommendations"]:
            print(f"  • {rec}")


if __name__ == "__main__":
    main()
