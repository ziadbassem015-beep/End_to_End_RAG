"""
streamlit_app.py
================
Streamlit user interface for the Interactive Book QA RAG Tool.
Implements a production-grade dark theme, custom fonts, visual progress tracking,
and complete RAG pipelines (Upload, OCR, Chunker, Cache, Hybrid, Rerank, LLM).
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import numpy as np

# Configure Logging to console only (safe for Windows)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Core RAG imports
from rag_system.ingestion.loader import PDFLoader
from rag_system.ingestion.chunker import DocumentChunker
from rag_system.embeddings.embedder import Embedder
from rag_system.vectorstore.faiss_store import FAISSStore
from rag_system.retrieval.retriever import BM25Index, HybridRetriever, RetrievalResult
from rag_system.retrieval.reranker import CrossEncoderReranker, BGEReranker
from rag_system.llm.generator import LLMGenerator

# ─── Page Settings ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interactive Book QA RAG Tool",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS (Premium Aesthetics) ─────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient styling */
    .main-title {
        background: linear-gradient(90deg, #a29bfe, #6c5ce7, #00cec9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #b2bec3;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Styles */
    .glass-card {
        background: rgba(23, 26, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Citation/Source Badge styling */
    .source-badge {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .score-badge {
        background: rgba(0, 206, 201, 0.15);
        color: #00cec9;
        border: 1px solid rgba(0, 206, 201, 0.3);
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    /* Custom Chat bubbles */
    .chat-bubble-user {
        background-color: #2d3436;
        border-left: 4px solid #6c5ce7;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .chat-bubble-assistant {
        background-color: #1e272e;
        border-left: 4px solid #00cec9;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Directories Setup ───────────────────────────────────────────────────────
UPLOAD_DIR = Path("data/uploads")
INDEX_DIR = Path("data/indices/faiss")
CACHE_DIR = Path("data/cache/embeddings")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ─── Helper Functions ────────────────────────────────────────────────────────

def check_keys() -> Dict[str, Optional[str]]:
    """Determine configured API keys from environment or user inputs."""
    keys = {
        "github": os.getenv("GITHUB_TOKEN"),
        "openai": os.getenv("OPENAI_API_KEY")
    }
    return keys

def get_llm_generator(api_key: str, provider: str, model_name: str) -> LLMGenerator:
    """Instantiate the LLM Generator with user configs."""
    return LLMGenerator(api_key=api_key, provider=provider, model_name=model_name)

def process_book_pipeline(
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    ocr_mode: str,
    embedding_model: str,
) -> bool:
    """Run the complete ingestion pipeline on the uploaded book."""
    try:
        # Step 1: Parse OCR settings
        ocr_fallback = False
        min_chars_for_ocr = 0.5
        if ocr_mode == "Force OCR":
            ocr_fallback = True
            min_chars_for_ocr = 0.0  # Trigger OCR for every page
        elif ocr_mode == "Auto-detect scanned pages":
            ocr_fallback = True
            min_chars_for_ocr = 0.5

        # Step 2: Load Document
        with st.status("Processing Book...", expanded=True) as status:
            status.update(label="1. Loading document pages...")
            loader = PDFLoader(
                pdf_path=file_path,
                ocr_fallback=ocr_fallback,
                min_chars_for_ocr=min_chars_for_ocr
            )
            try:
                documents = loader.load()
            except Exception as e:
                # Catch Tesseract missing error to explain gracefully
                if "tesseract" in str(e).lower() or "pytesseract" in str(e).lower():
                    st.error(
                        "OCR fallback failed because Tesseract OCR engine is not installed on this system. "
                        "Please download and install Tesseract, or change the OCR Mode to 'Disable OCR'."
                    )
                raise e

            st.write(f"Loaded {len(documents)} pages.")

            # Step 3: Chunking
            status.update(label="2. Splitting text into chunks...")
            chunker = DocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = chunker.chunk(documents)
            chunk_dicts = [c.to_dict() for c in chunks]
            st.write(f"Created {len(chunk_dicts)} distinct text chunks.")

            # Step 4: Embedding
            status.update(label="3. Generating embeddings (utilizing disk cache)...")
            embedder = Embedder(
                model_name=embedding_model,
                cache_dir=CACHE_DIR
            )
            embedded_chunks = embedder.embed(chunk_dicts)
            st.write(f"Generated embeddings. Cache hits occurred if previously computed.")

            # Step 5: FAISS Storage
            status.update(label="4. Saving to vector database...")
            # Detect dimensions dynamically
            dim = len(embedded_chunks[0]["embedding"]) if embedded_chunks else 384
            vector_store = FAISSStore(dimension=dim)
            vector_store.add_batch(embedded_chunks)
            # Save to disk for state persistence
            vector_store.save(str(INDEX_DIR))
            st.write("FAISS Index saved successfully.")

            # Step 6: Create BM25 Index
            status.update(label="5. Building keyword search (BM25) index...")
            bm25_index = BM25Index(embedded_chunks)
            
            # Save into session state
            st.session_state["vector_store"] = vector_store
            st.session_state["bm25_index"] = bm25_index
            st.session_state["processed_book_name"] = file_path.name
            st.session_state["embedding_model_used"] = embedding_model
            
            status.update(label="Ingestion complete!", state="complete")
        
        return True
    except Exception as e:
        st.error(f"In-pipeline error occurred: {str(e)}")
        logger.error("Error processing book: %s", e, exc_info=True)
        return False

# ─── Load Existing Index on startup ──────────────────────────────────────────
def try_load_existing_index() -> None:
    """Attempts to load a saved index from disk on app reload."""
    if (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "metadata.json").exists():
        try:
            # We can reconstruct dimension from metadata file
            import json
            with open(INDEX_DIR / "metadata.json", "r", encoding="utf-8") as f:
                state = json.load(f)
            dim = state.get("dimension", 384)
            
            vector_store = FAISSStore(dimension=dim)
            vector_store.load(str(INDEX_DIR))
            
            # Reconstruct BM25 Index from FAISS metadata
            chunks = list(vector_store.metadata.values())
            if chunks:
                bm25_index = BM25Index(chunks)
                st.session_state["vector_store"] = vector_store
                st.session_state["bm25_index"] = bm25_index
                
                # Derive book name from metadata
                sample_chunk = chunks[0]
                source_name = sample_chunk.get("source", "Indexed Book")
                st.session_state["processed_book_name"] = source_name
                st.session_state["embedding_model_used"] = sample_chunk.get("embedding_model", "BAAI/bge-small-en-v1.5")
                logger.info("Successfully restored index for %s", source_name)
        except Exception as e:
            logger.warning("Could not restore saved index: %s", e)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "processed_book_name" not in st.session_state:
    try_load_existing_index()

# ─── Sidebar Configuration ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/book.png", width=60)
st.sidebar.markdown("### RAG Control Center")

# API Keys Configuration
env_keys = check_keys()
st.sidebar.markdown("#### 1. Authentication")
provider_choice = st.sidebar.selectbox("LLM Provider", ["GitHub Models", "OpenAI"])

resolved_api_key = ""
if provider_choice == "GitHub Models":
    env_token = env_keys["github"]
    key_input = st.sidebar.text_input(
        "GitHub Token",
        value=env_token or "",
        type="password",
        placeholder="github_pat_...",
        help="Reads GITHUB_TOKEN from env if present. Create one under GitHub developer settings."
    )
    resolved_api_key = key_input
    model_options = ["gpt-4o", "gpt-4o-mini", "meta-llama-3-8b-instruct", "cohere-command-r-plus"]
else:
    env_token = env_keys["openai"]
    key_input = st.sidebar.text_input(
        "OpenAI API Key",
        value=env_token or "",
        type="password",
        placeholder="sk-...",
        help="Reads OPENAI_API_KEY from env if present."
    )
    resolved_api_key = key_input
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

selected_llm_model = st.sidebar.selectbox("LLM Model", model_options)
llm_temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1, step=0.05)

# Book Upload & Ingestion Configurations
st.sidebar.markdown("---")
st.sidebar.markdown("#### 2. Book Processing Settings")
uploaded_file = st.sidebar.file_uploader("Upload Book (PDF/TXT)", type=["pdf", "txt"])

ocr_mode = st.sidebar.selectbox(
    "OCR Mode",
    ["Disable OCR", "Auto-detect scanned pages", "Force OCR"],
    help="Enable OCR fallback if you upload scanned books."
)

chunk_size = st.sidebar.slider("Chunk Size (characters)", 200, 2000, 800, step=50)
chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", 0, 500, 150, step=10)

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-small-v2"]
)

# Trigger Ingestion
if uploaded_file is not None:
    file_label = f"Process '{uploaded_file.name}'"
    if st.sidebar.button(file_label, type="primary"):
        # Save uploaded file
        temp_path = UPLOAD_DIR / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Clear chat history for the new book
        st.session_state["chat_history"] = []
        
        success = process_book_pipeline(
            file_path=temp_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ocr_mode=ocr_mode,
            embedding_model=embedding_model
        )
        if success:
            st.sidebar.success(f"Loaded and indexed: {uploaded_file.name}")
            st.rerun()

# Search and Retriever Configurations
st.sidebar.markdown("---")
st.sidebar.markdown("#### 3. Retrieval Parameters")
top_k = st.sidebar.slider("Retrieve Top K", 1, 20, 5, step=1)
hybrid_alpha = st.sidebar.slider(
    "Hybrid Alpha (Dense Weight)",
    0.0, 1.0, 0.7, step=0.05,
    help="1.0 = Pure Semantic Search, 0.0 = Pure Keyword Search (BM25)"
)

use_reranker = st.sidebar.toggle("Use Reranker", value=True)
reranker_choice = st.sidebar.selectbox(
    "Reranker Model",
    ["Cross-Encoder (ms-marco-MiniLM)", "BGE Reranker (bge-reranker-base)"],
    disabled=not use_reranker
)

# Reset Button
if st.sidebar.button("Clear Index & Chat"):
    st.session_state.clear()
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
    st.success("Cleared environment index.")
    st.rerun()

# ─── Main Panel ──────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>📖 Interactive Book QA RAG Tool</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload any textbook or document, process it with a hybrid retrieval pipeline, "
    "and query it using state-of-the-art LLMs. Built with production-grade modular components.</div>",
    unsafe_allow_html=True
)

# Status card
if "processed_book_name" in st.session_state:
    st.markdown(f"""
    <div class='glass-card'>
        <strong>Active Book:</strong> {st.session_state['processed_book_name']} <br>
        <strong>Embedding Model:</strong> {st.session_state['embedding_model_used']} <br>
        <strong>Stored Chunks:</strong> {st.session_state['vector_store'].count()}
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("👈 Please upload and process a book in the sidebar control center to get started!")

# ─── Chat and Query Interface ────────────────────────────────────────────────

# Verify that both index and API keys are set up
index_ready = "vector_store" in st.session_state and "bm25_index" in st.session_state
api_ready = bool(resolved_api_key.strip())

if index_ready:
    # Initialize Reranker if selected
    reranker_obj = None
    if use_reranker:
        if reranker_choice == "Cross-Encoder (ms-marco-MiniLM)":
            reranker_obj = CrossEncoderReranker()
        else:
            reranker_obj = BGEReranker()

    # Reconstruct Retriever
    retriever = HybridRetriever(
        vector_store=st.session_state["vector_store"],
        bm25_index=st.session_state["bm25_index"],
        model_name=st.session_state["embedding_model_used"],
        alpha=hybrid_alpha,
        reranker=reranker_obj
    )

    # Reconstruct LLM Generator
    if api_ready:
        provider_id = "github" if provider_choice == "GitHub Models" else "openai"
        try:
            generator = get_llm_generator(
                api_key=resolved_api_key,
                provider=provider_id,
                model_name=selected_llm_model
            )
        except Exception as e:
            st.error(f"Failed to load LLM Generator: {e}")
            generator = None
    else:
        st.warning("⚠️ LLM Key missing! You can retrieve relevant chunks, but LLM Answering is disabled. Fill in your Token in the sidebar.")
        generator = None

    # Display chat history
    for chat in st.session_state["chat_history"]:
        role = chat["role"]
        content = chat["content"]
        
        if role == "user":
            st.markdown(f"<div class='chat-bubble-user'><strong>You:</strong><br>{content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-assistant'><strong>Assistant:</strong><br>{content}</div>", unsafe_allow_html=True)
            
            # Show sources for assistant answers if saved in state
            if "sources" in chat and chat["sources"]:
                with st.expander("Show Answer Sources & Match Details"):
                    for idx, src in enumerate(chat["sources"], 1):
                        st.markdown(
                            f"<span class='source-badge'>Source {idx}</span> "
                            f"Page {src['page']} of {src['source']} "
                            f"(Rank: {src['rank']} | Score: {src['score']:.4f})",
                            unsafe_allow_html=True
                        )
                        st.text_area(f"Snippet from Page {src['page']}", src["content"], height=100, key=f"src_box_{chat['query_id']}_{idx}")

    # Query Input
    query_input = st.chat_input("Ask a question about the book...")
    
    if query_input:
        # Display user bubble immediately
        st.markdown(f"<div class='chat-bubble-user'><strong>You:</strong><br>{query_input}</div>", unsafe_allow_html=True)
        
        with st.spinner("Retrieving relevant passages..."):
            # Step 1: Retrieve context chunks
            retrieved_results: List[RetrievalResult] = retriever.retrieve(query_input, top_k=top_k)
        
        sources_to_store = []
        for r in retrieved_results:
            sources_to_store.append({
                "chunk_id": r.chunk_id,
                "rank": r.rank,
                "score": r.score,
                "content": r.content,
                "source": r.source,
                "page": r.page,
                "semantic_score": r.semantic_score,
                "bm25_score": r.bm25_score,
                "rerank_score": r.rerank_score
            })
            
        answer_text = ""
        # Step 2: Query LLM Generator
        if generator:
            with st.spinner("Generating answer using LLM..."):
                answer_text = generator.generate_answer(
                    query=query_input,
                    retrieved_chunks=retrieved_results,
                    temperature=llm_temp
                )
        else:
            answer_text = "*(LLM Answering skipped due to missing API Key. Showing retrieved chunks below.)*"
            
        # Display assistant answer
        st.markdown(f"<div class='chat-bubble-assistant'><strong>Assistant:</strong><br>{answer_text}</div>", unsafe_allow_html=True)
        
        # Show citation snippets directly
        if sources_to_store:
            with st.expander("Show Answer Sources & Match Details", expanded=True):
                for idx, src in enumerate(sources_to_store, 1):
                    st.markdown(
                        f"<span class='source-badge'>Source {idx}</span> "
                        f"Page {src['page']} of {src['source']} "
                        f"(Rank: {src['rank']} | Score: {src['score']:.4f})",
                        unsafe_allow_html=True
                    )
                    st.text_area(f"Snippet from Page {src['page']}", src["content"], height=100, key=f"src_box_new_{idx}")

        # Update chat history state
        query_id = len(st.session_state["chat_history"])
        st.session_state["chat_history"].append({
            "query_id": query_id,
            "role": "user",
            "content": query_input
        })
        st.session_state["chat_history"].append({
            "query_id": query_id,
            "role": "assistant",
            "content": answer_text,
            "sources": sources_to_store
        })
        
        # Rerun to preserve conversation layout
        st.rerun()

else:
    # Standard initial welcome info
    st.markdown("""
    <div class='glass-card' style='text-align: center; padding: 3rem 1.5rem;'>
        <img src='https://img.icons8.com/color/144/000000/opened-folder-key.png' width='80'/>
        <h3 style='margin-top: 1rem;'>System Ready for Ingestion</h3>
        <p style='color: #b2bec3;'>Upload your PDF or text book in the left sidebar control center. 
        Adjust parameters like chunk size and retrieval strategies, then process the book to begin querying!</p>
    </div>
    """, unsafe_allow_html=True)
