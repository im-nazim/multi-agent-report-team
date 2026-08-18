"""RAG (Retrieval-Augmented Generation) layer.

Handles turning uploaded PDF/DOCX/TXT files into a searchable knowledge base:
extract text -> chunk -> embed -> store as vectors -> cosine-similarity search.

Uses a lightweight numpy-based vector store (no Chroma/FAISS dependency) so
it's easy to install on Windows and easy to explain end-to-end in an interview.
The knowledge base itself lives in Streamlit's session_state so it persists
across reruns without needing a real database.
"""

import hashlib
from typing import List, Dict

import numpy as np
import streamlit as st
from pypdf import PdfReader
import docx as docx_lib
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "models/gemini-embedding-001"


def hash_uploaded_files(files) -> str:
    """Fingerprint the current set of uploaded files so we only re-index when they change."""
    if not files:
        return ""
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(str(f.size).encode())
    return h.hexdigest()


def extract_pages(uploaded_file) -> List[Dict]:
    """Extracts text from a PDF, DOCX, or TXT file, returning a list of
    {text, source, page} dicts. PDFs keep real page numbers so citations
    can point to an exact page; DOCX/TXT are treated as a single 'page'."""
    name = uploaded_file.name
    pages = []

    if name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "source": name, "page": i})

    elif name.lower().endswith(".docx"):
        doc = docx_lib.Document(uploaded_file)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if full_text.strip():
            pages.append({"text": full_text, "source": name, "page": None})

    elif name.lower().endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        if text.strip():
            pages.append({"text": text, "source": name, "page": None})

    return pages


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """Splits each page's text into overlapping chunks, keeping source/page metadata attached."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append({"text": piece, "source": page["source"], "page": page["page"]})
    return chunks


def build_knowledge_base(uploaded_files, api_key: str) -> None:
    """Extracts, chunks, and embeds all uploaded files. Stores chunks + vectors in session_state."""
    all_pages = []
    for f in uploaded_files:
        all_pages.extend(extract_pages(f))

    chunks = chunk_pages(all_pages)
    if not chunks:
        st.session_state.kb_chunks = []
        st.session_state.kb_vectors = None
        return

    embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    vectors = embeddings_model.embed_documents([c["text"] for c in chunks])

    st.session_state.kb_chunks = chunks
    st.session_state.kb_vectors = np.array(vectors)


def doc_search(query: str, api_key: str, max_results: int = 4) -> List[Dict[str, str]]:
    """Embeds the query and returns the top-k most similar chunks from the
    session's knowledge base, formatted like web_search results so the rest
    of the pipeline (citations, structured output) doesn't need to know the difference."""
    chunks = st.session_state.get("kb_chunks", [])
    vectors = st.session_state.get("kb_vectors")
    if not chunks or vectors is None or len(chunks) == 0:
        return []

    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
        query_vec = np.array(embeddings_model.embed_query(query))
    except Exception as e:
        return [{"title": "Document search error", "body": f"[Embedding failed: {e}]", "href": ""}]

    # Cosine similarity against every stored chunk vector
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
    norms[norms == 0] = 1e-8
    scores = vectors.dot(query_vec) / norms
    top_idx = np.argsort(scores)[::-1][:max_results]

    results = []
    for idx in top_idx:
        c = chunks[idx]
        label = f"{c['source']} (page {c['page']})" if c["page"] else c["source"]
        results.append({
            "title": label,
            "body": c["text"][:500],
            # unique fake "URL" per chunk so it dedupes correctly alongside web sources
            "href": f"doc://{c['source']}#{idx}",
        })
    return results