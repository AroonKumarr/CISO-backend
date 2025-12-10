# rag_backend.py
"""
AIgilityX CISO Advisor Backend:
- AI-powered cybersecurity advisory system for CISOs
- RAG pipeline with Qdrant vector database
- OpenAI-powered responses with CISO-specific prompts
- Integrated frontend serving
- Document upload and intelligent chunking
"""

import os
import re
import uuid
import json
import logging
from typing import List
from pathlib import Path

import requests
import pandas as pd
import nltk
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pypdf import PdfReader

# Download NLTK data quietly
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except:
    pass

logging.basicConfig(level=logging.INFO)

# ------------------- CONFIG -------------------
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(os.getcwd(), "uploads"))
CHUNKS_DIR = os.environ.get("CHUNKS_DIR", os.path.join(os.getcwd(), "chunks"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# Frontend directory - fixed path resolution
FRONTEND_DIR = os.path.join(os.path.dirname(os.getcwd()), "frontend")
if not os.path.exists(FRONTEND_DIR):
    FRONTEND_DIR = os.path.join(os.getcwd(), "..", "frontend")
if not os.path.exists(FRONTEND_DIR):
    # Try parent directory
    FRONTEND_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "frontend"))

QDRANT_URL = os.environ.get("QDRANT_URL", "...")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "384"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
QDRANT_DUPLICATE_SCORE_THRESHOLD = float(os.environ.get("QDRANT_DUP_SCORE", "0.90"))

OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

OPENAI_PROJECT_ID = "proj_mA8J7MQh8U7rDgM7PtEBZZAi"

# CISO Collection Configuration
CISO_CONFIG = {
    "name": "AIgilityX CISO Advisor",
    "collection": "ciso-embeddings",
    "system_prompt": """You are AIgilityX CISO Advisor, an AI-powered cybersecurity assistant for CISOs and senior security executives. 
Provide professional, concise, and actionable cybersecurity advice. Use your expertise in risk management, security architecture, incident response, compliance, and emerging threats. 
Always give clear, executive-level answers and reference frameworks when relevant. Keep responses brief and focused on business impact."""
}

# ------------------- INIT MODELS & CLIENTS -------------------
print("Loading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
print("Model loaded successfully!")

print("Connecting to Qdrant...")
qdrant_kwargs = {"url": QDRANT_URL}
if QDRANT_API_KEY:
    qdrant_kwargs["api_key"] = QDRANT_API_KEY
client = QdrantClient(**qdrant_kwargs)
print("Qdrant connection established!")

# Create CISO collection
try:
    if not client.collection_exists(CISO_CONFIG["collection"]):
        client.create_collection(
            collection_name=CISO_CONFIG["collection"],
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logging.info(f"Created Qdrant collection: {CISO_CONFIG['collection']}")
    else:
        logging.info(f"Collection already exists: {CISO_CONFIG['collection']}")
except Exception as e:
    logging.warning(f"Collection check/create error: {e}")

# ------------------- FASTAPI APP -------------------
app = FastAPI(title="AIgilityX CISO Advisor")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.vercel.app",
        "https://ciso-frontend-fteflchj3-aigilityxs-projects.vercel.app",  # Your specific Vercel URL
        "https://ciso-backend-production.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------- SERVE FRONTEND -------------------
@app.get("/")
def root():
    return {
        "service": "AIgilityX CISO Advisor API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AIgilityX CISO Advisor",
        "collection": CISO_CONFIG["collection"],
        "frontend_available": os.path.exists(FRONTEND_DIR),
        "frontend_path": FRONTEND_DIR
    }

# ------------------- UTIL FUNCTIONS -------------------
def clean_pdf(file_path: str) -> str:
    """Read PDF pages and extract text"""
    try:
        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n\n".join(pages)
    except Exception as e:
        logging.error(f"PDF reading error: {e}")
        return ""

def sentence_chunk(text: str, max_sentences: int = 5, overlap: int = 2) -> List[str]:
    """Chunk text by sentences with overlap"""
    try:
        sents = nltk.sent_tokenize(text)
    except:
        # Fallback to simple splitting
        sents = text.split('. ')
    
    chunks = []
    i = 0
    while i < len(sents):
        chunk = " ".join(sents[i : i + max_sentences]).strip()
        if len(chunk.split()) >= 5:
            chunks.append(chunk)
        i += max_sentences - overlap
    return chunks

def is_valid_chunk(chunk: str, min_words: int = 5, min_chars: int = 30) -> bool:
    """Validate chunk quality"""
    chunk = chunk.strip()
    if len(chunk.split()) < min_words or len(chunk) < min_chars:
        return False
    if not any(c.isalpha() for c in chunk):
        return False
    return True

def deduplicate_embeddings(embeddings, texts, threshold: float = 0.90):
    """Remove near-duplicate texts based on cosine similarity"""
    from sklearn.metrics.pairwise import cosine_similarity
    keep = [True] * len(texts)
    for i in range(len(embeddings)):
        if not keep[i]:
            continue
        if i + 1 < len(embeddings):
            sims = cosine_similarity([embeddings[i]], embeddings[i + 1 :])[0]
            for j, sim in enumerate(sims, start=i + 1):
                if sim > threshold:
                    keep[j] = False
    filtered_texts = [t for k, t in zip(keep, texts) if k]
    filtered_embs = [e for k, e in zip(keep, embeddings) if k]
    return filtered_texts, filtered_embs

def filter_against_qdrant(collection_name: str, texts: List[str], embeddings, score_threshold: float = QDRANT_DUPLICATE_SCORE_THRESHOLD):
    """Check against Qdrant to filter duplicates"""
    unique_texts = []
    unique_embeddings = []
    for txt, emb in zip(texts, embeddings):
        try:
            res = client.search(collection_name=collection_name, query_vector=emb.tolist(), limit=1, with_payload=False)
            if not res or res[0].score < score_threshold:
                unique_texts.append(txt)
                unique_embeddings.append(emb)
        except Exception as e:
            logging.warning(f"Qdrant search failed: {e}; keeping chunk")
            unique_texts.append(txt)
            unique_embeddings.append(emb)
    return unique_texts, unique_embeddings

# ------------------- LLM FUNCTION -------------------
def ask_openai_ciso(context: str, question: str) -> str:
    """Query OpenAI with CISO-specific prompt - ALWAYS FALLBACK"""
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
    
    system_prompt = CISO_CONFIG["system_prompt"]
    
    # If we have context from documents, include it
    if context and context.strip():
        user_message = f"""Based on the following context from our knowledge base and your cybersecurity expertise, please answer this question:

KNOWLEDGE BASE CONTEXT:
{context}

QUESTION:
{question}

Please provide a comprehensive, executive-level response that combines the specific information from our knowledge base with your broader CISO expertise."""
    else:
        # No context - pure CISO advisory
        user_message = f"""As an experienced CISO advisor, please answer this cybersecurity question with your expert knowledge:

QUESTION:
{question}

Please provide a comprehensive, executive-level response based on industry best practices, frameworks, and your cybersecurity expertise."""
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENAI_PROJECT_ID:
        headers["OpenAI-Project"] = OPENAI_PROJECT_ID
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }
    
    try:
        logging.info("Sending request to OpenAI...")
        resp = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        logging.info("Received response from OpenAI")
        return answer
    except Exception as e:
        logging.error(f"OpenAI call failed: {e}")
        return f"I apologize, but I'm having trouble connecting to my AI backend. Error: {str(e)}"

# ------------------- API ENDPOINTS -------------------
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process cybersecurity documents for CISO knowledge base"""
    collection_name = CISO_CONFIG["collection"]
    
    logging.info(f"Received file upload: {file.filename}")
    
    # Ensure collection exists
    try:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    except Exception as e:
        logging.warning(f"Collection init error: {e}")
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    ext = file.filename.lower().split(".")[-1]
    logging.info(f"Processing file type: {ext}")
    
    # Extract text based on file type
    text = ""
    if ext == "pdf":
        text = clean_pdf(file_path)
    elif ext == "txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as r:
            text = r.read()
    elif ext == "csv":
        try:
            df = pd.read_csv(file_path, dtype=str, encoding="utf-8", on_bad_lines="skip")
            text = "\n".join(df.fillna("").astype(str).values.flatten())
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": "CSV read failed"})
    elif ext in ("xls", "xlsx"):
        try:
            df = pd.read_excel(file_path, dtype=str)
            text = "\n".join(df.fillna("").astype(str).values.flatten())
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": "Excel read failed"})
    elif ext == "docx":
        try:
            import docx
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        except:
            return JSONResponse(status_code=400, content={"error": "DOCX read failed - install python-docx"})
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type. Use PDF, TXT, CSV, XLSX, or DOCX"})
    
    if not text or len(text.strip()) < 50:
        return {"status": "ok", "chunks": 0, "message": "No meaningful text extracted."}
    
    # Chunk and validate
    raw_chunks = sentence_chunk(text, max_sentences=5, overlap=2)
    valid_chunks = [c for c in raw_chunks if is_valid_chunk(c)]
    
    if not valid_chunks:
        return {"status": "ok", "chunks": 0, "message": "No valid chunks after filtering."}
    
    # Embed
    try:
        embeddings = embedder.encode(valid_chunks, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Embedding failed: {str(e)}"})
    
    # Deduplicate
    dedup_chunks, dedup_embeddings = deduplicate_embeddings(embeddings, valid_chunks)
    unique_chunks, unique_embeddings = filter_against_qdrant(collection_name, dedup_chunks, dedup_embeddings)
    
    if not unique_chunks:
        return {"status": "ok", "chunks": 0, "message": "No new content (all duplicates)."}
    
    # Prepare points
    ids = [str(uuid.uuid4()) for _ in unique_chunks]
    payloads = [{"source_file": file.filename, "text": t} for t in unique_chunks]
    points = [PointStruct(id=i, vector=v.tolist(), payload=p) for i, v, p in zip(ids, unique_embeddings, payloads)]
    
    # Batched upsert
    uploaded = 0
    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start : start + BATCH_SIZE]
        try:
            client.upsert(collection_name=collection_name, points=batch)
            uploaded += len(batch)
        except Exception as e:
            logging.error(f"Qdrant upsert error: {e}")
    
    logging.info(f"Uploaded {uploaded} chunks to CISO knowledge base")
    return {"status": "ok", "chunks": uploaded, "filename": file.filename}

@app.post("/query/")
async def query(
    question: str = Form(...),
    top_k: int = Form(5),
    score_threshold: float = Form(0.3)
):
    """Query the CISO advisor - ALWAYS uses OpenAI with or without context"""
    collection_name = CISO_CONFIG["collection"]
    
    logging.info(f"Received query: {question}")
    
    # Try to get context from knowledge base
    context = ""
    citations = []
    chunks_info = []
    
    try:
        q_vec = embedder.encode([question], convert_to_numpy=True)[0].tolist()
        
        results = client.search(
            collection_name=collection_name,
            query_vector=q_vec,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        filtered = [r for r in results if getattr(r, "score", 0) >= score_threshold]
        
        if filtered:
            context = "\n\n".join([r.payload.get("text", "") for r in filtered])
            citations = list(set([r.payload.get("source_file", "") for r in filtered if r.payload.get("source_file")]))
            chunks_info = [{
                "score": r.score,
                "file": r.payload.get("source_file", ""),
                "chunk": r.payload.get("text", "")[:250]
            } for r in filtered]
            
            logging.info(f"Found {len(filtered)} relevant chunks for query")
        else:
            logging.info("No relevant chunks found, using pure CISO knowledge")
            
    except Exception as e:
        logging.warning(f"Search failed: {e}, falling back to pure CISO knowledge")
    
    # ALWAYS call OpenAI (fallback behavior)
    answer = ask_openai_ciso(context, question)
    
    logging.info("Sending response to frontend")
    
    return {
        "answer": answer,
        "citations": citations,
        "chunks": chunks_info,
        "has_context": bool(context),
        "advisor": CISO_CONFIG["name"]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Print startup information
    print("\n" + "="*60)
    print("🛡️  AIgilityX CISO Advisor - Starting...")
    print("="*60)
    print(f"📁 Frontend Directory: {FRONTEND_DIR}")
    print(f"📊 Vector Database: Qdrant")
    print(f"🤖 AI Model: OpenAI GPT-4")
    print(f"🔒 Collection: {CISO_CONFIG['collection']}")
    print("="*60)
    print("🌐 Access the application at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)