from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
import os
import pdfplumber
import json
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
import psutil
from datetime import datetime

load_dotenv()

app = FastAPI()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configs
UPLOAD_DIR = "uploads"
CHUNKS_DIR = "chunks"
VECTOR_DIR = "vectorstores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

GLOBAL_INDEX_PATH = os.path.join(VECTOR_DIR, "global.index")
METADATA_PATH = os.path.join(VECTOR_DIR, "metadata.json")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL_NAME)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load/create FAISS index
if os.path.exists(GLOBAL_INDEX_PATH):
    index = faiss.read_index(GLOBAL_INDEX_PATH)
else:
    index = faiss.IndexFlatL2(384)

# Load/create metadata
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_store = json.load(f)
else:
    metadata_store = {}

# --- UTILS ---

def save_faiss():
    faiss.write_index(index, GLOBAL_INDEX_PATH)

def save_metadata():
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, ensure_ascii=False, indent=2)

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def batch_encode(texts: List[str], batch_size: int = 32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch)
        embeddings.append(emb)
    return np.vstack(embeddings)

def truncate_texts(texts: List[str], max_tokens=2000):
    total = 0
    output = []
    for t in texts:
        tokens = len(t.split())
        if total + tokens > max_tokens:
            break
        output.append(t)
        total += tokens
    return output

# --- API ---

@app.get("/")
async def root():
    return {"message": "StudyPath Backend v2.6 is alive! 🧠"}

@app.get("/languages/")
async def list_languages():
    return {
        "languages": {
            "en": "English",
            "es": "Español",
            "zh-cn": "简体中文 (Simplified Chinese)"
        }
    }

@app.post("/upload/")
async def upload_and_ingest(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as f:
        f.write(await file.read())

    chunks = []
    with pdfplumber.open(file_location) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chunks.extend(chunk_text(text))

    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found")

    embeddings = batch_encode(chunks)
    current_count = len(metadata_store)

    index.add(embeddings)

    uploaded_at = datetime.now().isoformat()

    for i, chunk_text_data in enumerate(chunks):
        metadata_store[str(current_count + i)] = {
            "source": file.filename,
            "content": chunk_text_data,
            "uploaded_at": uploaded_at,
            "deleted": False
        }

    save_faiss()
    save_metadata()

    return JSONResponse(content={"message": f"Uploaded, processed, and embedded {len(chunks)} chunks from {file.filename}"})

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    language: str = "en"

@app.post("/ask/")
async def ask_question(req: QueryRequest):
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="Vectorstore is empty")

    valid_languages = ["en", "es", "zh-cn"]
    if req.language not in valid_languages:
        raise HTTPException(status_code=400, detail=f"Invalid language '{req.language}'. Must be one of {valid_languages}")

    question_embedding = model.encode([req.question])
    D, I = index.search(np.array(question_embedding), req.top_k)

    retrieved = []
    for idx in I[0]:
        meta = metadata_store.get(str(idx))
        if meta and not meta.get("deleted", False):
            retrieved.append(meta)

    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    context_chunks = [r["content"] for r in retrieved]
    context = "\n\n".join(truncate_texts(context_chunks))

    prompt = f"Use the following context to answer the user's question:\n\n{context}\n\nQuestion: {req.question}\nAnswer:"

    if req.language == "es":
        prompt += "\n(Respond in Spanish)"
    elif req.language == "zh-cn":
        prompt += "\n(Respond in Simplified Chinese)"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful college admissions advisor."},
            {"role": "user", "content": prompt},
        ]
    )

    final_answer = response['choices'][0]['message']['content']
    sources = list(set([r["source"] for r in retrieved]))

    return JSONResponse(content={"answer": final_answer, "sources": sources})

@app.get("/list_documents/")
async def list_documents():
    docs = {}
    for meta in metadata_store.values():
        if not meta.get("deleted", False):
            source = meta.get("source", "unknown")
            uploaded_at = meta.get("uploaded_at", "unknown")
            docs[source] = uploaded_at
    return {"documents": docs}

@app.delete("/delete_document/")
async def delete_document(filename: str, hard: bool = False):
    found = False
    keys_to_delete = []

    for key, meta in metadata_store.items():
        if meta.get("source") == filename and not meta.get("deleted", False):
            if hard:
                keys_to_delete.append(key)
            else:
                meta["deleted"] = True
            found = True

    if not found:
        raise HTTPException(status_code=404, detail="Document not found")

    for key in keys_to_delete:
        del metadata_store[key]

    save_metadata()
    return {"message": f"{'Hard deleted' if hard else 'Soft deleted'} {filename} successfully"}

@app.get("/stats/")
async def stats():
    total_docs = len(set([meta["source"] for meta in metadata_store.values() if not meta.get("deleted", False)]))
    total_chunks = len([1 for meta in metadata_store.values() if not meta.get("deleted", False)])
    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "vectorstore_size": index.ntotal
    }

@app.get("/health/")
async def health_check():
    ram_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=0.5)
    return {
        "status": "🫀 Alive",
        "RAM_Usage_Percent": ram_usage,
        "CPU_Usage_Percent": cpu_usage
    }
