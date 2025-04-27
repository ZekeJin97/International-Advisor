from fastapi import FastAPI, UploadFile, File
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
from fastapi import HTTPException

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
FAISS_INDEX_DIR = "vectorstores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Models
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL_NAME)
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- UTILS ---

def chunk_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def load_faiss_index(filename: str):
    path = os.path.join(FAISS_INDEX_DIR, f"{filename}.index")
    return faiss.read_index(path) if os.path.exists(path) else None

def save_faiss_index(index, filename: str):
    path = os.path.join(FAISS_INDEX_DIR, f"{filename}.index")
    faiss.write_index(index, path)

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
        tokens = len(t.split())  # muy basic pero suficiente
        if total + tokens > max_tokens:
            break
        output.append(t)
        total += tokens
    return output

# --- API ---

@app.get("/")
async def root():
    return {"message": "StudyPath Backend is alive 🧠"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as f:
        f.write(await file.read())
    return JSONResponse(content={"filename": file.filename, "message": "File uploaded successfully"})

@app.post("/process/")
async def process_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chunks.extend(chunk_text(text))

    if not chunks:
        return JSONResponse(content={"error": "No readable text found"}, status_code=400)

    base_filename = os.path.splitext(filename)[0]
    chunk_file_path = os.path.join(CHUNKS_DIR, f"{base_filename}_chunks.jsonl")
    with open(chunk_file_path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            metadata = {"id": i, "content": c, "source": filename, "page_hint": i}
            f.write(json.dumps(metadata) + "\n")

    return JSONResponse(content={"message": f"Processed {len(chunks)} chunks"})

@app.post("/embed/")
async def embed_chunks(filename: str):
    base_filename = os.path.splitext(filename)[0]
    chunk_file_path = os.path.join(CHUNKS_DIR, f"{base_filename}_chunks.jsonl")
    if not os.path.exists(chunk_file_path):
        return JSONResponse(content={"error": "Chunks file not found"}, status_code=404)

    chunks = []
    with open(chunk_file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    texts = [c["content"] for c in chunks]
    embeddings = batch_encode(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    save_faiss_index(index, base_filename)

    return JSONResponse(content={"message": f"Embedded {len(texts)} chunks into FAISS index"})

class QueryRequest(BaseModel):
    question: str
    filename: str
    top_k: int = 3

@app.post("/ask/")
async def ask_question(req: QueryRequest):
    base_filename = os.path.splitext(req.filename)[0]
    index = load_faiss_index(base_filename)
    if index is None:
        return JSONResponse(content={"error": "Vectorstore not found"}, status_code=404)

    chunk_file_path = os.path.join(CHUNKS_DIR, f"{base_filename}_chunks.jsonl")
    if not os.path.exists(chunk_file_path):
        return JSONResponse(content={"error": "Chunks file not found"}, status_code=404)

    # Search
    question_embedding = model.encode([req.question])
    D, I = index.search(np.array(question_embedding), req.top_k)

    chunks = []
    with open(chunk_file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    retrieved = []
    for idx in I[0]:
        if idx < len(chunks):
            retrieved.append(chunks[idx])

    context_chunks = [chunk['content'] for chunk in retrieved]
    context = "\n\n".join(truncate_texts(context_chunks))

    prompt = f"Use the following context to answer the user's question:\n\n{context}\n\nQuestion: {req.question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful college admissions advisor."},
            {"role": "user", "content": prompt},
        ]
    )

    final_answer = response['choices'][0]['message']['content']
    return JSONResponse(content={"answer": final_answer, "chunks": retrieved})

import psutil

@app.get("/health/")
async def health_check():
    ram_usage = psutil.virtual_memory().percent  # % of RAM used
    cpu_usage = psutil.cpu_percent(interval=0.5)  # % of CPU used (over 0.5s)
    return {
        "status": "🫀 Alive",
        "RAM_Usage_Percent": ram_usage,
        "CPU_Usage_Percent": cpu_usage
    }

@app.get("/list_uploads/")
async def list_uploaded_files():
    try:
        files = os.listdir(UPLOAD_DIR)
        pdfs = [f for f in files if f.lower().endswith(".pdf")]
        return {"uploaded_pdfs": pdfs}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.delete("/delete_upload/")
async def delete_uploaded_file(filename: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        os.remove(file_path)
        return {"message": f"Deleted {filename} successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{filename}")
async def file_status(filename: str):
    try:
        base_filename = os.path.splitext(filename)[0]

        upload_path = os.path.join(UPLOAD_DIR, filename)
        chunk_path = os.path.join(CHUNKS_DIR, f"{base_filename}_chunks.jsonl")
        index_path = os.path.join(FAISS_INDEX_DIR, f"{base_filename}.index")

        status = {
            "uploaded": os.path.exists(upload_path),
            "processed": os.path.exists(chunk_path),
            "embedded": os.path.exists(index_path)
        }

        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
