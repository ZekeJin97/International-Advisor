from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import pdfplumber
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload dir
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Chunking params
CHUNK_SIZE = 500
OVERLAP = 50

# Embedding model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL_NAME)

# FAISS path
FAISS_INDEX_PATH = "vectorstore.index"

# Load FAISS index if exists
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    index = None

# ---------------------- ROUTES ----------------------

@app.get("/")
async def root():
    return {"message": "StudyPath Backend is alive 🧠"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())
    return JSONResponse(content={"filename": file.filename, "message": "File uploaded successfully!"})

@app.post("/process/")
async def process_uploaded_file(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    with pdfplumber.open(file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    if not full_text.strip():
        return JSONResponse(content={"error": "No readable text found in PDF"}, status_code=400)

    chunks = chunk_text(full_text)

    chunks_dir = os.path.join("chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    base_filename = os.path.splitext(filename)[0]
    chunk_file = os.path.join(chunks_dir, f"{base_filename}_chunks.txt")

    with open(chunk_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c + "\n---\n")

    return JSONResponse(content={"message": f"File processed into {len(chunks)} chunks!"})

@app.post("/embed/")
async def embed_chunks(filename: str):
    chunk_file = os.path.join("chunks", f"{os.path.splitext(filename)[0]}_chunks.txt")

    if not os.path.exists(chunk_file):
        return JSONResponse(content={"error": "Chunks file not found"}, status_code=404)

    with open(chunk_file, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = [c.strip() for c in content.split("---") if c.strip()]

    if not chunks:
        return JSONResponse(content={"error": "No chunks to embed"}, status_code=400)

    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]

    global index
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, FAISS_INDEX_PATH)

    return JSONResponse(content={"message": f"Embedded {len(chunks)} chunks into FAISS index!"})

# ---------------------- UTILS ----------------------

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

# ---------------------- RAG / ASK ----------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask/")
async def ask_question(req: QueryRequest):
    if index is None:
        return JSONResponse(content={"error": "Vectorstore is empty!"}, status_code=400)

    question_embedding = model.encode([req.question])
    D, I = index.search(np.array(question_embedding), req.top_k)

    base_filename = "f1-travel-visa-guide"  # HARDCODED for now
    chunk_file = os.path.join("chunks", f"{base_filename}_chunks.txt")

    with open(chunk_file, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = [c.strip() for c in content.split("---") if c.strip()]

    retrieved_chunks = []
    for idx in I[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return JSONResponse(content={"chunks": retrieved_chunks})
