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
import uuid
import pytesseract
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

global_index = faiss.read_index(GLOBAL_INDEX_PATH) if os.path.exists(GLOBAL_INDEX_PATH) else faiss.IndexFlatL2(384)

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata_store = json.load(f)
else:
    metadata_store = {}

def save_faiss():
    faiss.write_index(global_index, GLOBAL_INDEX_PATH)

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

@app.get("/")
async def root():
    return {"message": "StudyPath Backend v3.0 is alive! 🧠"}

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
    print("🔥 /upload/ endpoint triggered")

    if not file:
        print("❌ No file received!")
        raise HTTPException(status_code=400, detail="No file uploaded.")

    print(f"📄 Received file: {file.filename}")

    file_id = str(uuid.uuid4())
    file_location = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    try:
        contents = await file.read()
        print(f"📦 File size received: {len(contents)} bytes")
        with open(file_location, "wb") as f_out:
            f_out.write(contents)
    except Exception as e:
        print(f"❌ Failed writing file: {e}")
        raise HTTPException(status_code=500, detail="File save failed")

    chunks = []
    try:
        with pdfplumber.open(file_location) as pdf:
            print(f"📚 PDF has {len(pdf.pages)} page(s)")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    print(f"🔍 Page {i + 1} has no text — running OCR")
                    pil_stream = io.BytesIO()
                    page.to_image(resolution=300).save(pil_stream, format="PNG")
                    pil_stream.seek(0)
                    pil_image = Image.open(pil_stream)
                    text = pytesseract.image_to_string(pil_image)
                else:
                    print(f"📃 Page {i + 1}: {len(text)} characters (text layer)")
                if text:
                    chunks.extend(chunk_text(text))
                    for c in chunk_text(text):
                        print("🧾", c[:300])


    except Exception as e:
        print(f"💥 PDF processing error: {e}")
        raise HTTPException(status_code=400, detail="Error reading PDF")

    if not chunks:
        print("⚠️ No chunks extracted from the PDF")
        raise HTTPException(status_code=400, detail="No readable text found")

    print(f"✅ Extracted {len(chunks)} chunks")

    embeddings = batch_encode(chunks)
    doc_index = faiss.IndexFlatL2(384)
    doc_index.add(embeddings)
    faiss.write_index(doc_index, os.path.join(VECTOR_DIR, f"{file_id}.index"))

    doc_metadata = {
        str(i): {
            "content": chunk,
            "source": file.filename,
            "uploaded_at": datetime.now().isoformat()
        } for i, chunk in enumerate(chunks)
    }

    with open(os.path.join(VECTOR_DIR, f"{file_id}.json"), "w", encoding="utf-8") as f:
        json.dump(doc_metadata, f, ensure_ascii=False, indent=2)

    global_index.add(embeddings)
    for i, chunk in enumerate(chunks):
        metadata_store[str(len(metadata_store) + i)] = {
            "source": file.filename,
            "content": chunk,
            "uploaded_at": datetime.now().isoformat(),
            "deleted": False
        }

    save_faiss()
    save_metadata()

    print("✅ Upload and processing complete")
    return {"message": "Upload complete", "document_id": file_id}

class QueryRequest(BaseModel):
    question: str
    document_id: str = None
    top_k: int = 3
    language: str = "en"

@app.post("/ask/")
async def ask_question(req: QueryRequest):
    chunks = []

    if req.document_id:
        index_path = os.path.join(VECTOR_DIR, f"{req.document_id}.index")
        meta_path = os.path.join(VECTOR_DIR, f"{req.document_id}.json")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            user_index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                user_meta = json.load(f)
            D_user, I_user = user_index.search(model.encode([req.question]), req.top_k)
            chunks += [user_meta[str(i)]["content"] for i in I_user[0] if str(i) in user_meta]

    global_meta = {k: v for k, v in metadata_store.items() if not v.get("deleted", False)}
    D_global, I_global = global_index.search(model.encode([req.question]), req.top_k)
    chunks += [global_meta[str(i)]["content"] for i in I_global[0] if str(i) in global_meta]

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant context found")

    context = "\n\n".join(truncate_texts(chunks))
    prompt = f"Use the following context (which may include both a user's personal document and general knowledge) to answer the user's question:\n\n{context}\n\nQuestion: {req.question}\nAnswer:"

    if req.language == "es":
        prompt += "\n(Responde en español)"
    elif req.language == "zh-cn":
        prompt += "\n(用简体中文回答)"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful college admissions advisor."},
            {"role": "user", "content": prompt},
        ]
    )

    return {"answer": response['choices'][0]['message']['content']}

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
        "vectorstore_size": global_index.ntotal
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
