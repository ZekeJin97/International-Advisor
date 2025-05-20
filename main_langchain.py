from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os, uuid, io, json, shutil
import pdfplumber
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from datetime import datetime
import psutil

# Load .env and set tesseract path
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ────────────────────────────── APP SETUP ──────────────────────────────

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectorstores"
DOC_MAP_PATH = os.path.join(VECTOR_DIR, "doc_map.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
memory_store = {}

# ────────────────────────────── METADATA ──────────────────────────────

def load_doc_map():
    if os.path.exists(DOC_MAP_PATH):
        with open(DOC_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_doc_map(doc_map):
    with open(DOC_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_map, f, ensure_ascii=False, indent=2)

# ────────────────────────────── OCR ──────────────────────────────

def extract_with_ocr(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            pil_stream = io.BytesIO()
            page.to_image(resolution=300).save(pil_stream, format="PNG")
            pil_stream.seek(0)
            pil_image = Image.open(pil_stream).convert("L")
            pil_image = pil_image.point(lambda x: 0 if x < 160 else 255, '1')
            text += pytesseract.image_to_string(pil_image, config="--psm 6") + "\n"
    return text

# ────────────────────────────── MODELS ──────────────────────────────

class AskRequest(BaseModel):
    question: str
    doc_id: str
    language: str = "en"

# ────────────────────────────── ROUTES ──────────────────────────────

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    filename = file.filename
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Try native PDF extraction
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Fallback to OCR
    if not pages or all(not p.page_content.strip() for p in pages):
        print("🔍 Falling back to OCR")
        text = extract_with_ocr(file_path)
        pages = [Document(page_content=text)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    # Save vectorstore
    vector_path = os.path.join(VECTOR_DIR, doc_id)
    os.makedirs(vector_path, exist_ok=True)
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(vector_path)

    # Save doc map
    doc_map = load_doc_map()
    doc_map[doc_id] = {
        "filename": filename,
        "uploaded_at": datetime.now().isoformat(),
        "deleted": False
    }
    save_doc_map(doc_map)

    return {"message": f"Uploaded and embedded {len(docs)} chunks", "doc_id": doc_id}

@app.post("/ask/")
async def ask_question(req: AskRequest):
    doc_map = load_doc_map()
    doc_id = req.doc_id

    if doc_id not in doc_map or doc_map[doc_id].get("deleted", False):
        raise HTTPException(status_code=404, detail="Document not found")

    vector_path = os.path.join(VECTOR_DIR, doc_id)
    if not os.path.exists(vector_path):
        raise HTTPException(status_code=404, detail="Vectorstore not found")

    db = FAISS.load_local(vector_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()

    if doc_id not in memory_store:
        memory_store[doc_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    prompt = req.question
    if req.language == "es":
        prompt = f"(Responde en español)\n{prompt}"
    elif req.language == "zh-cn":
        prompt = f"(用简体中文回答)\n{prompt}"

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        retriever=retriever,
        memory=memory_store[doc_id],
        return_source_documents=True,
        output_key="answer"
    )

    result = chain.invoke({"question": prompt})
    answer = result["answer"]
    sources = list(set(doc.metadata.get("source", "unknown") for doc in result["source_documents"]))

    return {"answer": answer, "sources": sources}

@app.get("/list_documents/")
async def list_documents():
    doc_map = load_doc_map()
    docs = []
    for doc_id, meta in doc_map.items():
        if not meta.get("deleted", False):
            docs.append({"doc_id": doc_id, "filename": meta["filename"], "uploaded_at": meta["uploaded_at"]})
    return {"documents": docs}

@app.delete("/delete_document/")
async def delete_document(doc_id: str):
    doc_map = load_doc_map()
    if doc_id not in doc_map:
        raise HTTPException(status_code=404, detail="Document not found")

    vector_path = os.path.join(VECTOR_DIR, doc_id)
    try:
        if os.path.exists(vector_path):
            shutil.rmtree(vector_path)
        doc_map[doc_id]["deleted"] = True
        save_doc_map(doc_map)
        return {"message": f"Soft-deleted {doc_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")

@app.get("/health/")
async def health():
    ram_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=0.5)
    return {
        "status": "🧠 LangChain RAG backend alive",
        "RAM_Usage_Percent": ram_usage,
        "CPU_Usage_Percent": cpu_usage
    }
