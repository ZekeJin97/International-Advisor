from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os, uuid, io, json, shutil, asyncio, time, re
import pdfplumber
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from datetime import datetime
import psutil
import logging
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env and set tesseract path
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ═══════════════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="StudyPath AI Advisor - Final",
    description="RAG system with OpenAI embeddings and proper chunking",
    version="4.0.0"
)

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

# USE OPENAI EMBEDDINGS - Much better for document understanding
embedding_model = OpenAIEmbeddings()


# ═══════════════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════════════

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    upload_time: datetime
    chunk_count: int
    processing_time: Optional[float] = None
    file_size: Optional[int] = None


class QueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    top_k: int = 3
    language: str = "en"

    @validator('question')
    def question_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: Optional[float] = None
    response_time: float
    chunk_count: int


# ═══════════════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════════════

class StudyPathException(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code


@app.exception_handler(StudyPathException)
async def studypath_exception_handler(request, exc: StudyPathException):
    logger.error(f"StudyPath error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "timestamp": datetime.now().isoformat()}
    )


# ═══════════════════════════════════════════════════════════════════════════════════════
# METADATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def load_doc_map():
    if os.path.exists(DOC_MAP_PATH):
        with open(DOC_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_doc_map(doc_map):
    with open(DOC_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(doc_map, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════════════
# OCR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def extract_with_ocr(file_path):
    """Enhanced OCR with better preprocessing"""
    logger.info(f"Starting OCR extraction for: {file_path}")
    text = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # Convert page to image
                    pil_stream = io.BytesIO()
                    page.to_image(resolution=300).save(pil_stream, format="PNG")
                    pil_stream.seek(0)

                    # Enhanced image preprocessing
                    pil_image = Image.open(pil_stream)
                    pil_image = pil_image.convert("L")
                    pil_image = pil_image.point(lambda x: 0 if x < 160 else 255, '1')

                    # OCR with optimized config
                    page_text = pytesseract.image_to_string(
                        pil_image,
                        config="--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}\"'-+="
                    )

                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        logger.info(f"OCR extracted {len(page_text)} characters from page {page_num + 1}")

                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num + 1}: {e}")
                    continue

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        raise StudyPathException(f"OCR processing failed: {str(e)}", 500)

    return text


# ═══════════════════════════════════════════════════════════════════════════════════════
# SIMPLE, PROPER CHUNKING - LARGER CHUNKS TO PRESERVE CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_proper_chunks(pages):
    """
    Simple, proper chunking with larger sizes to preserve context
    """
    # Combine all pages
    full_text = "\n\n".join([page.page_content for page in pages])

    logger.info(f"Full document length: {len(full_text)} characters")

    # Use recursive text splitter with larger chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # LARGER chunks
        chunk_overlap=200,  # Good overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split the text
    texts = text_splitter.split_text(full_text)

    # Create documents
    documents = []
    for i, text in enumerate(texts):
        documents.append(Document(
            page_content=text,
            metadata={
                "source": pages[0].metadata.get("source", "Document"),
                "chunk_index": i,
                "chunk_size": len(text)
            }
        ))

    logger.info(
        f"Created {len(documents)} chunks with average size {sum(len(doc.page_content) for doc in documents) // len(documents)} characters")

    # Log first few chunks for debugging
    for i, doc in enumerate(documents[:3]):
        logger.info(f"Chunk {i + 1} preview: {doc.page_content[:200]}...")

    return documents


# ═══════════════════════════════════════════════════════════════════════════════════════
# SIMPLE RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════════════

def get_relevant_context(db, question, top_k):
    """
    Simple retrieval - let OpenAI embeddings do the work
    """

    # With better embeddings and larger chunks, simple similarity search should work
    docs = db.similarity_search(question, k=top_k * 2)

    logger.info(f"Retrieved {len(docs)} chunks for question: {question}")
    for i, doc in enumerate(docs):
        logger.info(f"Retrieved chunk {i + 1}: {doc.page_content[:150]}...")

    return docs


# ═══════════════════════════════════════════════════════════════════════════════════════
# SIMPLE PROMPTING
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_simple_prompt(question, relevant_docs, language="en"):
    """
    Improved prompt that combines document info with general knowledge
    """

    # Build context
    context = "\n\n".join([
        f"Section {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(relevant_docs)
    ])

    # Enhanced prompt that allows external knowledge
    prompt = f"""You are an expert international student advisor. Answer the question using BOTH the document information provided AND your general knowledge about F-1 visa and international student regulations.

Question: {question}

Document sections:
{context}

Instructions:
1. First, extract relevant information from the document sections above
2. Then, apply your knowledge of F-1 visa rules and international student regulations
3. Combine both sources to provide a complete, helpful answer
4. If the document doesn't contain specific information but you know the general rules, include that knowledge
5. Always clarify what information comes from the document vs. general F-1 visa knowledge

Provide a comprehensive answer that helps the student understand both what their specific document says and the general regulations that apply."""

    if language == "es":
        prompt += "\n\nResponde en español."
    elif language == "zh-cn":
        prompt += "\n\n用简体中文回答。"

    return prompt
# ═══════════════════════════════════════════════════════════════════════════════════════
# BACKGROUND PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════════════

async def process_document_background(doc_id: str, file_path: str, filename: str):
    """Clean background processing"""
    doc_map = load_doc_map()

    try:
        logger.info(f"Starting background processing for {doc_id}")

        # Update status
        doc_map[doc_id]["status"] = "processing"
        doc_map[doc_id]["progress"] = 0
        save_doc_map(doc_map)

        # Try native PDF extraction
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            logger.info(f"Extracted {len(pages)} pages with native extraction")
            doc_map[doc_id]["progress"] = 30
            save_doc_map(doc_map)
        except Exception as e:
            logger.warning(f"Native PDF extraction failed: {e}")
            pages = []

        # OCR fallback
        if not pages or all(not p.page_content.strip() for p in pages):
            logger.info(f"Using OCR for {filename}")
            doc_map[doc_id]["ocr_used"] = True
            text = extract_with_ocr(file_path)
            pages = [Document(page_content=text, metadata={"source": filename})]
            doc_map[doc_id]["progress"] = 60
            save_doc_map(doc_map)

        # Create proper chunks
        docs = create_proper_chunks(pages)

        doc_map[doc_id]["progress"] = 80
        save_doc_map(doc_map)

        # Create vectorstore with OpenAI embeddings
        vector_path = os.path.join(VECTOR_DIR, doc_id)
        os.makedirs(vector_path, exist_ok=True)
        db = FAISS.from_documents(docs, embedding_model)
        db.save_local(vector_path)

        # Update final status
        doc_map[doc_id]["status"] = "completed"
        doc_map[doc_id]["chunk_count"] = len(docs)
        doc_map[doc_id]["progress"] = 100
        doc_map[doc_id]["completed_at"] = datetime.now().isoformat()
        save_doc_map(doc_map)

        logger.info(f"Completed processing for {doc_id}: {len(docs)} chunks created")

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        doc_map[doc_id]["status"] = "error"
        doc_map[doc_id]["error_message"] = str(e)
        save_doc_map(doc_map)


# ═══════════════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "message": "StudyPath AI Advisor v4.0 - OpenAI Embeddings + Proper Chunking",
        "status": "operational",
        "features": ["OpenAI Embeddings", "Larger Chunks", "Simple Retrieval", "Clean Architecture"]
    }


@app.post("/upload/", response_model=DocumentResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_time = time.time()

    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise StudyPathException("Only PDF files are supported", 400)

        # Read file content to get size
        contents = await file.read()
        file_size = len(contents)

        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise StudyPathException("File size must be less than 10MB", 400)

        # Generate document ID and save file
        doc_id = str(uuid.uuid4())
        filename = file.filename
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")

        with open(file_path, "wb") as f:
            f.write(contents)

        # Save initial document metadata
        doc_map = load_doc_map()
        doc_map[doc_id] = {
            "filename": filename,
            "uploaded_at": datetime.now().isoformat(),
            "deleted": False,
            "status": "uploaded",
            "file_size": file_size,
            "progress": 0
        }
        save_doc_map(doc_map)

        # Start background processing
        background_tasks.add_task(process_document_background, doc_id, file_path, filename)

        processing_time = time.time() - start_time

        return DocumentResponse(
            document_id=doc_id,
            filename=filename,
            status="processing",
            upload_time=datetime.now(),
            chunk_count=0,
            processing_time=processing_time,
            file_size=file_size
        )

    except StudyPathException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise StudyPathException(f"Upload failed: {str(e)}", 500)


@app.post("/ask/", response_model=QueryResponse)
async def ask_question(req: QueryRequest):
    start_time = time.time()

    try:
        logger.info(f"Processing query: {req.question}")

        # Validate document exists and is ready
        doc_map = load_doc_map()
        if req.document_id:
            if req.document_id not in doc_map:
                raise StudyPathException("Document not found", 404)

            if doc_map[req.document_id].get("deleted", False):
                raise StudyPathException("Document has been deleted", 404)

            if doc_map[req.document_id].get("status") != "completed":
                status = doc_map[req.document_id].get("status", "unknown")
                raise StudyPathException(f"Document is not ready yet. Current status: {status}", 400)

        # Load vectorstore
        vector_path = os.path.join(VECTOR_DIR, req.document_id)
        if not os.path.exists(vector_path):
            raise StudyPathException("Document vectorstore not found", 404)

        db = FAISS.load_local(vector_path, embedding_model, allow_dangerous_deserialization=True)

        # Get relevant context
        relevant_docs = get_relevant_context(db, req.question, req.top_k)

        # Create prompt
        enhanced_prompt = create_simple_prompt(req.question, relevant_docs, req.language)

        # Try different models
        models_to_try = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4"]

        llm = None
        for model_name in models_to_try:
            try:
                llm = ChatOpenAI(model_name=model_name, temperature=0)
                llm.invoke("Hi")
                logger.info(f"Successfully using model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Model {model_name} not available: {str(e)}")
                continue

        if llm is None:
            raise StudyPathException("No available OpenAI models found.", 500)

        # Get response
        response = llm.invoke(enhanced_prompt)
        answer = response.content

        logger.info(f"AI Response: {answer}")

        # Extract sources
        sources = list(set([doc.metadata.get("source", "Document") for doc in relevant_docs]))

        # Simple confidence
        confidence = min(0.95, 0.8 + (len(relevant_docs) * 0.03))

        response_time = time.time() - start_time

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            response_time=response_time,
            chunk_count=len(relevant_docs)
        )

    except StudyPathException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise StudyPathException(f"Query processing failed: {str(e)}", 500)


@app.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Get document processing status"""
    try:
        doc_map = load_doc_map()
        if document_id not in doc_map:
            raise StudyPathException("Document not found", 404)

        doc_info = doc_map[document_id]
        return {
            "document_id": document_id,
            "status": doc_info.get("status", "unknown"),
            "progress": doc_info.get("progress", 0),
            "filename": doc_info.get("filename", "unknown"),
            "chunk_count": doc_info.get("chunk_count", 0),
            "ocr_used": doc_info.get("ocr_used", False),
            "error_message": doc_info.get("error_message")
        }

    except StudyPathException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise StudyPathException("Failed to get document status", 500)


@app.get("/list_documents/")
async def list_documents():
    try:
        doc_map = load_doc_map()
        docs = []
        for doc_id, meta in doc_map.items():
            if not meta.get("deleted", False):
                docs.append({
                    "document_id": doc_id,
                    "filename": meta["filename"],
                    "uploaded_at": meta["uploaded_at"],
                    "status": meta.get("status", "unknown"),
                    "chunk_count": meta.get("chunk_count", 0),
                    "file_size": meta.get("file_size", 0),
                    "ocr_used": meta.get("ocr_used", False)
                })
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise StudyPathException("Failed to retrieve documents", 500)


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        doc_map = load_doc_map()
        if document_id not in doc_map:
            raise StudyPathException("Document not found", 404)

        # Mark as deleted
        doc_map[document_id]["deleted"] = True
        doc_map[document_id]["deleted_at"] = datetime.now().isoformat()
        save_doc_map(doc_map)

        # Delete vector store files
        vector_path = os.path.join(VECTOR_DIR, document_id)
        if os.path.exists(vector_path):
            shutil.rmtree(vector_path)

        return {"message": f"Document {document_id} deleted successfully"}

    except StudyPathException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise StudyPathException("Failed to delete document", 500)


@app.get("/health/")
async def health():
    try:
        ram_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=0.1)

        # Count documents
        doc_map = load_doc_map()
        total_docs = len([d for d in doc_map.values() if not d.get("deleted", False)])

        return {
            "status": "🧠 Clean RAG with OpenAI Embeddings",
            "version": "4.0.0",
            "timestamp": datetime.now().isoformat(),
            "RAM_Usage_Percent": ram_usage,
            "CPU_Usage_Percent": cpu_usage,
            "total_documents": total_docs,
            "features": {
                "openai_embeddings": True,
                "larger_chunks": True,
                "simple_retrieval": True,
                "clean_architecture": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}