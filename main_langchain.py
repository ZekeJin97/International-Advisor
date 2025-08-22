from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
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
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Tesseract path based on environment
if os.environ.get('GOOGLE_CLOUD_PROJECT') or os.environ.get('GAE_ENV'):
    # Running on Google Cloud
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    logger.info("Using Google Cloud Tesseract configuration")
else:
    # Running locally on Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    logger.info("Using local Windows Tesseract configuration")

# ═══════════════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="StudyPath AI Advisor - Cloud Deployment",
    description="RAG system with OpenAI embeddings optimized for Google Cloud",
    version="4.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS configuration for cloud deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins for demo purposes
        "http://localhost:3000",  # Local development
        "https://studypath-frontend.vercel.app",  # Vercel deployment
        "https://studypath-ai-demo.vercel.app",  # Alternative Vercel URL
        "https://studypath.ai",  # Custom domain if you have one
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment configuration
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectorstores"
DOC_MAP_PATH = os.path.join(VECTOR_DIR, "doc_map.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# OpenAI Embeddings - Better for cloud deployment
try:
    embedding_model = OpenAIEmbeddings()
    logger.info("Successfully initialized OpenAI embeddings")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI embeddings: {e}")
    raise

# Rate limiting storage (in production, use Redis)
request_counts = defaultdict(list)


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
    environment: Optional[str] = None


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

    @validator('top_k')
    def validate_top_k(cls, v):
        if v < 1 or v > 10:
            raise ValueError('top_k must be between 1 and 10')
        return v


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: Optional[float] = None
    response_time: float
    chunk_count: int
    environment: Optional[str] = None


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
        content={
            "error": exc.message,
            "timestamp": datetime.now().isoformat(),
            "environment": ENVIRONMENT
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════════════════════════════════════════

def rate_limit_check(client_ip: str, max_requests: int = 20, window: int = 3600):
    """Simple rate limiting: max_requests per hour"""
    now = time.time()

    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if now - req_time < window
    ]

    # Check if over limit
    if len(request_counts[client_ip]) >= max_requests:
        return False

    # Add current request
    request_counts[client_ip].append(now)
    return True


# ═══════════════════════════════════════════════════════════════════════════════════════
# METADATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def load_doc_map():
    if os.path.exists(DOC_MAP_PATH):
        try:
            with open(DOC_MAP_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load doc_map: {e}")
            return {}
    return {}


def save_doc_map(doc_map):
    try:
        with open(DOC_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(doc_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save doc_map: {e}")


# ═══════════════════════════════════════════════════════════════════════════════════════
# OCR FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def extract_with_ocr(file_path):
    """Enhanced OCR with better preprocessing for cloud deployment"""
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

                    # OCR with optimized config for cloud
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
# CHUNKING - OPTIMIZED FOR CLOUD
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_proper_chunks(pages):
    """
    Optimized chunking for cloud deployment with better memory management
    """
    # Combine all pages
    full_text = "\n\n".join([page.page_content for page in pages])

    logger.info(f"Full document length: {len(full_text)} characters")

    # Use recursive text splitter with cloud-optimized parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Optimal for OpenAI embeddings
        chunk_overlap=200,  # Good overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # Split the text
    texts = text_splitter.split_text(full_text)

    # Create documents with enhanced metadata
    documents = []
    for i, text in enumerate(texts):
        documents.append(Document(
            page_content=text,
            metadata={
                "source": pages[0].metadata.get("source", "Document"),
                "chunk_index": i,
                "chunk_size": len(text),
                "environment": ENVIRONMENT,
                "processed_at": datetime.now().isoformat()
            }
        ))

    avg_size = sum(len(doc.page_content) for doc in documents) // len(documents) if documents else 0
    logger.info(f"Created {len(documents)} chunks with average size {avg_size} characters")

    return documents


# ═══════════════════════════════════════════════════════════════════════════════════════
# RETRIEVAL - CLOUD OPTIMIZED
# ═══════════════════════════════════════════════════════════════════════════════════════

def get_relevant_context(db, question, top_k):
    """
    Cloud-optimized retrieval with error handling
    """
    try:
        # Simple similarity search with OpenAI embeddings
        docs = db.similarity_search(question, k=top_k * 2)

        logger.info(f"Retrieved {len(docs)} chunks for question: {question[:50]}...")
        for i, doc in enumerate(docs[:3]):  # Log first 3 chunks only
            logger.info(f"Retrieved chunk {i + 1}: {doc.page_content[:100]}...")

        return docs
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise StudyPathException(f"Failed to retrieve relevant context: {str(e)}", 500)


# ═══════════════════════════════════════════════════════════════════════════════════════
# PROMPTING - ENHANCED FOR CLOUD
# ═══════════════════════════════════════════════════════════════════════════════════════

def create_cloud_optimized_prompt(question, relevant_docs, language="en"):
    """
    Cloud-optimized prompt with better context management
    """

    # Build context with size limits for cloud deployment
    max_context_length = 8000  # Stay within token limits
    context_parts = []
    current_length = 0

    for i, doc in enumerate(relevant_docs):
        section = f"Section {i + 1}:\n{doc.page_content}\n\n"
        if current_length + len(section) > max_context_length:
            break
        context_parts.append(section)
        current_length += len(section)

    context = "".join(context_parts)

    # Enhanced prompt for international student advisory
    system_prompt = """You are an expert international student advisor with deep knowledge of F-1 visa regulations and U.S. university policies. 

Your expertise includes:
- F-1 visa entry rules (30-day rule, program start dates)
- OPT application procedures and timelines
- I-20 document interpretation
- Academic program requirements
- Immigration compliance

Answer using BOTH the document information provided AND your regulatory knowledge."""

    user_prompt = f"""Question: {question}

Document sections:
{context}

Instructions:
1. Extract relevant information from the document sections
2. Apply your knowledge of F-1 visa and international student regulations
3. Provide a comprehensive, helpful answer
4. Distinguish between document-specific info and general regulations
5. Be specific about dates, deadlines, and requirements

Answer comprehensively to help the student understand both their specific document and applicable regulations."""

    if language == "es":
        user_prompt += "\n\nResponde en español, manteniendo la precisión técnica."
    elif language == "zh-cn":
        user_prompt += "\n\n用简体中文回答，保持技术准确性。"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# ═══════════════════════════════════════════════════════════════════════════════════════
# BACKGROUND PROCESSING - CLOUD OPTIMIZED
# ═══════════════════════════════════════════════════════════════════════════════════════

async def process_document_background(doc_id: str, file_path: str, filename: str):
    """Cloud-optimized background processing with better error handling"""
    doc_map = load_doc_map()

    try:
        logger.info(f"Starting background processing for {doc_id} in {ENVIRONMENT}")

        # Update status
        doc_map[doc_id]["status"] = "processing"
        doc_map[doc_id]["progress"] = 0
        doc_map[doc_id]["environment"] = ENVIRONMENT
        save_doc_map(doc_map)

        # Try native PDF extraction first
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

        # Cloud-optimized FAISS creation
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
        logger.error(f"Processing failed for {doc_id}: {str(e)}")
        doc_map[doc_id]["status"] = "error"
        doc_map[doc_id]["error_message"] = str(e)
        doc_map[doc_id]["failed_at"] = datetime.now().isoformat()
        save_doc_map(doc_map)

    finally:
        # Clean up uploaded file to save space
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up uploaded file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════════════
# HEALTH CHECKS
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "message": "StudyPath AI Advisor v4.1 - Cloud Deployment Ready",
        "status": "operational",
        "environment": ENVIRONMENT,
        "features": ["OpenAI Embeddings", "Cloud Optimized", "Rate Limited", "Auto-Deploy Ready"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Enhanced health check for Google Cloud Load Balancer"""
    try:
        # Test OpenAI connection
        test_embedding = embedding_model.embed_query("test")

        # Test filesystem
        test_file = os.path.join(UPLOAD_DIR, "health_test.txt")
        with open(test_file, "w") as f:
            f.write("health check")
        os.remove(test_file)

        return {
            "status": "healthy",
            "service": "studypath-backend",
            "environment": ENVIRONMENT,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "openai_embeddings": "ok",
                "filesystem": "ok"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness check"""
    return {"status": "ready", "timestamp": datetime.now().isoformat()}


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN ROUTES
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.post("/upload/", response_model=DocumentResponse)
async def upload_pdf(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    start_time = time.time()
    client_ip = request.client.host

    try:
        # Rate limiting
        if not rate_limit_check(client_ip, max_requests=10):
            raise StudyPathException("Rate limit exceeded. Please try again later.", 429)

        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise StudyPathException("Only PDF files are supported", 400)

        # Read file content
        contents = await file.read()
        file_size = len(contents)

        # Size validation
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            raise StudyPathException(f"File size must be less than {max_size // 1024 // 1024}MB", 400)

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
            "progress": 0,
            "environment": ENVIRONMENT,
            "client_ip": client_ip
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
            file_size=file_size,
            environment=ENVIRONMENT
        )

    except StudyPathException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise StudyPathException(f"Upload failed: {str(e)}", 500)


@app.post("/ask/", response_model=QueryResponse)
async def ask_question(request: Request, req: QueryRequest):
    start_time = time.time()
    client_ip = request.client.host

    try:
        # Rate limiting
        if not rate_limit_check(client_ip, max_requests=20):
            raise StudyPathException("Rate limit exceeded. Please try again later.", 429)

        logger.info(f"Processing query: {req.question[:100]}... (Environment: {ENVIRONMENT})")

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

        # Create optimized prompt
        messages = create_cloud_optimized_prompt(req.question, relevant_docs, req.language)

        # Try different models with cloud-optimized settings
        models_to_try = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"]

        llm = None
        for model_name in models_to_try:
            try:
                llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=0,
                    max_tokens=1000,  # Limit for cloud efficiency
                    timeout=30  # Timeout for cloud deployment
                )
                # Test the model
                test_response = llm.invoke([{"role": "user", "content": "Hi"}])
                logger.info(f"Successfully using model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Model {model_name} not available: {str(e)}")
                continue

        if llm is None:
            raise StudyPathException("No available OpenAI models found.", 500)

        # Get response
        response = llm.invoke(messages)
        answer = response.content

        logger.info(f"AI Response generated successfully (Length: {len(answer)} chars)")

        # Extract sources
        sources = list(set([doc.metadata.get("source", "Document") for doc in relevant_docs]))

        # Calculate confidence
        confidence = min(0.95, 0.8 + (len(relevant_docs) * 0.03))

        response_time = time.time() - start_time

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            response_time=response_time,
            chunk_count=len(relevant_docs),
            environment=ENVIRONMENT
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
            "error_message": doc_info.get("error_message"),
            "environment": doc_info.get("environment", ENVIRONMENT)
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
                    "ocr_used": meta.get("ocr_used", False),
                    "environment": meta.get("environment", ENVIRONMENT)
                })
        return {"documents": docs, "total": len(docs), "environment": ENVIRONMENT}
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

        return {
            "message": f"Document {document_id} deleted successfully",
            "environment": ENVIRONMENT
        }

    except StudyPathException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise StudyPathException("Failed to delete document", 500)


@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint for monitoring"""
    try:
        doc_map = load_doc_map()
        total_docs = len([d for d in doc_map.values() if not d.get("deleted", False)])
        completed_docs = len([d for d in doc_map.values() if d.get("status") == "completed"])

        return {
            "environment": ENVIRONMENT,
            "total_documents": total_docs,
            "completed_documents": completed_docs,
            "success_rate": completed_docs / total_docs if total_docs > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics failed: {str(e)}")
        return {"error": "Failed to get metrics"}


# ═══════════════════════════════════════════════════════════════════════════════════════
# STARTUP AND MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    logger.info(f"StudyPath AI Advisor starting up in {ENVIRONMENT} environment")
    logger.info(f"OpenAI embeddings initialized: {embedding_model is not None}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"

    logger.info(f"Starting StudyPath AI Advisor on {host}:{port} in {ENVIRONMENT} mode")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )