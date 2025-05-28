# StudyPath AI-Advisor 🧠📄  
**Multilingual GPT-4-Powered RAG System for International Students**

StudyPath is a multilingual Retrieval-Augmented Generation (RAG) system that helps international students navigate U.S. college admissions and visa policies. It supports scanned PDF documents like I-20s, extracts critical data via OCR, and uses GPT-4 to answer questions in natural language — all through a lightweight frontend. 🔄 Easily extensible to other document-heavy domains (e.g., immigration, healthcare, legal aid) with minimal changes.

![Demo Screenshot](./ScreenshotDEMO.png)

---

### 💡 Features

- ✅ **PDF Uploads:** Supports scanned or digital PDFs (Form I-20, visa instructions, admission letters)  
- 🧠 **Auto Chunk & Embed:** Uses `sentence-transformers` + FAISS to index documents  
- 🌍 **Multilingual Q&A:** English, Spanish, or Mandarin Chinese  
- 🧾 **Contextual Understanding:** Merges personal docs with global policy info  
- 💬 **GPT-4 Responses:** Hybrid reasoning with document + background knowledge  
- 🧼 **Document Management:** Soft & hard delete endpoints  
- 🧪 **OCR Fallback:** Extracts text from scanned files using Tesseract  

---

## 🧱 Stack

**Frontend:**  
- React + Next.js  
- Tailwind CSS

**Backend:**  
- FastAPI  
- pdfplumber + pytesseract (OCR)  
- FAISS + Sentence-Transformers (MiniLM-L6-v2)  
- OpenAI API


