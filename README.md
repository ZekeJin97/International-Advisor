# StudyPath AI-Advisor: Multilingual GPT-4 Advisor for International Students

StudyPath is a minimal but complete prototype of a multilingual Retrieval-Augmented Generation (RAG) system that helps international students navigate U.S. college admissions and visa policies. It combines FastAPI, FAISS, sentence-transformer embeddings, and OpenAI GPT-4, wrapped in a lightweight React frontend.

💡 Features

✅ Upload PDF documents (e.g., I-20, F-1 visa guides)

🪀 Automatically chunk, embed, and index documents with FAISS

🧠 Ask questions in English, Spanish, or Mandarin Chinese

🌐 Uses GPT-4 to generate context-aware responses

⚖️ Soft and hard delete functionality for document management

✨ FastAPI backend + Next.js frontend with Tailwind UI

🪧 Stack

Frontend: React + Next.js + Tailwind CSS

Backend: main.py - FastAPI + pdfplumber + FAISS + Sentence-Transformer (MiniLM-L6-v2)

LLM: OpenAI GPT-4 via openai.ChatCompletion

Infra: Local dev only for now (privacy-conscious)

