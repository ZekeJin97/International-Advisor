// src/app/page.tsx
"use client";
import React, { useState } from 'react';
import { Brain, Menu, Settings } from 'lucide-react';
import FileUpload from './components/FileUpload';
import QuestionInput from './components/QuestionInput';
import ResponseDisplay from './components/ResponseDisplay';

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState("");
  const [language, setLanguage] = useState("en");
  const [docId, setDocId] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [sources, setSources] = useState<string[]>([]);
  const [confidence, setConfidence] = useState(0);
  const [responseTime, setResponseTime] = useState(0);

  const handleFileSelect = async (selectedFile: File) => {
    setFile(selectedFile);
    setUploading(true);
    setResponse(""); // Clear previous response

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch("http://127.0.0.1:8000/upload/", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setDocId(data.document_id || data.doc_id);
      
      // You could show a success notification here
      console.log("Upload successful:", data.message);
    } catch (err) {
      console.error("Upload failed:", err);
      // You could show an error notification here
    } finally {
      setUploading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!question.trim()) return;
    
    setIsAsking(true);
    const startTime = Date.now();

    try {
      const payload: Record<string, any> = {
        question,
        top_k: 3,
        language,
      };
      if (docId) payload.document_id = docId;

      const res = await fetch("http://127.0.0.1:8000/ask/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      setResponse(data.answer || "No response received.");
      setSources(data.sources || []);
      setConfidence(data.confidence || 0);
      setResponseTime((Date.now() - startTime) / 1000);
    } catch (err) {
      console.error("Ask failed:", err);
      setResponse("Sorry, something went wrong. Please try again.");
      setSources([]);
      setConfidence(0);
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Navigation */}
      <nav className="relative z-10 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white">StudyPath</span>
          </div>
          
          <div className="flex items-center gap-4">
            <button className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-white/10 transition-colors">
              <Settings className="w-5 h-5" />
            </button>
            <button className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-white/10 transition-colors">
              <Menu className="w-5 h-5" />
            </button>
          </div>
        </div>
      </nav>

      {/* Header */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-400/20 to-pink-600/20 blur-3xl"></div>
        <div className="relative px-6 py-16 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 mb-6">
            <Brain className="w-5 h-5 text-purple-400" />
            <span className="text-white font-medium">AI-Powered Document Advisor</span>
          </div>
          <h1 className="text-5xl font-bold text-white mb-4">
            Get Instant Answers
          </h1>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Upload your visa or college documents and get intelligent, accurate answers to your questions in seconds
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-6 pb-16">
        {/* File Upload */}
        <FileUpload 
          onFileSelect={handleFileSelect}
          selectedFile={file}
          isUploading={uploading}
        />

        {/* Question Input */}
        <QuestionInput
          question={question}
          onQuestionChange={setQuestion}
          onSubmit={handleAskQuestion}
          language={language}
          onLanguageChange={setLanguage}
          disabled={!file || uploading}
          isLoading={isAsking}
        />

        {/* Response Display */}
        <ResponseDisplay
          response={response}
          sources={sources}
          confidence={confidence}
          responseTime={responseTime}
          isLoading={isAsking}
        />
      </div>

      {/* Footer */}
      <footer className="border-t border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="max-w-4xl mx-auto px-6 py-8 text-center">
          <p className="text-gray-400 text-sm">
            Powered by AI • Built for International Students • 
            <span className="text-purple-400 ml-1">StudyPath v2.0</span>
          </p>
        </div>
      </footer>
    </div>
  );
}