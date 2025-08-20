// src/app/components/FileUpload.tsx
'use client';
import React, { useState } from 'react';
import { Upload, FileText, CheckCircle } from 'lucide-react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  isUploading: boolean;
}

export default function FileUpload({ onFileSelect, selectedFile, isUploading }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type === 'application/pdf') {
      onFileSelect(droppedFile);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  return (
    <div className="bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 p-8 mb-8">
      <div
        className={`
          relative border-2 border-dashed rounded-2xl p-12 transition-all duration-300 cursor-pointer
          ${isDragging 
            ? 'border-purple-400 bg-purple-400/10 scale-105' 
            : selectedFile 
            ? 'border-green-400 bg-green-400/10'
            : 'border-gray-400 hover:border-purple-400 hover:bg-purple-400/5'
          }
          ${isUploading ? 'pointer-events-none opacity-50' : ''}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isUploading && document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".pdf"
          onChange={handleFileInput}
          className="hidden"
          disabled={isUploading}
        />

        <div className="text-center">
          {isUploading ? (
            <div className="space-y-4">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-500/20 rounded-full">
                <div className="w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
              </div>
              <div>
                <p className="text-lg font-semibold text-white">Processing...</p>
                <p className="text-purple-400">Analyzing your document</p>
              </div>
            </div>
          ) : selectedFile ? (
            <div className="space-y-4">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-green-500/20 rounded-full">
                <CheckCircle className="w-8 h-8 text-green-400" />
              </div>
              <div>
                <p className="text-lg font-semibold text-white">{selectedFile.name}</p>
                <p className="text-green-400">Ready to analyze • {(selectedFile.size / 1024 / 1024).toFixed(1)} MB</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-purple-500/20 rounded-full">
                <Upload className="w-8 h-8 text-purple-400" />
              </div>
              <div>
                <p className="text-lg font-semibold text-white">
                  {isDragging ? 'Drop your file here' : 'Upload Your Document'}
                </p>
                <p className="text-gray-400">
                  Drag & drop your PDF or click to browse
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  Supports PDF files up to 10MB
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}