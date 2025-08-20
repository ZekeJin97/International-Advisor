// src/app/components/QuestionInput.tsx
'use client';
import React, { useState } from 'react';
import { MessageCircle, Sparkles, Send } from 'lucide-react';

interface QuestionInputProps {
  question: string;
  onQuestionChange: (question: string) => void;
  onSubmit: () => void;
  language: string;
  onLanguageChange: (language: string) => void;
  disabled: boolean;
  isLoading: boolean;
}

const LANGUAGES = [
  { code: 'en', label: '🇺🇸 English', name: 'English' },
  { code: 'es', label: '🇪🇸 Español', name: 'Spanish' },
  { code: 'zh-cn', label: '🇨🇳 中文', name: 'Chinese' }
];

const SAMPLE_QUESTIONS = [
  "What is the I-20 deadline for Fall Semester 2025?",
  "When can I enter the U.S. for my program?",
  "What documents do I need for my visa application?",
  "What are the admission requirements?",
];

export default function QuestionInput({
  question,
  onQuestionChange,
  onSubmit,
  language,
  onLanguageChange,
  disabled,
  isLoading
}: QuestionInputProps) {
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (question.trim() && !disabled && !isLoading) {
      onSubmit();
    }
  };

  const handleSampleClick = (sampleQuestion: string) => {
    onQuestionChange(sampleQuestion);
  };

  const canSubmit = question.trim() && !disabled && !isLoading;

  return (
    <div className="bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 p-8 mb-8">
      <div className="flex items-center gap-3 mb-6">
        <MessageCircle className="w-6 h-6 text-purple-400" />
        <h2 className="text-xl font-semibold text-white">Ask Your Question</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Question Input */}
        <div className={`relative transition-all duration-200 ${isFocused ? 'ring-2 ring-purple-500' : ''} rounded-xl`}>
          <textarea
            value={question}
            onChange={(e) => onQuestionChange(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Ask me anything about your documents..."
            className="w-full h-32 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none resize-none transition-all duration-200"
            disabled={disabled}
          />

          {/* Character count */}
          <div className="absolute bottom-3 right-3 text-xs text-gray-500">
            {question.length}/500
          </div>
        </div>

        {/* Sample Questions */}
        {!question && (
          <div className="space-y-2">
            <p className="text-sm text-gray-400">Try asking:</p>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_QUESTIONS.map((sample, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => handleSampleClick(sample)}
                  className="text-xs px-3 py-1 bg-white/5 border border-white/10 rounded-full text-gray-300 hover:text-white hover:border-purple-400 transition-colors"
                  disabled={disabled}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="flex gap-4">
          {/* Language Selector */}
          <select
            value={language}
            onChange={(e) => onLanguageChange(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 min-w-[140px]"
            disabled={disabled}
          >
            {LANGUAGES.map(lang => (
              <option key={lang.code} value={lang.code} className="bg-gray-800">
                {lang.label}
              </option>
            ))}
          </select>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={!canSubmit}
            className={`
              flex-1 font-semibold py-3 px-6 rounded-xl transition-all duration-200 flex items-center justify-center gap-2
              ${canSubmit
                ? 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white transform hover:scale-[1.02]'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
              }
            `}
          >
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                Thinking...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Ask AI
                <Send className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}