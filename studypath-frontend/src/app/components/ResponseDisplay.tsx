// src/app/components/ResponseDisplay.tsx
'use client';
import React, { useState } from 'react';
import { Brain, Copy, ThumbsUp, ThumbsDown, BookOpen, Clock, CheckCircle } from 'lucide-react';

interface ResponseDisplayProps {
  response: string;
  sources?: string[];
  confidence?: number;
  responseTime?: number;
  isLoading?: boolean;
}

export default function ResponseDisplay({
  response,
  sources = [],
  confidence = 0,
  responseTime = 0,
  isLoading = false
}: ResponseDisplayProps) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(response);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text');
    }
  };

  const handleFeedback = (type: 'up' | 'down') => {
    setFeedback(type);
    // Here you would typically send feedback to your backend
    console.log(`User feedback: ${type}`);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    return 'Low';
  };

  if (isLoading) {
    return (
      <div className="bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 p-8">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-6 h-6 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
          <h2 className="text-xl font-semibold text-white">AI is thinking...</h2>
        </div>

        <div className="space-y-4">
          {/* Loading skeleton */}
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-white/10 rounded w-full"></div>
            <div className="h-4 bg-white/10 rounded w-5/6"></div>
            <div className="h-4 bg-white/10 rounded w-4/6"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!response) return null;

  return (
    <div className="bg-white/5 backdrop-blur-xl rounded-3xl border border-white/10 p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Brain className="w-6 h-6 text-purple-400" />
          <h2 className="text-xl font-semibold text-white">AI Response</h2>
        </div>

        {/* Metadata */}
        <div className="flex items-center gap-4 text-sm text-gray-400">
          {confidence > 0 && (
            <div className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded-full ${getConfidenceColor(confidence).replace('text-', 'bg-')}`}></div>
              <span className={getConfidenceColor(confidence)}>
                {getConfidenceLabel(confidence)} confidence
              </span>
            </div>
          )}

          {responseTime > 0 && (
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              <span>{responseTime.toFixed(1)}s</span>
            </div>
          )}
        </div>
      </div>

      {/* Response Content */}
      <div className="bg-white/5 rounded-xl p-6 border border-white/10 mb-6">
        <div className="prose prose-invert max-w-none">
          <p className="text-gray-100 leading-relaxed whitespace-pre-line mb-0">
            {response}
          </p>
        </div>
      </div>

      {/* Sources */}
      {sources.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="w-5 h-5 text-blue-400" />
            <h3 className="text-sm font-semibold text-white">Sources</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {sources.map((source, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-blue-400/10 border border-blue-400/20 rounded-full text-xs text-blue-400"
              >
                {source}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center justify-between pt-4 border-t border-white/10">
        <div className="flex items-center gap-2">
          {/* Copy Button */}
          <button
            onClick={handleCopy}
            className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-300 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
          >
            {copied ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-400" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                Copy
              </>
            )}
          </button>
        </div>

        {/* Feedback */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-gray-400 mr-2">Was this helpful?</span>

          <button
            onClick={() => handleFeedback('up')}
            className={`p-1.5 rounded-lg transition-colors ${
              feedback === 'up' 
                ? 'bg-green-400/20 text-green-400' 
                : 'text-gray-400 hover:text-green-400 hover:bg-green-400/10'
            }`}
          >
            <ThumbsUp className="w-4 h-4" />
          </button>

          <button
            onClick={() => handleFeedback('down')}
            className={`p-1.5 rounded-lg transition-colors ${
              feedback === 'down' 
                ? 'bg-red-400/20 text-red-400' 
                : 'text-gray-400 hover:text-red-400 hover:bg-red-400/10'
            }`}
          >
            <ThumbsDown className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}