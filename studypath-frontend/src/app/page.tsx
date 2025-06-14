"use client";
import { useState } from "react";

export default function HomePage() {
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [question, setQuestion] = useState("");
    const [message, setMessage] = useState("");
    const [language, setLanguage] = useState("en");
    const [docId, setDocId] = useState("");

    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        setMessage("");

        const formData = new FormData();
        formData.append("file", file); // name must match backend param

        try {
            const res = await fetch("http://127.0.0.1:8000/upload/", {
                method: "POST",
                body: formData, // do NOT set Content-Type manually
            });

            const data = await res.json();
            setMessage(data.message || "Upload complete.");
            setDocId(data.document_id);
        } catch (err) {
            console.error("Upload failed:", err);
            setMessage("Upload failed.");
        } finally {
            setUploading(false);
        }
    };

    const handleAsk = async () => {
        setMessage("Thinking...");
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
            setMessage(data.answer || "No response.");
        } catch (err) {
            console.error("Ask failed:", err);
            setMessage("Something went wrong.");
        }
    };

    return (
        <main className="min-h-screen bg-black text-white flex flex-col items-center justify-center gap-6 p-6">
            <h1 className="text-2xl font-bold">📄 Upload Your Visa or College Doc</h1>

            <input
                accept="application/pdf"
                type="file"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="file-input file-input-bordered w-full max-w-xs"
            />

            <button
                onClick={handleUpload}
                disabled={!file || uploading}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
            >
                {uploading ? "Uploading..." : "Upload PDF"}
            </button>

            <div className="flex flex-col items-center gap-4 w-full max-w-lg">
                <label className="text-lg font-medium mt-6">🧠 Ask your question:</label>
                <textarea
                    placeholder="e.g., When can I enter the U.S. for my program?"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    className="textarea textarea-bordered w-full"
                />

                <select
                    className="select select-bordered w-full"
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                >
                    <option value="en">🇺🇸 English</option>
                    <option value="es">🇪🇸 Español</option>
                    <option value="zh-cn">🇨🇳 中文</option>
                </select>

                <button
                    onClick={handleAsk}
                    disabled={!question}
                    className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:opacity-50"
                >
                    Ask
                </button>

                {message && (
                    <div className="mt-4 p-4 bg-gray-800 rounded w-full">
                        <p className="whitespace-pre-line">{message}</p>
                    </div>
                )}
            </div>
        </main>
    );
}
