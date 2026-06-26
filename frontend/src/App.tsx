import React, { useState, useEffect, useRef } from 'react';

interface Book {
  id: string;
  name: string;
  chunk_count: number;
  embedding_model: string;
}

interface Source {
  chunk_id: string;
  rank: number;
  score: number;
  content: string;
  source: string;
  page: number;
  semantic_score: number;
  bm25_score: number;
  rerank_score: number;
}

interface Message {
  sender: 'user' | 'assistant';
  text: string;
  sources?: Source[];
}

const BACKEND_URL = 'http://localhost:8000';

export default function App() {
  // --- LLM & Authentication States ---
  const [apiKey, setApiKey] = useState('');
  const [llmProvider, setLlmProvider] = useState<'github' | 'openai' | 'mock'>('github');
  const [llmModel, setLlmModel] = useState('gpt-4o');
  const [llmTemp, setLlmTemp] = useState(0.1);

  // --- Ingestion States ---
  const [chunkSize, setChunkSize] = useState(800);
  const [chunkOverlap, setChunkOverlap] = useState(150);
  const [ocrMode, setOcrMode] = useState('Disable OCR');
  const [embeddingModel, setEmbeddingModel] = useState('BAAI/bge-small-en-v1.5');
  const [uploading, setUploading] = useState(false);

  // --- Retrieval States ---
  const [topK, setTopK] = useState(5);
  const [alpha, setAlpha] = useState(0.7);
  const [useReranker, setUseReranker] = useState(true);
  const [rerankerModel, setRerankerModel] = useState('Cross-Encoder (ms-marco-MiniLM)');

  // --- Data & In-Memory States ---
  const [books, setBooks] = useState<Book[]>([]);
  const [selectedBook, setSelectedBook] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<Message[]>([]);
  const [queryInput, setQueryInput] = useState('');
  const [queryLoading, setQueryLoading] = useState(false);

  // --- Background Ingestion Polling States ---
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<{
    status: string;
    progress: number;
    book_name: string;
    error: string | null;
  } | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load books list on startup
  useEffect(() => {
    fetchBooks();
  }, []);

  // Scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, queryLoading]);

  // Adjust default models when provider changes
  useEffect(() => {
    if (llmProvider === 'github') {
      setLlmModel('gpt-4o');
    } else if (llmProvider === 'openai') {
      setLlmModel('gpt-4o-mini');
    } else if (llmProvider === 'mock') {
      setLlmModel('mock-generator');
    }
  }, [llmProvider]);

  // Poll Ingestion task status if active
  useEffect(() => {
    if (!activeTaskId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/status/${activeTaskId}`);
        if (!res.ok) throw new Error('Failed to fetch status');
        
        const data = await res.json();
        setTaskStatus(data);

        if (data.status === 'completed') {
          setActiveTaskId(null);
          setTaskStatus(null);
          setUploading(false);
          fetchBooks();
          // Auto-select the newly uploaded book (sanitize stem format to match backend folder name)
          const derivedId = data.book_name.replace(/\.[^/.]+$/, "").replace(/\s+/g, "_");
          setSelectedBook(derivedId);
        } else if (data.status === 'failed') {
          setActiveTaskId(null);
          setUploading(false);
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [activeTaskId]);

  const fetchBooks = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/books`);
      if (res.ok) {
        const data = await res.json();
        setBooks(data.books);
        if (data.books.length > 0 && !selectedBook) {
          setSelectedBook(data.books[0].id);
        }
      }
    } catch (err) {
      console.error('Failed to fetch books:', err);
    }
  };

  // --- Handlers ---
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    setUploading(true);
    setTaskStatus({
      status: 'Uploading file...',
      progress: 5,
      book_name: file.name,
      error: null
    });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('chunk_size', chunkSize.toString());
    formData.append('chunk_overlap', chunkOverlap.toString());
    formData.append('ocr_mode', ocrMode);
    formData.append('embedding_model', embeddingModel);

    try {
      const res = await fetch(`${BACKEND_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await res.json();
      setActiveTaskId(data.task_id);
    } catch (err: any) {
      setUploading(false);
      setTaskStatus({
        status: 'failed',
        progress: 100,
        book_name: file.name,
        error: err.message || 'Failed to connect to server'
      });
    }
  };

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryInput.trim() || !selectedBook || queryLoading) return;

    const userMessage = queryInput;
    setQueryInput('');
    setChatHistory(prev => [...prev, { sender: 'user', text: userMessage }]);
    setQueryLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          book_name: selectedBook,
          query: userMessage,
          top_k: topK,
          alpha: alpha,
          use_reranker: useReranker,
          reranker_model: rerankerModel,
          llm_provider: llmProvider,
          llm_key: apiKey ? apiKey : undefined,
          llm_model: llmModel,
          llm_temp: llmTemp
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Query failed');
      }

      const data = await res.json();
      setChatHistory(prev => [...prev, {
        sender: 'assistant',
        text: data.answer,
        sources: data.retrieved_chunks
      }]);
    } catch (err: any) {
      setChatHistory(prev => [...prev, {
        sender: 'assistant',
        text: `Error querying backend: ${err.message}`
      }]);
    } finally {
      setQueryLoading(false);
    }
  };

  const handleClearSystem = async () => {
    if (!window.confirm("Are you sure you want to clear all indexed books and cache?")) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/clear`, { method: 'POST' });
      if (res.ok) {
        setBooks([]);
        setSelectedBook('');
        setChatHistory([]);
        alert('System cleared successfully.');
      }
    } catch (err) {
      console.error('Clear failed:', err);
    }
  };

  return (
    <div className="app-container">
      {/* ─── Sidebar Control Panel ─── */}
      <aside className="sidebar">
        <div className="logo-container">
          <span className="logo-icon">📖</span>
          <span className="logo-text">RAG Book Control</span>
        </div>

        {/* 1. Authentication */}
        <div className="sidebar-section">
          <label className="sidebar-label">1. LLM Authentication</label>
          <select 
            className="select-field"
            value={llmProvider}
            onChange={(e) => setLlmProvider(e.target.value as any)}
          >
            <option value="github">GitHub Models</option>
            <option value="openai">OpenAI API</option>
            <option value="mock">Mock / Offline Mode</option>
          </select>
          <input 
            type="password"
            placeholder={
              llmProvider === 'github' ? "GitHub Token..." :
              llmProvider === 'openai' ? "OpenAI API Key..." :
              "Mock Mode - No Token Needed"
            }
            className="input-field"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            disabled={llmProvider === 'mock'}
          />
          <select
            className="select-field"
            value={llmModel}
            onChange={(e) => setLlmModel(e.target.value)}
          >
            {llmProvider === 'github' ? (
              <>
                <option value="gpt-4o">gpt-4o</option>
                <option value="gpt-4o-mini">gpt-4o-mini</option>
                <option value="meta-llama-3-8b-instruct">meta-llama-3-8b-instruct</option>
                <option value="cohere-command-r-plus">cohere-command-r-plus</option>
              </>
            ) : llmProvider === 'openai' ? (
              <>
                <option value="gpt-4o-mini">gpt-4o-mini</option>
                <option value="gpt-4o">gpt-4o</option>
                <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
              </>
            ) : (
              <>
                <option value="mock-generator">mock-generator (Offline)</option>
              </>
            )}
          </select>
          <div className="slider-container">
            <div className="slider-header">
              <span>Temperature</span>
              <span>{llmTemp.toFixed(2)}</span>
            </div>
            <input 
              type="range" min="0" max="1" step="0.05"
              className="range-slider"
              value={llmTemp}
              onChange={(e) => setLlmTemp(parseFloat(e.target.value))}
            />
          </div>
        </div>

        {/* 2. Ingestion Settings */}
        <div className="sidebar-section">
          <label className="sidebar-label">2. Book Settings</label>
          <div className="uploader-box">
            <span className="uploader-icon">📁</span>
            <div style={{ fontSize: '0.85rem', fontWeight: 500 }}>Upload Book PDF/TXT</div>
            <input 
              type="file" 
              accept=".pdf,.txt" 
              className="file-input" 
              onChange={handleFileUpload}
              disabled={uploading}
            />
          </div>

          <select 
            className="select-field"
            value={ocrMode}
            onChange={(e) => setOcrMode(e.target.value)}
          >
            <option value="Disable OCR">Disable OCR</option>
            <option value="Auto-detect scanned pages">Auto-detect scanned pages</option>
            <option value="Force OCR">Force OCR</option>
          </select>

          <div className="slider-container">
            <div className="slider-header">
              <span>Chunk Size</span>
              <span>{chunkSize}</span>
            </div>
            <input 
              type="range" min="200" max="2000" step="50"
              className="range-slider"
              value={chunkSize}
              onChange={(e) => setChunkSize(parseInt(e.target.value))}
            />
          </div>

          <div className="slider-container">
            <div className="slider-header">
              <span>Chunk Overlap</span>
              <span>{chunkOverlap}</span>
            </div>
            <input 
              type="range" min="0" max="500" step="10"
              className="range-slider"
              value={chunkOverlap}
              onChange={(e) => setChunkOverlap(parseInt(e.target.value))}
            />
          </div>

          <select
            className="select-field"
            value={embeddingModel}
            onChange={(e) => setEmbeddingModel(e.target.value)}
          >
            <option value="BAAI/bge-small-en-v1.5">BAAI/bge-small-en-v1.5</option>
            <option value="sentence-transformers/all-MiniLM-L6-v2">all-MiniLM-L6-v2</option>
            <option value="intfloat/e5-small-v2">e5-small-v2</option>
          </select>
        </div>

        {/* 3. Retrieval Parameters */}
        <div className="sidebar-section">
          <label className="sidebar-label">3. Retrieval Parameters</label>
          <div className="slider-container">
            <div className="slider-header">
              <span>Retrieve Top K</span>
              <span>{topK}</span>
            </div>
            <input 
              type="range" min="1" max="20" step="1"
              className="range-slider"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
            />
          </div>

          <div className="slider-container">
            <div className="slider-header">
              <span>Hybrid Weight (Alpha)</span>
              <span>{alpha.toFixed(2)}</span>
            </div>
            <input 
              type="range" min="0" max="1" step="0.05"
              className="range-slider"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem' }}>
            <input 
              type="checkbox" 
              checked={useReranker}
              onChange={(e) => setUseReranker(e.target.checked)}
              style={{ accentColor: '#6c5ce7', width: '16px', height: '16px' }}
            />
            <span>Use Reranker</span>
          </div>

          <select 
            className="select-field"
            value={rerankerModel}
            onChange={(e) => setRerankerModel(e.target.value)}
            disabled={!useReranker}
          >
            <option value="Cross-Encoder (ms-marco-MiniLM)">Cross-Encoder (MiniLM)</option>
            <option value="BGE Reranker (bge-reranker-base)">BGE Reranker (Base)</option>
          </select>
        </div>

        <div style={{ marginTop: 'auto' }}>
          <button className="btn-secondary" onClick={handleClearSystem}>Clear System Caches</button>
        </div>
      </aside>

      {/* ─── Main Content Chat Panel ─── */}
      <main className="main-content">
        {/* Header */}
        <div className="chat-header">
          <div className="chat-header-title">
            <span>📖 Active Context:</span>
            {books.length > 0 && selectedBook ? (
              <select
                className="select-field"
                style={{ width: 'auto', padding: '0.4rem 2rem 0.4rem 1rem', background: '#1c1e29', border: 'none' }}
                value={selectedBook}
                onChange={(e) => {
                  setSelectedBook(e.target.value);
                  setChatHistory([]);
                }}
              >
                {books.map(b => (
                  <option key={b.id} value={b.id}>
                    {b.name} ({b.chunk_count} Chunks)
                  </option>
                ))}
              </select>
            ) : (
              <span style={{ color: '#b2bec3', fontSize: '0.95rem' }}>No book loaded</span>
            )}
          </div>
          {books.length > 0 && selectedBook && (
            <span className="chat-header-badge">
              Model: {books.find(b => b.id === selectedBook)?.embedding_model.split('/').pop()}
            </span>
          )}
        </div>

        {/* Ingestion Status Overlay */}
        {taskStatus && (
          <div style={{ padding: '0.5rem 2rem' }}>
            <div className="progress-card">
              <div className="progress-header">
                <span>Ingesting '{taskStatus.book_name}': {taskStatus.status}</span>
                <span>{taskStatus.progress}%</span>
              </div>
              <div className="progress-bar-bg">
                <div 
                  className="progress-bar-fill"
                  style={{ width: `${taskStatus.progress}%` }}
                />
              </div>
              {taskStatus.error && (
                <div style={{ color: '#ff7675', fontSize: '0.85rem', marginTop: '0.5rem', fontWeight: 500 }}>
                  Error: {taskStatus.error}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Chat Messages */}
        {chatHistory.length === 0 ? (
          <div className="welcome-screen">
            <span className="welcome-icon">📖</span>
            <div className="welcome-title">Welcome to RAG Book QA Tool</div>
            <p className="welcome-desc">
              Upload a textbook, configuration file, or paper in the left control panel. 
              The system will chunk, embed, and index it into a dense-sparse vector space. 
              Ask anything, and the assistant will answer with precise page citations.
            </p>
          </div>
        ) : (
          <div className="chat-messages">
            {chatHistory.map((chat, idx) => (
              <div 
                key={idx} 
                className={`chat-bubble-container ${chat.sender === 'user' ? 'user' : 'assistant'}`}
              >
                <div className={`chat-bubble ${chat.sender === 'user' ? 'user' : 'assistant'}`}>
                  <strong>{chat.sender === 'user' ? 'You:' : 'Assistant:'}</strong>
                  <div style={{ marginTop: '0.25rem', whiteSpace: 'pre-wrap' }}>{chat.text}</div>
                  
                  {/* Sources display */}
                  {chat.sources && chat.sources.length > 0 && (
                    <div className="sources-container">
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#00cec9', marginBottom: '0.25rem' }}>
                        Sources & Match Metrics:
                      </div>
                      {chat.sources.map((src, sidx) => (
                        <details key={sidx} style={{ outline: 'none' }}>
                          <summary className="source-item-header">
                            <span className="source-badge">Source {sidx + 1}</span>
                            <span>Page {src.page} of {src.source}</span>
                            <span className="score-badge">Score: {src.score.toFixed(4)}</span>
                          </summary>
                          <pre className="source-snippet">{src.content}</pre>
                        </details>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {queryLoading && (
              <div className="chat-bubble-container assistant">
                <div className="chat-bubble assistant" style={{ fontStyle: 'italic', color: '#b2bec3' }}>
                  Assistant is searching relevant passages and generating answer...
                </div>
              </div>
            )}
            
            <div ref={chatEndRef} />
          </div>
        )}

        {/* Input */}
        <form className="chat-input-container" onSubmit={handleQuerySubmit}>
          <input 
            type="text"
            className="chat-input-box"
            placeholder={selectedBook ? "Ask a question about the book..." : "Please upload a book first to query..."}
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            disabled={!selectedBook || queryLoading}
          />
          <button 
            type="submit" 
            className="btn-send"
            disabled={!selectedBook || queryLoading}
          >
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
