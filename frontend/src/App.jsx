import React, { useState, useEffect, useRef } from 'react';
import { 
  Sparkles, 
  UploadCloud, 
  FileText, 
  CheckCircle2, 
  AlertCircle, 
  TrendingUp, 
  BarChart3, 
  RefreshCw, 
  Download, 
  ArrowRight,
  ShieldCheck,
  Search,
  Filter,
  BrainCircuit,
  PieChart,
  HelpCircle,
  ThumbsUp,
  ThumbsDown,
  Minus
} from 'lucide-react';

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://127.0.0.1:8000'
  : '/api';

export default function App() {
  const [backendStatus, setBackendStatus] = useState('checking'); // 'connected' | 'offline' | 'checking'
  const [activeTab, setActiveTab] = useState('playground'); // 'playground' | 'csv' | 'methodology'

  // Global aggregate metrics (starts with chronological test set benchmark or dynamic updates)
  const [metrics, setMetrics] = useState({
    total: 78716,
    positive: 60016,
    neutral: 6245,
    negative: 12455,
    source: 'Test Benchmark (78,716 reviews)'
  });

  // Single review prediction state
  const [reviewText, setReviewText] = useState('');
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionError, setPredictionError] = useState(null);
  const [sessionHistory, setSessionHistory] = useState([]);

  // CSV upload state
  const [csvFile, setCsvFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [csvResponse, setCsvResponse] = useState(null);
  const [csvError, setCsvError] = useState(null);
  const [tableSearch, setTableSearch] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('ALL');
  const [tablePage, setTablePage] = useState(1);
  const fileInputRef = useRef(null);

  // Check backend health on mount
  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        setBackendStatus('connected');
      } else {
        setBackendStatus('offline');
      }
    } catch {
      setBackendStatus('offline');
    }
  };

  // Handle single review prediction
  const handlePredict = async (textToPredict = null) => {
    const text = (textToPredict || reviewText).trim();
    if (!text) {
      setPredictionError('Please enter a review before analyzing.');
      return;
    }

    setIsPredicting(true);
    setPredictionError(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Prediction failed');
      }

      const data = await res.json();
      setPredictionResult(data);

      // Add to session history
      const historyItem = {
        id: Date.now(),
        text: data.text,
        sentiment: data.sentiment,
        confidence: data.confidence,
        timestamp: new Date().toLocaleTimeString()
      };
      setSessionHistory(prev => [historyItem, ...prev.slice(0, 9)]);

      // Update global metrics count
      setMetrics(prev => {
        const key = data.sentiment.toLowerCase();
        return {
          ...prev,
          total: prev.total + 1,
          [key]: prev[key] + 1,
          source: 'Live Session Aggregates'
        };
      });

    } catch (err) {
      setPredictionError(err.message || 'Failed to connect to prediction backend.');
    } finally {
      setIsPredicting(false);
    }
  };

  // Preset example chips
  const examples = [
    {
      label: 'Fresh & Delicious',
      type: 'Positive',
      text: 'The dark chocolate almonds arrived fresh, crunchy, and absolutely delicious! Best purchase ever.'
    },
    {
      label: 'Stale & Damaged',
      type: 'Negative',
      text: 'Terrible experience. The bag was torn open, contents spilled, and the chips tasted stale and awful.'
    },
    {
      label: 'Average & Mild',
      type: 'Neutral',
      text: 'The flavor is okay, not bad but nothing exceptional. A bit too sweet for my personal taste.'
    },
    {
      label: 'Negation Handling',
      type: 'Negative',
      text: 'Not good at all. I would not buy this again and do not recommend it.'
    }
  ];

  // Handle CSV file selection
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setCsvFile(e.target.files[0]);
      setCsvError(null);
    }
  };

  // Upload CSV
  const handleUploadCsv = async () => {
    if (!csvFile) {
      setCsvError('Please select a CSV file first.');
      return;
    }

    setIsUploading(true);
    setCsvError(null);

    const formData = new FormData();
    formData.append('file', csvFile);

    try {
      const res = await fetch(`${API_BASE}/upload-csv`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'CSV upload failed');
      }

      const data = await res.json();
      setCsvResponse(data);
      setTablePage(1);

      // Update global dashboard metrics to match uploaded file
      setMetrics({
        total: data.total_reviews,
        positive: data.sentiment_distribution.Positive,
        neutral: data.sentiment_distribution.Neutral,
        negative: data.sentiment_distribution.Negative,
        source: `Uploaded CSV (${data.filename})`
      });

    } catch (err) {
      setCsvError(err.message || 'Error processing CSV file.');
    } finally {
      setIsUploading(false);
    }
  };

  // Download analyzed CSV
  const handleExportCsv = async () => {
    if (!csvFile) return;
    const formData = new FormData();
    formData.append('file', csvFile);

    try {
      const res = await fetch(`${API_BASE}/export-csv`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('Export failed');

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `analyzed_${csvFile.name}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      alert('Failed to export CSV: ' + err.message);
    }
  };

  // Quick 10-Row Sample CSV generator for instant demo testing
  const handleLoadSampleCsv = () => {
    const sampleCsvData = 
`Id,ProductId,Text
1,B001,"The organic peanut butter is creamy, rich, and tastes amazing!"
2,B002,"Arrived completely stale and expired. Smelled rancid. Never again!"
3,B003,"It is average tea. Not terrible, but certainly not high grade either."
4,B004,"Super crunchy cookies! My kids loved every single bite."
5,B005,"Horrible packaging. Bottle was shattered inside the box. Waste of money."
6,B006,"A bit too bland for my liking, though the ingredients are healthy."
7,B007,"Hands down the finest dark roast coffee I have ever brewed."
8,B008,"Disappointed with the texture. It felt like rubber and had no flavor."
9,B009,"Decent price for the quantity, though taste is quite plain."
10,B010,"Outstanding flavor and prompt delivery. Highly recommend to everyone!"`;

    const blob = new Blob([sampleCsvData], { type: 'text/csv' });
    const file = new File([blob], 'sample_food_reviews.csv', { type: 'text/csv' });
    setCsvFile(file);
    setCsvError(null);
  };

  // Calculated percentage metrics
  const positivePct = metrics.total > 0 ? ((metrics.positive / metrics.total) * 100).toFixed(1) : 0;
  const neutralPct = metrics.total > 0 ? ((metrics.neutral / metrics.total) * 100).toFixed(1) : 0;
  const negativePct = metrics.total > 0 ? ((metrics.negative / metrics.total) * 100).toFixed(1) : 0;

  // Filtered CSV table results
  const filteredTableResults = (csvResponse?.results_sample || []).filter(item => {
    const matchesSearch = item.original_text.toLowerCase().includes(tableSearch.toLowerCase()) ||
                          item.predicted_sentiment.toLowerCase().includes(tableSearch.toLowerCase());
    const matchesFilter = sentimentFilter === 'ALL' || item.predicted_sentiment === sentimentFilter;
    return matchesSearch && matchesFilter;
  });

  const pageSize = 10;
  const totalPages = Math.ceil(filteredTableResults.length / pageSize) || 1;
  const paginatedRows = filteredTableResults.slice((tablePage - 1) * pageSize, tablePage * pageSize);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Top Navbar */}
      <header style={{ 
        borderBottom: '1px solid var(--border-subtle)', 
        background: 'rgba(11, 15, 25, 0.85)', 
        backdropFilter: 'blur(12px)',
        position: 'sticky', 
        top: 0, 
        zIndex: 50 
      }}>
        <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '0.875rem 1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ 
              width: '2.5rem', 
              height: '2.5rem', 
              borderRadius: '0.625rem', 
              background: 'linear-gradient(135deg, #10b981 0%, #6366f1 100%)', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              boxShadow: '0 4px 12px rgba(16, 185, 129, 0.25)'
            }}>
              <BrainCircuit size={22} color="#ffffff" />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <h1 style={{ fontSize: '1.125rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
                  Fine-Food Sentiment Analyzer
                </h1>
                <span className="badge badge-primary" style={{ fontSize: '0.65rem' }}>OJT PROJECT</span>
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                TF-IDF & Logistic Regression • Amazon Fine Foods Dataset
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {/* Backend Status Pill */}
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem', 
              padding: '0.375rem 0.75rem', 
              borderRadius: '9999px',
              background: backendStatus === 'connected' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              border: `1px solid ${backendStatus === 'connected' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
              fontSize: '0.75rem',
              fontWeight: 600
            }}>
              <span style={{ 
                width: '8px', 
                height: '8px', 
                borderRadius: '50%', 
                backgroundColor: backendStatus === 'connected' ? 'var(--positive)' : 'var(--negative)',
                boxShadow: backendStatus === 'connected' ? '0 0 8px var(--positive)' : 'none'
              }} />
              <span style={{ color: backendStatus === 'connected' ? 'var(--positive)' : 'var(--negative)' }}>
                {backendStatus === 'connected' ? 'API Online (FastAPI)' : 'API Disconnected'}
              </span>
            </div>

            <button 
              onClick={checkHealth}
              title="Refresh connection"
              style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', display: 'flex' }}
            >
              <RefreshCw size={16} />
            </button>
          </div>

        </div>
      </header>

      {/* Main Content Container */}
      <main style={{ maxWidth: '1280px', margin: '0 auto', padding: '2rem 1.5rem', width: '100%', flex: 1 }}>

        {/* Top KPI Metric Cards */}
        <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
          
          {/* Card: Total */}
          <div className="glass-panel glass-panel-hover" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>TOTAL REVIEWS</span>
              <FileText size={18} color="var(--primary)" />
            </div>
            <div style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              {metrics.total.toLocaleString()}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              {metrics.source}
            </div>
          </div>

          {/* Card: Positive */}
          <div className="glass-panel glass-panel-hover" style={{ padding: '1.25rem', borderLeft: '3px solid var(--positive)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--positive)', fontWeight: 600 }}>POSITIVE (4–5 ★)</span>
              <ThumbsUp size={18} color="var(--positive)" />
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
              <div style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                {metrics.positive.toLocaleString()}
              </div>
              <span className="badge badge-positive">{positivePct}%</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Precision: 95.8% • Recall: 83.9%
            </div>
          </div>

          {/* Card: Neutral */}
          <div className="glass-panel glass-panel-hover" style={{ padding: '1.25rem', borderLeft: '3px solid var(--neutral)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--neutral)', fontWeight: 600 }}>NEUTRAL (3 ★)</span>
              <Minus size={18} color="var(--neutral)" />
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
              <div style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                {metrics.neutral.toLocaleString()}
              </div>
              <span className="badge badge-neutral">{neutralPct}%</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Balanced Recall: 54.0%
            </div>
          </div>

          {/* Card: Negative */}
          <div className="glass-panel glass-panel-hover" style={{ padding: '1.25rem', borderLeft: '3px solid var(--negative)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--negative)', fontWeight: 600 }}>NEGATIVE (1–2 ★)</span>
              <ThumbsDown size={18} color="var(--negative)" />
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
              <div style={{ fontSize: '1.875rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                {metrics.negative.toLocaleString()}
              </div>
              <span className="badge badge-negative">{negativePct}%</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Customer Complaints Detected: 75.3%
            </div>
          </div>

        </section>

        {/* Visual Analytics Bar */}
        <section className="glass-panel" style={{ padding: '1.25rem', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <BarChart3 size={18} color="var(--primary)" />
              <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>Sentiment Distribution Share</span>
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Model: Logistic Regression (C=1.0, class_weight='balanced')
            </span>
          </div>

          {/* Stacked Progress Bar */}
          <div style={{ 
            height: '1.25rem', 
            borderRadius: '0.625rem', 
            overflow: 'hidden', 
            display: 'flex', 
            background: 'rgba(0,0,0,0.3)',
            boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.5)' 
          }}>
            <div style={{ width: `${positivePct}%`, background: 'var(--positive)', transition: 'width 0.4s ease' }} title={`Positive: ${positivePct}%`} />
            <div style={{ width: `${neutralPct}%`, background: 'var(--neutral)', transition: 'width 0.4s ease' }} title={`Neutral: ${neutralPct}%`} />
            <div style={{ width: `${negativePct}%`, background: 'var(--negative)', transition: 'width 0.4s ease' }} title={`Negative: ${negativePct}%`} />
          </div>

          {/* Legend */}
          <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem', fontSize: '0.75rem' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: 'var(--positive)' }} />
              Positive ({positivePct}%)
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: 'var(--neutral)' }} />
              Neutral ({neutralPct}%)
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: 'var(--negative)' }} />
              Negative ({negativePct}%)
            </span>
          </div>
        </section>

        {/* Tab Navigation */}
        <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid var(--border-subtle)', marginBottom: '1.5rem' }}>
          
          <button 
            onClick={() => setActiveTab('playground')}
            style={{ 
              padding: '0.75rem 1.25rem', 
              background: 'transparent', 
              border: 'none', 
              borderBottom: activeTab === 'playground' ? '2px solid var(--primary)' : '2px solid transparent',
              color: activeTab === 'playground' ? 'var(--text-primary)' : 'var(--text-muted)',
              fontWeight: activeTab === 'playground' ? 700 : 500,
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s'
            }}
          >
            <Sparkles size={16} color={activeTab === 'playground' ? 'var(--primary)' : 'currentColor'} />
            Manual Review Tester
          </button>

          <button 
            onClick={() => setActiveTab('csv')}
            style={{ 
              padding: '0.75rem 1.25rem', 
              background: 'transparent', 
              border: 'none', 
              borderBottom: activeTab === 'csv' ? '2px solid var(--primary)' : '2px solid transparent',
              color: activeTab === 'csv' ? 'var(--text-primary)' : 'var(--text-muted)',
              fontWeight: activeTab === 'csv' ? 700 : 500,
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s'
            }}
          >
            <UploadCloud size={16} color={activeTab === 'csv' ? 'var(--primary)' : 'currentColor'} />
            CSV Dataset Upload & Analytics
          </button>

          <button 
            onClick={() => setActiveTab('methodology')}
            style={{ 
              padding: '0.75rem 1.25rem', 
              background: 'transparent', 
              border: 'none', 
              borderBottom: activeTab === 'methodology' ? '2px solid var(--primary)' : '2px solid transparent',
              color: activeTab === 'methodology' ? 'var(--text-primary)' : 'var(--text-muted)',
              fontWeight: activeTab === 'methodology' ? 700 : 500,
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s'
            }}
          >
            <ShieldCheck size={16} color={activeTab === 'methodology' ? 'var(--primary)' : 'currentColor'} />
            Model Architecture & Viva Guide
          </button>

        </div>

        {/* TAB 1: Single Review Prediction Playground */}
        {activeTab === 'playground' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
            
            {/* Input Column */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h2 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                Analyze Review Sentiment
              </h2>
              <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
                Type or paste any customer review. The NLP pipeline normalizes the text and applies the trained TF-IDF vectorizer and Logistic Regression classifier.
              </p>

              {/* Textarea */}
              <textarea 
                className="textarea-custom"
                value={reviewText}
                onChange={(e) => setReviewText(e.target.value)}
                placeholder="E.g., The dark chocolate almonds were fresh, crunchy, and absolutely delicious!"
              />

              {predictionError && (
                <div style={{ 
                  marginTop: '0.75rem', 
                  padding: '0.625rem 0.875rem', 
                  borderRadius: '0.5rem', 
                  background: 'var(--negative-bg)', 
                  border: '1px solid var(--negative-border)',
                  color: 'var(--negative)',
                  fontSize: '0.8rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  <AlertCircle size={16} />
                  {predictionError}
                </div>
              )}

              {/* Action Buttons */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {reviewText.length} characters • {reviewText.trim() ? reviewText.trim().split(/\s+/).length : 0} words
                </span>
                
                <button 
                  className="btn btn-primary"
                  onClick={() => handlePredict()}
                  disabled={isPredicting || !reviewText.trim()}
                >
                  {isPredicting ? (
                    <>
                      <RefreshCw size={16} className="animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Sparkles size={16} />
                      Predict Sentiment
                    </>
                  )}
                </button>
              </div>

              {/* Preset Chips */}
              <div style={{ marginTop: '1.5rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-subtle)' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, display: 'block', marginBottom: '0.625rem' }}>
                  QUICK EXAMPLES TO TEST:
                </span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                  {examples.map((ex, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setReviewText(ex.text);
                        handlePredict(ex.text);
                      }}
                      style={{
                        padding: '0.375rem 0.75rem',
                        borderRadius: '0.5rem',
                        fontSize: '0.75rem',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid var(--border-subtle)',
                        color: 'var(--text-secondary)',
                        cursor: 'pointer',
                        transition: 'all 0.15s'
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--primary)'}
                      onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--border-subtle)'}
                    >
                      {ex.label}
                    </button>
                  ))}
                </div>
              </div>

            </div>

            {/* Prediction Result Column */}
            <div>
              {predictionResult ? (
                <div className="glass-panel" style={{ padding: '1.5rem', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                  <div>
                    <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                      INFERENCE RESULT
                    </span>

                    {/* Sentiment Badge Banner */}
                    <div style={{ 
                      marginTop: '0.75rem', 
                      padding: '1.25rem', 
                      borderRadius: '0.75rem', 
                      background: predictionResult.sentiment === 'Positive' 
                        ? 'var(--positive-bg)' 
                        : (predictionResult.sentiment === 'Neutral' ? 'var(--neutral-bg)' : 'var(--negative-bg)'),
                      border: `1px solid ${
                        predictionResult.sentiment === 'Positive' 
                          ? 'var(--positive-border)' 
                          : (predictionResult.sentiment === 'Neutral' ? 'var(--neutral-border)' : 'var(--negative-border)')
                      }`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between'
                    }}>
                      <div>
                        <span style={{ 
                          fontSize: '0.8rem', 
                          fontWeight: 600, 
                          color: predictionResult.sentiment === 'Positive' 
                            ? 'var(--positive)' 
                            : (predictionResult.sentiment === 'Neutral' ? 'var(--neutral)' : 'var(--negative)') 
                        }}>
                          PREDICTED SENTIMENT
                        </span>
                        <div style={{ fontSize: '1.75rem', fontWeight: 800, marginTop: '0.25rem' }}>
                          {predictionResult.sentiment}
                        </div>
                      </div>

                      <div style={{ textAlign: 'right' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Confidence</span>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>
                          {(predictionResult.confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>

                    {/* Probability Breakdown Progress Bars */}
                    <div style={{ marginTop: '1.5rem' }}>
                      <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                        Class Probabilities
                      </span>

                      {/* Positive Bar */}
                      <div style={{ marginTop: '0.625rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                          <span style={{ color: 'var(--positive)', fontWeight: 600 }}>Positive</span>
                          <span>{(predictionResult.probabilities.Positive * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                          <div style={{ width: `${predictionResult.probabilities.Positive * 100}%`, height: '100%', background: 'var(--positive)', transition: 'width 0.4s ease' }} />
                        </div>
                      </div>

                      {/* Neutral Bar */}
                      <div style={{ marginTop: '0.625rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                          <span style={{ color: 'var(--neutral)', fontWeight: 600 }}>Neutral</span>
                          <span>{(predictionResult.probabilities.Neutral * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                          <div style={{ width: `${predictionResult.probabilities.Neutral * 100}%`, height: '100%', background: 'var(--neutral)', transition: 'width 0.4s ease' }} />
                        </div>
                      </div>

                      {/* Negative Bar */}
                      <div style={{ marginTop: '0.625rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                          <span style={{ color: 'var(--negative)', fontWeight: 600 }}>Negative</span>
                          <span>{(predictionResult.probabilities.Negative * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ height: '6px', borderRadius: '3px', background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                          <div style={{ width: `${predictionResult.probabilities.Negative * 100}%`, height: '100%', background: 'var(--negative)', transition: 'width 0.4s ease' }} />
                        </div>
                      </div>
                    </div>

                    {/* Cleaned text inspector */}
                    <div style={{ marginTop: '1.25rem', padding: '0.875rem', background: 'rgba(0,0,0,0.25)', borderRadius: '0.5rem', border: '1px solid var(--border-subtle)' }}>
                      <span style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                        NLP Normalized Text
                      </span>
                      <p style={{ fontSize: '0.785rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                        "{predictionResult.cleaned_text}"
                      </p>
                    </div>

                  </div>

                  <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    <span>Features: 25,000 TF-IDF N-grams</span>
                    <span>Status: Verified via FastAPI</span>
                  </div>

                </div>
              ) : (
                <div className="glass-panel" style={{ padding: '2.5rem 1.5rem', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                  <div style={{ width: '3.5rem', height: '3.5rem', borderRadius: '50%', background: 'rgba(255,255,255,0.03)', border: '1px dashed var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
                    <Sparkles size={24} color="var(--text-muted)" />
                  </div>
                  <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                    No Prediction Yet
                  </h3>
                  <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', maxWidth: '280px' }}>
                    Enter review text on the left and click Predict, or pick one of the quick test chips above.
                  </p>
                </div>
              )}
            </div>

          </div>
        )}

        {/* Session History (Playground Tab) */}
        {activeTab === 'playground' && sessionHistory.length > 0 && (
          <section className="glass-panel" style={{ marginTop: '1.5rem', padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>Recent Predictions in this Session</span>
              <button 
                onClick={() => setSessionHistory([])}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', fontSize: '0.75rem', cursor: 'pointer' }}
              >
                Clear History
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.75rem' }}>
              {sessionHistory.map(item => (
                <div 
                  key={item.id}
                  style={{ 
                    padding: '0.75rem', 
                    borderRadius: '0.5rem', 
                    background: 'rgba(255,255,255,0.02)', 
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between'
                  }}
                >
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                    "{item.text}"
                  </p>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem' }}>
                    <span className={`badge badge-${item.sentiment.toLowerCase()}`}>
                      {item.sentiment}
                    </span>
                    <span style={{ color: 'var(--text-muted)' }}>
                      {(item.confidence * 100).toFixed(0)}% • {item.timestamp}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* TAB 2: CSV Upload & Batch Analytics */}
        {activeTab === 'csv' && (
          <div>
            {/* Upload Box */}
            <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', marginBottom: '1.5rem' }}>
              <div style={{ maxWidth: '520px', margin: '0 auto' }}>
                <div style={{ 
                  width: '4rem', 
                  height: '4rem', 
                  borderRadius: '1rem', 
                  background: 'var(--primary-bg)', 
                  border: '1px solid var(--primary-border)',
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  margin: '0 auto 1.25rem' 
                }}>
                  <UploadCloud size={32} color="var(--primary)" />
                </div>

                <h2 style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: '0.5rem' }}>
                  Upload CSV for Batch Sentiment Analytics
                </h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
                  Upload any customer reviews CSV. The system auto-detects review columns (<code style={{ color: 'var(--primary)', fontFamily: 'var(--font-mono)' }}>Text</code>, <code style={{ color: 'var(--primary)', fontFamily: 'var(--font-mono)' }}>Review</code>, etc.), runs vectorized batch inference, and updates the entire dashboard.
                </p>

                {/* Hidden native input */}
                <input 
                  type="file" 
                  ref={fileInputRef}
                  accept=".csv"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />

                <div style={{ display: 'flex', justifyContent: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                  <button 
                    className="btn btn-secondary"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <FileText size={16} />
                    {csvFile ? csvFile.name : 'Choose CSV File...'}
                  </button>

                  <button 
                    className="btn btn-primary"
                    onClick={handleUploadCsv}
                    disabled={!csvFile || isUploading}
                  >
                    {isUploading ? (
                      <>
                        <RefreshCw size={16} className="animate-spin" />
                        Classifying Reviews...
                      </>
                    ) : (
                      <>
                        <Sparkles size={16} />
                        Run Batch Analysis
                      </>
                    )}
                  </button>

                  <button 
                    className="btn btn-secondary"
                    onClick={handleLoadSampleCsv}
                    title="Load an in-memory 10-row dataset for quick testing"
                  >
                    Load Demo Sample CSV
                  </button>
                </div>

                {csvError && (
                  <div style={{ 
                    marginTop: '1rem', 
                    padding: '0.75rem', 
                    borderRadius: '0.5rem', 
                    background: 'var(--negative-bg)', 
                    border: '1px solid var(--negative-border)',
                    color: 'var(--negative)',
                    fontSize: '0.8rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem'
                  }}>
                    <AlertCircle size={16} />
                    {csvError}
                  </div>
                )}
              </div>
            </div>

            {/* CSV Results Breakdown & Table */}
            {csvResponse && (
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
                  <div>
                    <h3 style={{ fontSize: '1.1rem', fontWeight: 800 }}>
                      Analyzed Dataset: {csvResponse.filename}
                    </h3>
                    <p style={{ fontSize: '0.775rem', color: 'var(--text-muted)' }}>
                      Detected text column: <strong style={{ color: 'var(--text-primary)' }}>{csvResponse.text_column_detected}</strong> • Total reviews: {csvResponse.total_reviews.toLocaleString()}
                    </p>
                  </div>

                  <button 
                    className="btn btn-secondary"
                    onClick={handleExportCsv}
                  >
                    <Download size={16} />
                    Export Full Analyzed CSV
                  </button>
                </div>

                {/* Table Filter Controls */}
                <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
                  <div style={{ position: 'relative', flex: 1, minWidth: '220px' }}>
                    <Search size={16} color="var(--text-muted)" style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)' }} />
                    <input 
                      type="text" 
                      value={tableSearch}
                      onChange={(e) => { setTableSearch(e.target.value); setTablePage(1); }}
                      placeholder="Search reviews or sentiment..."
                      style={{ 
                        width: '100%', 
                        padding: '0.5rem 0.75rem 0.5rem 2.25rem', 
                        background: 'rgba(0,0,0,0.3)', 
                        border: '1px solid var(--border-subtle)', 
                        borderRadius: '0.5rem', 
                        color: 'var(--text-primary)',
                        fontSize: '0.85rem'
                      }}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: '0.375rem' }}>
                    {['ALL', 'Positive', 'Neutral', 'Negative'].map(filter => (
                      <button
                        key={filter}
                        onClick={() => { setSentimentFilter(filter); setTablePage(1); }}
                        style={{
                          padding: '0.4rem 0.75rem',
                          borderRadius: '0.5rem',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          background: sentimentFilter === filter ? 'var(--primary)' : 'rgba(255,255,255,0.05)',
                          color: sentimentFilter === filter ? '#ffffff' : 'var(--text-secondary)',
                          border: '1px solid var(--border-subtle)',
                          cursor: 'pointer'
                        }}
                      >
                        {filter}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Reviews Preview Table */}
                <div style={{ overflowX: 'auto', border: '1px solid var(--border-subtle)', borderRadius: '0.75rem' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.825rem', textAlign: 'left' }}>
                    <thead>
                      <tr style={{ background: 'rgba(255,255,255,0.03)', borderBottom: '1px solid var(--border-subtle)' }}>
                        <th style={{ padding: '0.75rem 1rem', width: '60px', color: 'var(--text-muted)' }}>ID</th>
                        <th style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)' }}>REVIEW TEXT</th>
                        <th style={{ padding: '0.75rem 1rem', width: '130px', color: 'var(--text-muted)' }}>PREDICTION</th>
                        <th style={{ padding: '0.75rem 1rem', width: '110px', color: 'var(--text-muted)' }}>CONFIDENCE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedRows.length > 0 ? (
                        paginatedRows.map((row, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid var(--border-subtle)', transition: 'background 0.15s' }}>
                            <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                              {row.id}
                            </td>
                            <td style={{ padding: '0.75rem 1rem', color: 'var(--text-primary)', maxWidth: '480px' }}>
                              "{row.original_text}"
                            </td>
                            <td style={{ padding: '0.75rem 1rem' }}>
                              <span className={`badge badge-${row.predicted_sentiment.toLowerCase()}`}>
                                {row.predicted_sentiment}
                              </span>
                            </td>
                            <td style={{ padding: '0.75rem 1rem', fontWeight: 600 }}>
                              {(row.confidence * 100).toFixed(1)}%
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={4} style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                            No reviews match the selected filter.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Table Pagination */}
                {totalPages > 1 && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    <span>Showing {(tablePage - 1) * pageSize + 1} to {Math.min(tablePage * pageSize, filteredTableResults.length)} of {filteredTableResults.length} preview rows</span>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      <button 
                        className="btn btn-secondary"
                        style={{ padding: '0.25rem 0.625rem', fontSize: '0.75rem' }}
                        disabled={tablePage === 1}
                        onClick={() => setTablePage(p => Math.max(1, p - 1))}
                      >
                        Previous
                      </button>
                      <button 
                        className="btn btn-secondary"
                        style={{ padding: '0.25rem 0.625rem', fontSize: '0.75rem' }}
                        disabled={tablePage === totalPages}
                        onClick={() => setTablePage(p => Math.min(totalPages, p + 1))}
                      >
                        Next
                      </button>
                    </div>
                  </div>
                )}

              </div>
            )}
          </div>
        )}

        {/* TAB 3: Model Architecture & Viva Defense Reference */}
        {activeTab === 'methodology' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
            
            {/* Card 1: Pipeline Breakdown */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <BrainCircuit size={20} color="var(--primary)" />
                <h3 style={{ fontSize: '1.05rem', fontWeight: 800 }}>NLP & ML Pipeline</h3>
              </div>
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.875rem', fontSize: '0.825rem' }}>
                <li style={{ display: 'flex', gap: '0.75rem' }}>
                  <span style={{ color: 'var(--primary)', fontWeight: 700 }}>1.</span>
                  <div>
                    <strong>Data Anomaly Filtration:</strong> Stripped records where <code style={{ fontFamily: 'var(--font-mono)' }}>HelpfulnessNumerator &gt; HelpfulnessDenominator</code> (corrupted votes).
                  </div>
                </li>
                <li style={{ display: 'flex', gap: '0.75rem' }}>
                  <span style={{ color: 'var(--primary)', fontWeight: 700 }}>2.</span>
                  <div>
                    <strong>Deduplication:</strong> Eliminated 174,875 duplicate review texts (cross-posted product variations) to prevent train/test string memorization.
                  </div>
                </li>
                <li style={{ display: 'flex', gap: '0.75rem' }}>
                  <span style={{ color: 'var(--primary)', fontWeight: 700 }}>3.</span>
                  <div>
                    <strong>Negation Preservation:</strong> Customized English stopwords to retain critical negation cues (<code style={{ fontFamily: 'var(--font-mono)' }}>not</code>, <code style={{ fontFamily: 'var(--font-mono)' }}>no</code>, <code style={{ fontFamily: 'var(--font-mono)' }}>never</code>, <code style={{ fontFamily: 'var(--font-mono)' }}>cannot</code>).
                  </div>
                </li>
                <li style={{ display: 'flex', gap: '0.75rem' }}>
                  <span style={{ color: 'var(--primary)', fontWeight: 700 }}>4.</span>
                  <div>
                    <strong>Chronological Split:</strong> 80% train (1999–May 2012), 20% test (May 2012–Oct 2012) to realistically prevent future data leakage.
                  </div>
                </li>
                <li style={{ display: 'flex', gap: '0.75rem' }}>
                  <span style={{ color: 'var(--primary)', fontWeight: 700 }}>5.</span>
                  <div>
                    <strong>TF-IDF & Balanced Logistic Regression:</strong> Fitted 25,000 unigrams/bigrams with <code style={{ fontFamily: 'var(--font-mono)' }}>class_weight='balanced'</code> to overcome the 78% positive class skew.
                  </div>
                </li>
              </ul>
            </div>

            {/* Card 2: Confusion Matrix & Metrics */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <BarChart3 size={20} color="var(--positive)" />
                <h3 style={{ fontSize: '1.05rem', fontWeight: 800 }}>Evaluation Matrix</h3>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', textAlign: 'center', marginBottom: '1rem' }}>
                <div style={{ padding: '0.625rem', background: 'rgba(255,255,255,0.03)', borderRadius: '0.5rem' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>ACCURACY</div>
                  <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)' }}>80.16%</div>
                </div>
                <div style={{ padding: '0.625rem', background: 'rgba(99,102,241,0.1)', borderRadius: '0.5rem', border: '1px solid var(--primary-border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--primary)' }}>MACRO F1</div>
                  <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--primary)' }}>0.6560</div>
                </div>
                <div style={{ padding: '0.625rem', background: 'rgba(255,255,255,0.03)', borderRadius: '0.5rem' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>WEIGHTED F1</div>
                  <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)' }}>0.8246</div>
                </div>
              </div>

              <div style={{ fontSize: '0.785rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <p>
                  • <strong>Positive Recall (83.9%):</strong> High confidence in detecting satisfied reviews.
                </p>
                <p>
                  • <strong>Negative Recall (75.3%):</strong> Essential e-commerce capability to isolate customer churn and product defects.
                </p>
                <p>
                  • <strong>Neutral Recall (54.0%):</strong> Balanced weights prevented minority neutral reviews from collapsing to near zero.
                </p>
              </div>
            </div>

            {/* Card 3: Top N-Gram Drivers */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                <Sparkles size={20} color="var(--neutral)" />
                <h3 style={{ fontSize: '1.05rem', fontWeight: 800 }}>Learned Semantic Drivers</h3>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem', fontSize: '0.8rem' }}>
                <div>
                  <span className="badge badge-positive" style={{ marginBottom: '0.375rem' }}>Positive Drivers</span>
                  <div style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    great, best, delicious, love, excellent, perfect, favorite, highly recommend, awesome
                  </div>
                </div>

                <div>
                  <span className="badge badge-neutral" style={{ marginBottom: '0.375rem' }}>Neutral Drivers</span>
                  <div style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    okay, average, decent but, nothing special, alright, wanted love, fair, ok
                  </div>
                </div>

                <div>
                  <span className="badge badge-negative" style={{ marginBottom: '0.375rem' }}>Negative Drivers</span>
                  <div style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                    worst, disappointed, terrible, horrible, awful, stale, bad, waste money, threw away
                  </div>
                </div>
              </div>
            </div>

          </div>
        )}

      </main>

      {/* Footer */}
      <footer style={{ 
        borderTop: '1px solid var(--border-subtle)', 
        padding: '1.25rem 1.5rem', 
        fontSize: '0.75rem', 
        color: 'var(--text-muted)', 
        textAlign: 'center',
        background: 'rgba(11, 15, 25, 0.9)'
      }}>
        Fine-Food Review Sentiment Analyzer • OJT Machine Learning Project • Built with FastAPI & React
      </footer>

    </div>
  );
}
