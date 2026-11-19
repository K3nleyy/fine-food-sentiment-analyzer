import React, { useState, useEffect, useRef } from 'react';
import { 
  Plus, 
  Upload, 
  Download, 
  Search, 
  Star, 
  TrendingUp, 
  AlertCircle, 
  CheckCircle2, 
  X, 
  MessageSquare, 
  RefreshCw,
  Sparkles,
  Inbox,
  ArrowUpDown,
  Filter,
  Layers,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://127.0.0.1:8000'
  : '/api';

export default function App() {
  // Main dataset summary state
  const [summary, setSummary] = useState({
    has_data: false,
    total_reviews: 0,
    sentiment_distribution: { Positive: 0, Neutral: 0, Negative: 0 },
    sentiment_percentages: { Positive: 0.0, Neutral: 0.0, Negative: 0.0 },
    rating_distribution: { '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, unrated: 0 },
    timeline: [],
    positive_themes: [],
    negative_themes: [],
    last_updated: null
  });

  const [isLoadingSummary, setIsLoadingSummary] = useState(true);

  // Reviews list table state
  const [reviewsData, setReviewsData] = useState({ total: 0, page: 1, pages: 1, items: [] });
  const [searchQuery, setSearchQuery] = useState('');
  const [sentimentFilter, setSentimentFilter] = useState('ALL');
  const [ratingFilter, setRatingFilter] = useState('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const [isLoadingReviews, setIsLoadingReviews] = useState(false);

  // Modals state
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);

  // Add review form state
  const [newReviewText, setNewReviewText] = useState('');
  const [newRating, setNewRating] = useState(0); // 0 = unrated, 1-5
  const [newCustomerName, setNewCustomerName] = useState('');
  const [newProductName, setNewProductName] = useState('');
  const [isSubmittingReview, setIsSubmittingReview] = useState(false);
  const [addReviewError, setAddReviewError] = useState(null);

  // Upload CSV state
  const [csvFile, setCsvFile] = useState(null);
  const [uploadMode, setUploadMode] = useState('replace'); // 'replace' | 'append'
  const [isUploadingCsv, setIsUploadingCsv] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);

  // Notification toast
  const [toastMessage, setToastMessage] = useState(null);

  const showToast = (msg) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  };

  // Fetch summary and reviews on mount
  useEffect(() => {
    fetchSummary();
  }, []);

  useEffect(() => {
    fetchReviews(currentPage, searchQuery, sentimentFilter, ratingFilter);
  }, [currentPage, searchQuery, sentimentFilter, ratingFilter]);

  // Fetch summary analytics
  const fetchSummary = async () => {
    setIsLoadingSummary(true);
    try {
      const res = await fetch(`${API_BASE}/api/dataset/summary`);
      if (res.ok) {
        const data = await res.json();
        setSummary(data);
      }
    } catch (err) {
      console.error('Failed to fetch dataset summary:', err);
    } finally {
      setIsLoadingSummary(false);
    }
  };

  // Fetch reviews table
  const fetchReviews = async (page = 1, search = '', sentiment = 'ALL', rating = 'ALL') => {
    setIsLoadingReviews(true);
    try {
      const params = new URLSearchParams({
        page: String(page),
        limit: '15',
        search: search.trim(),
        sentiment,
        rating
      });
      const res = await fetch(`${API_BASE}/api/reviews?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setReviewsData(data);
      }
    } catch (err) {
      console.error('Failed to fetch reviews:', err);
    } finally {
      setIsLoadingReviews(false);
    }
  };

  // Submit new review
  const handleAddReviewSubmit = async (e) => {
    e.preventDefault();
    const text = newReviewText.trim();
    if (!text) {
      setAddReviewError('Review text is required.');
      return;
    }

    setIsSubmittingReview(true);
    setAddReviewError(null);

    try {
      const payload = {
        text,
        rating: newRating > 0 ? newRating : null,
        customer_name: newCustomerName.trim() || null,
        product_name: newProductName.trim() || null
      };

      const res = await fetch(`${API_BASE}/api/reviews`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to submit review');
      }

      const resData = await res.json();
      
      // Automatic dashboard update with backend confirmed data
      setSummary(resData.summary);
      
      // Refresh reviews list
      fetchReviews(1, searchQuery, sentimentFilter, ratingFilter);
      setCurrentPage(1);

      // Reset form & close modal
      setNewReviewText('');
      setNewRating(0);
      setNewCustomerName('');
      setNewProductName('');
      setIsAddModalOpen(false);

      showToast(`Review added • Analyzed as ${resData.review.predicted_sentiment} (${(resData.review.confidence * 100).toFixed(0)}% confidence)`);

    } catch (err) {
      setAddReviewError(err.message || 'An error occurred while saving the review.');
    } finally {
      setIsSubmittingReview(false);
    }
  };

  // Upload CSV dataset
  const handleCsvUploadSubmit = async () => {
    if (!csvFile) {
      setUploadError('Please select a CSV file.');
      return;
    }

    setIsUploadingCsv(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append('file', csvFile);
    formData.append('mode', uploadMode);

    try {
      const res = await fetch(`${API_BASE}/api/upload-csv`, {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to process CSV file.');
      }

      const resData = await res.json();
      setSummary(resData.summary);
      fetchReviews(1, '', 'ALL', 'ALL');
      setCurrentPage(1);
      setSearchQuery('');
      setSentimentFilter('ALL');
      setRatingFilter('ALL');

      setIsUploadModalOpen(false);
      setCsvFile(null);
      showToast(`Successfully processed ${resData.total_reviews.toLocaleString()} reviews (${uploadMode === 'replace' ? 'replaced active dataset' : 'added to active dataset'})`);

    } catch (err) {
      setUploadError(err.message || 'Failed to analyze CSV.');
    } finally {
      setIsUploadingCsv(false);
    }
  };

  // Load demo sample dataset
  const handleLoadDemoDataset = async () => {
    setIsLoadingSummary(true);
    try {
      const res = await fetch(`${API_BASE}/api/dataset/sample`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setSummary(data);
        fetchReviews(1, '', 'ALL', 'ALL');
        setIsUploadModalOpen(false);
        showToast('Sample dataset loaded with 25 customer reviews');
      }
    } catch (err) {
      showToast('Failed to load sample dataset');
    } finally {
      setIsLoadingSummary(false);
    }
  };

  // Clear dataset
  const handleClearDataset = async () => {
    if (!window.confirm('Are you sure you want to clear the active dataset?')) return;
    try {
      const res = await fetch(`${API_BASE}/api/dataset/clear`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setSummary(data);
        fetchReviews(1, '', 'ALL', 'ALL');
        showToast('Active review dataset cleared');
      }
    } catch (err) {
      showToast('Failed to clear dataset');
    }
  };

  // Export CSV
  const handleExportCsv = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/dataset/export`);
      if (!res.ok) throw new Error('No data to export.');
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `fine_food_insights_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      showToast('Export downloaded successfully');
    } catch (err) {
      alert('Could not export dataset: ' + err.message);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* Toast Notification */}
      {toastMessage && (
        <div style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          background: 'var(--bg-surface-elevated)',
          border: '1px solid var(--border-hover)',
          padding: '0.75rem 1.25rem',
          borderRadius: 'var(--radius-md)',
          color: 'var(--text-primary)',
          fontSize: '0.85rem',
          boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.625rem',
          zIndex: 200,
          animation: 'modalFadeIn 0.2s ease-out'
        }}>
          <CheckCircle2 size={16} color="var(--positive)" />
          {toastMessage}
        </div>
      )}

      {/* Clean Modern Navigation Bar */}
      <header style={{ 
        borderBottom: '1px solid var(--border-subtle)', 
        background: 'rgba(12, 13, 14, 0.85)', 
        backdropFilter: 'blur(8px)',
        position: 'sticky', 
        top: 0, 
        zIndex: 40 
      }}>
        <div style={{ 
          maxWidth: '1240px', 
          margin: '0 auto', 
          padding: '0.875rem 1.5rem', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between' 
        }}>
          
          {/* Logo & Product Name */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
            <div style={{ 
              width: '28px', 
              height: '28px', 
              borderRadius: '7px', 
              background: '#ffffff', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              color: '#0c0d0e'
            }}>
              <MessageSquare size={16} strokeWidth={2.5} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
                <span style={{ fontSize: '0.95rem', fontWeight: 700, letterSpacing: '-0.015em', color: 'var(--text-primary)' }}>
                  Fine-Food Insights
                </span>
                {summary.has_data && (
                  <span className="badge badge-subtle">
                    {summary.total_reviews.toLocaleString()} reviews
                  </span>
                )}
              </div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Understand your customer feedback.
              </p>
            </div>
          </div>

          {/* Action Header Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
            
            {summary.has_data && (
              <button 
                className="btn btn-ghost" 
                onClick={handleExportCsv}
                title="Export active dataset to CSV"
                style={{ padding: '0.45rem 0.75rem', fontSize: '0.8rem' }}
              >
                <Download size={14} />
                Export
              </button>
            )}

            <button 
              className="btn btn-secondary"
              onClick={() => { setUploadError(null); setIsUploadModalOpen(true); }}
              style={{ fontSize: '0.825rem' }}
            >
              <Upload size={14} />
              Upload CSV
            </button>

            <button 
              className="btn btn-primary"
              onClick={() => { setAddReviewError(null); setIsAddModalOpen(true); }}
              style={{ fontSize: '0.825rem' }}
            >
              <Plus size={14} strokeWidth={2.5} />
              Add New Review
            </button>

          </div>

        </div>
      </header>

      {/* Main Container */}
      <main style={{ maxWidth: '1240px', margin: '0 auto', padding: '2rem 1.5rem', width: '100%', flex: 1 }}>

        {/* EMPTY STATE (When no data has been uploaded) */}
        {!summary.has_data && !isLoadingSummary ? (
          <div style={{ 
            maxWidth: '560px', 
            margin: '4rem auto', 
            textAlign: 'center', 
            padding: '3rem 2rem' 
          }} className="surface-card">
            
            <div style={{ 
              width: '48px', 
              height: '48px', 
              borderRadius: '12px', 
              background: 'var(--bg-surface-elevated)', 
              border: '1px solid var(--border-subtle)',
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              margin: '0 auto 1.25rem',
              color: 'var(--text-secondary)'
            }}>
              <Inbox size={22} />
            </div>

            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem', color: 'var(--text-primary)' }}>
              No Customer Reviews Analyzed Yet
            </h2>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '1.75rem' }}>
              Upload your customer feedback CSV or add individual reviews to generate real-time sentiment distribution, customer satisfaction metrics, and complaint themes.
            </p>

            <div style={{ display: 'flex', justifyContent: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
              <button 
                className="btn btn-primary"
                onClick={() => { setUploadError(null); setIsUploadModalOpen(true); }}
              >
                <Upload size={14} />
                Upload Review CSV
              </button>

              <button 
                className="btn btn-secondary"
                onClick={handleLoadDemoDataset}
              >
                <Sparkles size={14} />
                Load Demo Reviews (25 items)
              </button>

              <button 
                className="btn btn-ghost"
                onClick={() => { setAddReviewError(null); setIsAddModalOpen(true); }}
              >
                <Plus size={14} />
                Add Review Manually
              </button>
            </div>

          </div>
        ) : (
          <>
            {/* OVERVIEW SECTION: 4 Clean Metric Cards */}
            <section style={{ marginBottom: '1.5rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
                
                {/* Total Reviews */}
                <div className="surface-card" style={{ padding: '1.25rem' }}>
                  <div style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', fontWeight: 500, marginBottom: '0.375rem' }}>
                    Total Reviews
                  </div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                    {summary.total_reviews.toLocaleString()}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.375rem' }}>
                    {summary.last_updated ? `Last entry: ${summary.last_updated.slice(0, 16)}` : 'Active dataset'}
                  </div>
                </div>

                {/* Positive Reviews */}
                <div className="surface-card" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.375rem' }}>
                    <span style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Positive Feedback</span>
                    <span className="badge badge-positive">{summary.sentiment_percentages.Positive}%</span>
                  </div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                    {summary.sentiment_distribution.Positive.toLocaleString()}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.375rem' }}>
                    Satisfied customer experiences
                  </div>
                </div>

                {/* Neutral Reviews */}
                <div className="surface-card" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.375rem' }}>
                    <span style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Neutral Reviews</span>
                    <span className="badge badge-neutral">{summary.sentiment_percentages.Neutral}%</span>
                  </div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                    {summary.sentiment_distribution.Neutral.toLocaleString()}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.375rem' }}>
                    Mixed feedback & balanced opinions
                  </div>
                </div>

                {/* Negative Reviews */}
                <div className="surface-card" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.375rem' }}>
                    <span style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Customer Complaints</span>
                    <span className="badge badge-negative">{summary.sentiment_percentages.Negative}%</span>
                  </div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
                    {summary.sentiment_distribution.Negative.toLocaleString()}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.375rem' }}>
                    Issues requiring attention
                  </div>
                </div>

              </div>
            </section>

            {/* SENTIMENT OVERVIEW PROPORTIONAL BAR */}
            <section className="surface-card" style={{ padding: '1.25rem', marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Customer Sentiment Overview
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {summary.total_reviews} total feedback items
                </span>
              </div>

              {/* Segmented bar */}
              <div style={{ 
                height: '10px', 
                borderRadius: '5px', 
                overflow: 'hidden', 
                display: 'flex', 
                background: 'var(--bg-input)' 
              }}>
                <div style={{ width: `${summary.sentiment_percentages.Positive}%`, background: 'var(--positive)', transition: 'width 0.3s ease' }} />
                <div style={{ width: `${summary.sentiment_percentages.Neutral}%`, background: 'var(--neutral)', transition: 'width 0.3s ease' }} />
                <div style={{ width: `${summary.sentiment_percentages.Negative}%`, background: 'var(--negative)', transition: 'width 0.3s ease' }} />
              </div>

              {/* Clean Legend */}
              <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.875rem', fontSize: '0.775rem' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--text-secondary)' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--positive)' }} />
                  Positive ({summary.sentiment_percentages.Positive}% • {summary.sentiment_distribution.Positive})
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--text-secondary)' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--neutral)' }} />
                  Neutral ({summary.sentiment_percentages.Neutral}% • {summary.sentiment_distribution.Neutral})
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: 'var(--text-secondary)' }}>
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--negative)' }} />
                  Negative ({summary.sentiment_percentages.Negative}% • {summary.sentiment_distribution.Negative})
                </span>
              </div>
            </section>

            {/* CUSTOMER INSIGHTS: Rating Distribution & Feedback Themes */}
            <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
              
              {/* Rating Distribution Card */}
              <div className="surface-card" style={{ padding: '1.25rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)', display: 'block', marginBottom: '1rem' }}>
                  Customer Ratings Breakdown
                </span>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {[5, 4, 3, 2, 1].map(stars => {
                    const count = summary.rating_distribution[String(stars)] || 0;
                    const pct = summary.total_reviews > 0 ? (count / summary.total_reviews * 100).toFixed(0) : 0;
                    return (
                      <div key={stars} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.775rem' }}>
                        <span style={{ width: '38px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                          {stars} <Star size={11} fill="#eab308" color="#eab308" />
                        </span>
                        <div style={{ flex: 1, height: '6px', background: 'var(--bg-input)', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{ 
                            width: `${pct}%`, 
                            height: '100%', 
                            background: stars >= 4 ? 'var(--positive)' : (stars === 3 ? 'var(--neutral)' : 'var(--negative)') 
                          }} />
                        </div>
                        <span style={{ width: '42px', textAlign: 'right', color: 'var(--text-muted)' }}>
                          {count}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Feedback Themes Card */}
              <div className="surface-card" style={{ padding: '1.25rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)', display: 'block', marginBottom: '1rem' }}>
                  Key Customer Feedback Themes
                </span>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  {/* Positive Mentions */}
                  <div>
                    <span style={{ fontSize: '0.725rem', fontWeight: 600, color: 'var(--positive)', textTransform: 'uppercase', display: 'block', marginBottom: '0.5rem' }}>
                      What Customers Love
                    </span>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                      {summary.positive_themes.length > 0 ? (
                        summary.positive_themes.map((item, idx) => (
                          <div key={idx} style={{ 
                            fontSize: '0.75rem', 
                            padding: '0.3rem 0.5rem', 
                            background: 'var(--positive-bg)', 
                            border: '1px solid var(--positive-border)', 
                            borderRadius: '4px',
                            display: 'flex',
                            justifyContent: 'space-between',
                            color: 'var(--positive)'
                          }}>
                            <span>{item.theme}</span>
                            <span style={{ opacity: 0.8 }}>{item.mentions}</span>
                          </div>
                        ))
                      ) : (
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>No positive themes yet.</span>
                      )}
                    </div>
                  </div>

                  {/* Negative Complaints */}
                  <div>
                    <span style={{ fontSize: '0.725rem', fontWeight: 600, color: 'var(--negative)', textTransform: 'uppercase', display: 'block', marginBottom: '0.5rem' }}>
                      Common Complaints
                    </span>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                      {summary.negative_themes.length > 0 ? (
                        summary.negative_themes.map((item, idx) => (
                          <div key={idx} style={{ 
                            fontSize: '0.75rem', 
                            padding: '0.3rem 0.5rem', 
                            background: 'var(--negative-bg)', 
                            border: '1px solid var(--negative-border)', 
                            borderRadius: '4px',
                            display: 'flex',
                            justifyContent: 'space-between',
                            color: 'var(--negative)'
                          }}>
                            <span>{item.theme}</span>
                            <span style={{ opacity: 0.8 }}>{item.mentions}</span>
                          </div>
                        ))
                      ) : (
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>No complaints detected.</span>
                      )}
                    </div>
                  </div>
                </div>

              </div>

            </section>

            {/* CUSTOMER REVIEWS EXPLORER TABLE */}
            <section className="surface-card" style={{ padding: '1.25rem' }}>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    Customer Reviews
                  </h3>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Showing {reviewsData.total} feedback records
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <button 
                    className="btn btn-ghost" 
                    onClick={() => fetchReviews(currentPage, searchQuery, sentimentFilter, ratingFilter)}
                    title="Refresh reviews"
                    style={{ padding: '0.35rem', color: 'var(--text-muted)' }}
                  >
                    <RefreshCw size={14} className={isLoadingReviews ? 'animate-spin' : ''} />
                  </button>
                  <button 
                    className="btn btn-ghost" 
                    onClick={handleClearDataset}
                    style={{ padding: '0.35rem 0.6rem', fontSize: '0.75rem', color: 'var(--negative)' }}
                  >
                    Reset Dataset
                  </button>
                </div>
              </div>

              {/* Search & Filter Toolbar */}
              <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
                
                {/* Search Box */}
                <div style={{ position: 'relative', flex: 1, minWidth: '220px' }}>
                  <Search size={14} color="var(--text-muted)" style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)' }} />
                  <input 
                    type="text"
                    value={searchQuery}
                    onChange={(e) => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                    placeholder="Search reviews, products, customers..."
                    className="input-text"
                    style={{ paddingLeft: '2.1rem', fontSize: '0.8rem' }}
                  />
                </div>

                {/* Sentiment Filter Tabs */}
                <div style={{ display: 'flex', background: 'var(--bg-input)', padding: '2px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  {['ALL', 'Positive', 'Neutral', 'Negative'].map(val => (
                    <button
                      key={val}
                      onClick={() => { setSentimentFilter(val); setCurrentPage(1); }}
                      style={{
                        padding: '0.35rem 0.75rem',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '0.75rem',
                        fontWeight: sentimentFilter === val ? 600 : 500,
                        background: sentimentFilter === val ? 'var(--bg-surface-elevated)' : 'transparent',
                        color: sentimentFilter === val ? 'var(--text-primary)' : 'var(--text-muted)',
                        border: 'none',
                        cursor: 'pointer'
                      }}
                    >
                      {val}
                    </button>
                  ))}
                </div>

                {/* Star Filter Dropdown */}
                <select
                  value={ratingFilter}
                  onChange={(e) => { setRatingFilter(e.target.value); setCurrentPage(1); }}
                  className="input-text"
                  style={{ width: 'auto', paddingRight: '1.5rem', fontSize: '0.8rem' }}
                >
                  <option value="ALL">All Ratings</option>
                  <option value="5">5 Stars</option>
                  <option value="4">4 Stars</option>
                  <option value="3">3 Stars</option>
                  <option value="2">2 Stars</option>
                  <option value="1">1 Star</option>
                </select>

              </div>

              {/* Table */}
              <div style={{ overflowX: 'auto', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ background: 'var(--bg-input)', borderBottom: '1px solid var(--border-subtle)' }}>
                      <th style={{ padding: '0.625rem 0.875rem', color: 'var(--text-secondary)', fontWeight: 600, width: '110px' }}>Rating</th>
                      <th style={{ padding: '0.625rem 0.875rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Review Feedback</th>
                      <th style={{ padding: '0.625rem 0.875rem', color: 'var(--text-secondary)', fontWeight: 600, width: '160px' }}>Product</th>
                      <th style={{ padding: '0.625rem 0.875rem', color: 'var(--text-secondary)', fontWeight: 600, width: '120px' }}>Sentiment</th>
                      <th style={{ padding: '0.625rem 0.875rem', color: 'var(--text-secondary)', fontWeight: 600, width: '85px', textAlign: 'right' }}>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reviewsData.items.length > 0 ? (
                      reviewsData.items.map((row) => (
                        <tr key={row.id} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                          
                          {/* Rating Stars */}
                          <td style={{ padding: '0.75rem 0.875rem', whiteSpace: 'nowrap' }}>
                            {row.rating ? (
                              <div style={{ display: 'flex', gap: '2px', alignItems: 'center' }}>
                                {[...Array(5)].map((_, i) => (
                                  <Star 
                                    key={i} 
                                    size={12} 
                                    fill={i < row.rating ? '#eab308' : 'none'} 
                                    color={i < row.rating ? '#eab308' : '#3f3f46'} 
                                  />
                                ))}
                              </div>
                            ) : (
                              <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Unrated</span>
                            )}
                          </td>

                          {/* Review Text */}
                          <td style={{ padding: '0.75rem 0.875rem', color: 'var(--text-primary)', maxWidth: '420px' }}>
                            <p style={{ margin: 0, lineHeight: 1.45 }}>
                              "{row.text}"
                            </p>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', display: 'block', marginTop: '0.25rem' }}>
                              by {row.customer_name} • {row.created_at ? row.created_at.slice(0, 10) : ''}
                            </span>
                          </td>

                          {/* Product Name */}
                          <td style={{ padding: '0.75rem 0.875rem', color: 'var(--text-secondary)' }}>
                            <span style={{ display: 'block', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', maxWidth: '140px' }} title={row.product_name}>
                              {row.product_name || 'Food Product'}
                            </span>
                          </td>

                          {/* Sentiment Result */}
                          <td style={{ padding: '0.75rem 0.875rem' }}>
                            <span className={`badge badge-${row.predicted_sentiment.toLowerCase()}`}>
                              {row.predicted_sentiment}
                            </span>
                          </td>

                          {/* Confidence */}
                          <td style={{ padding: '0.75rem 0.875rem', textAlign: 'right', fontWeight: 600, color: 'var(--text-secondary)' }}>
                            {(row.confidence * 100).toFixed(0)}%
                          </td>

                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={5} style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                          {isLoadingReviews ? 'Loading reviews...' : 'No customer reviews match your search or filter.'}
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {reviewsData.pages > 1 && (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  <span>Page {reviewsData.page} of {reviewsData.pages}</span>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button 
                      className="btn btn-secondary" 
                      style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                      disabled={currentPage <= 1}
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    >
                      <ChevronLeft size={14} />
                      Previous
                    </button>
                    <button 
                      className="btn btn-secondary" 
                      style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                      disabled={currentPage >= reviewsData.pages}
                      onClick={() => setCurrentPage(p => Math.min(reviewsData.pages, p + 1))}
                    >
                      Next
                      <ChevronRight size={14} />
                    </button>
                  </div>
                </div>
              )}

            </section>
          </>
        )}

      </main>

      {/* FOOTER */}
      <footer style={{ 
        borderTop: '1px solid var(--border-subtle)', 
        padding: '1.25rem 1.5rem', 
        fontSize: '0.75rem', 
        color: 'var(--text-muted)', 
        textAlign: 'center',
        background: 'var(--bg-app)'
      }}>
        Fine-Food Insights • Natural Language Customer Feedback Intelligence
      </footer>

      {/* ========================================================= */}
      {/* MODAL 1: ADD NEW REVIEW FORM                               */}
      {/* ========================================================= */}
      {isAddModalOpen && (
        <div className="modal-backdrop" onClick={() => !isSubmittingReview && setIsAddModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            
            <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Add Customer Review
                </h3>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Review sentiment is predicted using the trained machine learning model and saved to the active dataset.
                </p>
              </div>
              <button 
                onClick={() => setIsAddModalOpen(false)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
                disabled={isSubmittingReview}
              >
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleAddReviewSubmit} style={{ padding: '1.5rem' }}>
              
              {/* Review Text (Required) */}
              <div style={{ marginBottom: '1.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.375rem' }}>
                  <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Customer Feedback <span style={{ color: 'var(--negative)' }}>*</span>
                  </label>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    {newReviewText.length} chars
                  </span>
                </div>
                <textarea 
                  className="textarea-input"
                  value={newReviewText}
                  onChange={(e) => setNewReviewText(e.target.value)}
                  placeholder="Enter the customer review text (e.g. 'The dark roast coffee was rich and smooth, highly recommend!')..."
                  rows={4}
                  required
                />
              </div>

              {/* Star Rating (1-5 Stars) */}
              <div style={{ marginBottom: '1.25rem' }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', display: 'block', marginBottom: '0.375rem' }}>
                  Customer Rating (Optional)
                </label>
                <div style={{ display: 'flex', gap: '0.375rem', alignItems: 'center' }}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      type="button"
                      onClick={() => setNewRating(star === newRating ? 0 : star)}
                      style={{ 
                        background: 'transparent', 
                        border: 'none', 
                        cursor: 'pointer',
                        padding: '2px',
                        display: 'flex'
                      }}
                    >
                      <Star 
                        size={22} 
                        fill={star <= newRating ? '#eab308' : 'none'} 
                        color={star <= newRating ? '#eab308' : '#52525b'} 
                      />
                    </button>
                  ))}
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
                    {newRating > 0 ? `${newRating} of 5 Stars` : 'Click to select rating'}
                  </span>
                </div>
              </div>

              {/* Customer Name & Product Name Inputs */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1.5rem' }}>
                <div>
                  <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', display: 'block', marginBottom: '0.375rem' }}>
                    Customer Name (Optional)
                  </label>
                  <input 
                    type="text" 
                    value={newCustomerName}
                    onChange={(e) => setNewCustomerName(e.target.value)}
                    placeholder="e.g. Jane Doe"
                    className="input-text"
                  />
                </div>

                <div>
                  <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', display: 'block', marginBottom: '0.375rem' }}>
                    Product Name (Optional)
                  </label>
                  <input 
                    type="text" 
                    value={newProductName}
                    onChange={(e) => setNewProductName(e.target.value)}
                    placeholder="e.g. Organic Honey"
                    className="input-text"
                  />
                </div>
              </div>

              {addReviewError && (
                <div style={{ 
                  marginBottom: '1rem', 
                  padding: '0.625rem 0.875rem', 
                  borderRadius: 'var(--radius-sm)', 
                  background: 'var(--negative-bg)', 
                  border: '1px solid var(--negative-border)', 
                  color: 'var(--negative)', 
                  fontSize: '0.8rem' 
                }}>
                  {addReviewError}
                </div>
              )}

              {/* Form Buttons */}
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.625rem' }}>
                <button 
                  type="button" 
                  className="btn btn-secondary"
                  onClick={() => setIsAddModalOpen(false)}
                  disabled={isSubmittingReview}
                >
                  Cancel
                </button>
                <button 
                  type="submit" 
                  className="btn btn-primary"
                  disabled={isSubmittingReview || !newReviewText.trim()}
                >
                  {isSubmittingReview ? (
                    <>
                      <RefreshCw size={14} className="animate-spin" />
                      Analyzing & Saving...
                    </>
                  ) : (
                    'Submit Review'
                  )}
                </button>
              </div>

            </form>

          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* MODAL 2: UPLOAD CSV DATASET                                */}
      {/* ========================================================= */}
      {isUploadModalOpen && (
        <div className="modal-backdrop" onClick={() => !isUploadingCsv && setIsUploadModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            
            <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid var(--border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  Upload Customer Reviews (CSV)
                </h3>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Batch-analyze sentiment for your customer review records.
                </p>
              </div>
              <button 
                onClick={() => setIsUploadModalOpen(false)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}
                disabled={isUploadingCsv}
              >
                <X size={18} />
              </button>
            </div>

            <div style={{ padding: '1.5rem' }}>
              
              {/* File Picker Zone */}
              <input 
                type="file"
                ref={fileInputRef}
                accept=".csv"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setCsvFile(e.target.files[0]);
                    setUploadError(null);
                  }
                }}
                style={{ display: 'none' }}
              />

              <div 
                onClick={() => fileInputRef.current?.click()}
                style={{ 
                  border: '1px dashed var(--border-hover)', 
                  borderRadius: 'var(--radius-md)', 
                  padding: '2rem 1rem', 
                  textAlign: 'center', 
                  cursor: 'pointer',
                  background: 'var(--bg-input)',
                  marginBottom: '1.25rem',
                  transition: 'border-color 0.15s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--text-secondary)'}
                onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--border-hover)'}
              >
                <div style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'var(--bg-surface-elevated)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 0.75rem', color: 'var(--text-secondary)' }}>
                  <Upload size={18} />
                </div>
                <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                  {csvFile ? csvFile.name : 'Click to choose a .csv file'}
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Auto-detects review text, rating, and customer name columns.
                </p>
              </div>

              {/* Upload Mode Choice (Replace vs Append) */}
              <div style={{ marginBottom: '1.5rem' }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', display: 'block', marginBottom: '0.5rem' }}>
                  Dataset Action:
                </label>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <label style={{ 
                    display: 'flex', 
                    alignItems: 'flex-start', 
                    gap: '0.625rem', 
                    padding: '0.625rem 0.75rem', 
                    borderRadius: 'var(--radius-sm)', 
                    background: uploadMode === 'replace' ? 'var(--bg-surface-elevated)' : 'transparent',
                    border: '1px solid var(--border-subtle)',
                    cursor: 'pointer' 
                  }}>
                    <input 
                      type="radio" 
                      name="uploadMode" 
                      value="replace" 
                      checked={uploadMode === 'replace'} 
                      onChange={() => setUploadMode('replace')}
                      style={{ marginTop: '3px' }}
                    />
                    <div>
                      <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', display: 'block' }}>
                        Replace active dataset
                      </span>
                      <span style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>
                        Clear existing reviews and analyze this file as the active dataset.
                      </span>
                    </div>
                  </label>

                  <label style={{ 
                    display: 'flex', 
                    alignItems: 'flex-start', 
                    gap: '0.625rem', 
                    padding: '0.625rem 0.75rem', 
                    borderRadius: 'var(--radius-sm)', 
                    background: uploadMode === 'append' ? 'var(--bg-surface-elevated)' : 'transparent',
                    border: '1px solid var(--border-subtle)',
                    cursor: 'pointer' 
                  }}>
                    <input 
                      type="radio" 
                      name="uploadMode" 
                      value="append" 
                      checked={uploadMode === 'append'} 
                      onChange={() => setUploadMode('append')}
                      style={{ marginTop: '3px' }}
                    />
                    <div>
                      <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', display: 'block' }}>
                        Add to active reviews
                      </span>
                      <span style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>
                        Append the reviews in this file to currently saved records.
                      </span>
                    </div>
                  </label>
                </div>
              </div>

              {uploadError && (
                <div style={{ 
                  marginBottom: '1rem', 
                  padding: '0.625rem 0.875rem', 
                  borderRadius: 'var(--radius-sm)', 
                  background: 'var(--negative-bg)', 
                  border: '1px solid var(--negative-border)', 
                  color: 'var(--negative)', 
                  fontSize: '0.8rem' 
                }}>
                  {uploadError}
                </div>
              )}

              {/* Modal Buttons */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <button 
                  type="button" 
                  className="btn btn-ghost"
                  onClick={handleLoadDemoDataset}
                  style={{ fontSize: '0.75rem' }}
                >
                  <Sparkles size={12} />
                  Load Demo Reviews
                </button>

                <div style={{ display: 'flex', gap: '0.625rem' }}>
                  <button 
                    type="button" 
                    className="btn btn-secondary"
                    onClick={() => setIsUploadModalOpen(false)}
                    disabled={isUploadingCsv}
                  >
                    Cancel
                  </button>
                  <button 
                    type="button" 
                    className="btn btn-primary"
                    onClick={handleCsvUploadSubmit}
                    disabled={isUploadingCsv || !csvFile}
                  >
                    {isUploadingCsv ? (
                      <>
                        <RefreshCw size={14} className="animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      'Analyze CSV'
                    )}
                  </button>
                </div>
              </div>

            </div>

          </div>
        </div>
      )}

    </div>
  );
}
