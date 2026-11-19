import os
import io
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.predictor import predictor, LABEL_MAP
from backend.schemas import (
    ReviewRequest,
    ReviewResponse,
    BatchReviewRequest,
    BatchReviewResponse,
    CsvUploadResponse,
    CsvReviewItem,
    HealthResponse,
    SentimentDistribution,
    SentimentPercentages,
    Probabilities
)

app = FastAPI(
    title="Fine-Food Review Sentiment Analyzer API",
    description="""
    Production-grade Machine Learning API for classifying Amazon Fine Food customer reviews.
    
    ### Capabilities:
    * **Manual Review Prediction**: Classifies single user-inputted review with confidence scores.
    * **Batch Prediction**: High-throughput vectorized classification for multiple reviews.
    * **CSV Upload & Analysis**: Upload dataset CSV, auto-detect text column, and generate aggregated distribution analytics.
    * **CSV Export**: Download analyzed CSV with appended sentiment predictions and probability columns.
    """,
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
def root():
    return {
        "project": "Fine-Food Review Sentiment Analyzer",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "POST /predict": "Predict sentiment for a single review",
            "POST /predict-batch": "Predict sentiment for a list of reviews",
            "POST /upload-csv": "Upload CSV file for sentiment analytics",
            "POST /export-csv": "Download processed CSV with predicted sentiment"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts could not be located or loaded."
        )
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
        model_directory=predictor.model_dir,
        vocabulary_size=len(predictor.vectorizer.vocabulary_) if predictor.vectorizer else 0,
        classes=list(LABEL_MAP.values())
    )

@app.post("/predict", response_model=ReviewResponse, tags=["Inference"])
def predict_sentiment(request: ReviewRequest):
    """
    Predict sentiment (Positive, Neutral, Negative) for a single customer review.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Please verify artifacts in model/ directory."
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty or purely whitespace."
        )

    try:
        result = predictor.predict_single(text)
        return ReviewResponse(
            text=result["text"],
            cleaned_text=result["cleaned_text"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=Probabilities(**result["probabilities"])
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/predict-batch", response_model=BatchReviewResponse, tags=["Inference"])
def predict_batch_sentiment(request: BatchReviewRequest):
    """
    Predict sentiment for a batch of customer reviews.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready."
        )

    if not request.reviews:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reviews list cannot be empty."
        )

    try:
        batch_out = predictor.predict_batch(request.reviews)
        results = [
            ReviewResponse(
                text=r["text"],
                cleaned_text=r["cleaned_text"],
                sentiment=r["sentiment"],
                confidence=r["confidence"],
                probabilities=Probabilities(**r["probabilities"])
            )
            for r in batch_out["results"]
        ]
        return BatchReviewResponse(
            total_reviews=batch_out["total_reviews"],
            sentiment_distribution=SentimentDistribution(**batch_out["sentiment_distribution"]),
            sentiment_percentages=SentimentPercentages(**batch_out["sentiment_percentages"]),
            results=results
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}"
        )

def _detect_text_column(df: pd.DataFrame) -> str:
    """
    Auto-detect the most likely text column containing customer reviews.
    """
    candidates = ['Text', 'text', 'Review', 'review', 'Reviews', 'reviews', 'Comment', 'comment', 'Summary', 'summary', 'body']
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    
    # Fallback to the first string/object column
    for col in df.columns:
        if df[col].dtype == 'object':
            return col
            
    return None

@app.post("/upload-csv", response_model=CsvUploadResponse, tags=["CSV Analytics"])
async def upload_csv_analysis(file: UploadFile = File(...)):
    """
    Upload a CSV file containing reviews. Auto-detects the review text column,
    classifies all reviews, and returns aggregate sentiment distribution analytics.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready."
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a valid .csv file."
        )

    try:
        contents = await file.read()
        # Parse CSV safely with on_bad_lines fallback
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception:
            df = pd.read_csv(io.BytesIO(contents), on_bad_lines='skip')
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read CSV: {str(e)}"
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty."
        )

    text_col = _detect_text_column(df)
    if not text_col:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not automatically detect review text column. Available columns: {list(df.columns)}"
        )

    # Clean missing values in the text column
    raw_texts = df[text_col].fillna("").astype(str).tolist()

    # Limit to maximum 10,000 rows per single CSV upload request for latency protection
    max_rows = 10000
    is_truncated = False
    if len(raw_texts) > max_rows:
        raw_texts = raw_texts[:max_rows]
        is_truncated = True

    batch_out = predictor.predict_batch(raw_texts)

    # Build sample preview (first 100 rows for frontend table rendering)
    preview_limit = min(100, len(batch_out["results"]))
    results_sample = []
    id_col = 'Id' if 'Id' in df.columns else ('id' if 'id' in df.columns else None)

    for i in range(preview_limit):
        item = batch_out["results"][i]
        row_id = df.iloc[i][id_col] if id_col else i + 1
        results_sample.append(CsvReviewItem(
            id=str(row_id),
            original_text=item["text"][:300] + ("..." if len(item["text"]) > 300 else ""),
            cleaned_text=item["cleaned_text"][:200],
            predicted_sentiment=item["sentiment"],
            confidence=item["confidence"],
            probabilities=Probabilities(**item["probabilities"])
        ))

    return CsvUploadResponse(
        filename=file.filename + (" (Truncated to first 10,000 rows)" if is_truncated else ""),
        total_reviews=batch_out["total_reviews"],
        text_column_detected=text_col,
        sentiment_distribution=SentimentDistribution(**batch_out["sentiment_distribution"]),
        sentiment_percentages=SentimentPercentages(**batch_out["sentiment_percentages"]),
        results_sample=results_sample,
        download_available=True
    )

@app.post("/export-csv", tags=["CSV Analytics"])
async def export_analyzed_csv(file: UploadFile = File(...)):
    """
    Upload CSV file, run sentiment inference, and download the full CSV
    with Predicted_Sentiment, Confidence, and Probabilities appended.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready."
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a .csv format."
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse CSV: {str(e)}"
        )

    text_col = _detect_text_column(df)
    if not text_col:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not detect review text column."
        )

    raw_texts = df[text_col].fillna("").astype(str).tolist()
    batch_out = predictor.predict_batch(raw_texts)

    df["Predicted_Sentiment"] = [r["sentiment"] for r in batch_out["results"]]
    df["Confidence"] = [r["confidence"] for r in batch_out["results"]]
    df["Prob_Positive"] = [r["probabilities"]["Positive"] for r in batch_out["results"]]
    df["Prob_Neutral"] = [r["probabilities"]["Neutral"] for r in batch_out["results"]]
    df["Prob_Negative"] = [r["probabilities"]["Negative"] for r in batch_out["results"]]

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    export_filename = f"analyzed_{file.filename}"
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={export_filename}"}
    )
