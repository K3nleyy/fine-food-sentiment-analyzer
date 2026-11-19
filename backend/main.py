import os
import io
from datetime import datetime, timezone
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.predictor import predictor, LABEL_MAP
from backend.dataset_store import dataset_store
from backend.schemas import (
    ReviewRequest,
    ReviewResponse,
    BatchReviewRequest,
    BatchReviewResponse,
    SubmitReviewRequest,
    SubmitReviewResponse,
    DatasetSummaryResponse,
    ReviewsListResponse,
    ReviewItem,
    CsvUploadResponse,
    HealthResponse,
    SentimentDistribution,
    SentimentPercentages,
    Probabilities,
    FeedbackTheme,
    TimelineItem
)

app = FastAPI(
    title="Fine-Food Insights API",
    description="""
    Commercial Customer Feedback Sentiment Analysis Platform.
    
    ### Capabilities:
    * **Customer Feedback Analytics**: Live aggregated sentiment, rating distributions, and trend analysis.
    * **Review Submission**: Real-time review intake with instant model inference and persistent CSV storage.
    * **CSV Dataset Management**: Upload and batch-analyze customer reviews with 'replace' or 'append' workflows.
    * **Data Export**: Export analyzed review datasets with sentiment scores and confidence values.
    """,
    version="2.0.0"
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# System & Health Endpoints
# -------------------------------------------------------------

@app.get("/", tags=["General"])
def root():
    return {
        "platform": "Fine-Food Insights",
        "description": "Understand your customer feedback.",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "GET /api/dataset/summary": "Fetch aggregated sentiment metrics & themes",
            "GET /api/reviews": "Fetch paginated and filtered customer reviews",
            "POST /api/reviews": "Submit a new customer review with instant inference",
            "POST /api/upload-csv": "Upload and batch-analyze customer review CSV",
            "GET /api/dataset/export": "Download active analyzed review dataset",
            "POST /predict": "Ad-hoc stateless sentiment prediction"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sentiment analysis model is not ready."
        )
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded,
        model_directory=predictor.model_dir,
        vocabulary_size=len(predictor.vectorizer.vocabulary_) if predictor.vectorizer else 0,
        classes=list(LABEL_MAP.values())
    )

# -------------------------------------------------------------
# Active Dataset & Customer Analytics Endpoints
# -------------------------------------------------------------

@app.get("/api/dataset/summary", response_model=DatasetSummaryResponse, tags=["Analytics"])
def get_dataset_summary():
    """
    Returns high-level business analytics for the active review dataset:
    total count, sentiment counts/percentages, rating distributions, 
    sentiment timeline trends, and key feedback themes.
    """
    data = dataset_store.get_summary()
    return DatasetSummaryResponse(
        has_data=data["has_data"],
        total_reviews=data["total_reviews"],
        sentiment_distribution=SentimentDistribution(**data["sentiment_distribution"]),
        sentiment_percentages=SentimentPercentages(**data["sentiment_percentages"]),
        rating_distribution=data["rating_distribution"],
        timeline=[TimelineItem(**item) for item in data["timeline"]],
        positive_themes=[FeedbackTheme(**item) for item in data["positive_themes"]],
        negative_themes=[FeedbackTheme(**item) for item in data["negative_themes"]],
        last_updated=data["last_updated"]
    )

@app.get("/api/reviews", response_model=ReviewsListResponse, tags=["Reviews"])
def list_reviews(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query("", description="Keyword search in review text, customer, or product"),
    sentiment: Optional[str] = Query("ALL", description="Filter by sentiment: ALL, Positive, Neutral, Negative"),
    rating: Optional[str] = Query("ALL", description="Filter by rating: ALL, 1, 2, 3, 4, 5")
):
    """
    Returns paginated, searchable, and filtered customer reviews from the active dataset.
    """
    out = dataset_store.get_reviews(
        page=page, 
        limit=limit, 
        search=search, 
        sentiment=sentiment, 
        rating=rating
    )
    items = [
        ReviewItem(
            id=item["id"],
            text=item["text"],
            rating=item["rating"],
            customer_name=item["customer_name"],
            product_name=item["product_name"],
            predicted_sentiment=item["predicted_sentiment"],
            confidence=item["confidence"],
            probabilities=Probabilities(**item["probabilities"]),
            created_at=item["created_at"]
        )
        for item in out["items"]
    ]
    return ReviewsListResponse(
        total=out["total"],
        page=out["page"],
        pages=out["pages"],
        items=items
    )

@app.post("/api/reviews", response_model=SubmitReviewResponse, tags=["Reviews"])
def submit_new_review(request: SubmitReviewRequest):
    """
    Submits a new customer review:
    1. Validates submitted review information.
    2. Runs sentiment inference with the trained TF-IDF & Logistic Regression model.
    3. Saves review, rating, and predicted sentiment to the persistent dataset CSV.
    4. Automatically recalculates and returns the updated dashboard metrics.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for prediction."
        )

    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review feedback text is required."
        )

    try:
        new_row = dataset_store.add_single_review(
            text=text,
            rating=request.rating,
            customer_name=request.customer_name,
            product_name=request.product_name
        )

        review_item = ReviewItem(
            id=new_row["id"],
            text=new_row["text"],
            rating=int(new_row["rating"]) if new_row["rating"] else None,
            customer_name=new_row["customer_name"],
            product_name=new_row["product_name"],
            predicted_sentiment=new_row["predicted_sentiment"],
            confidence=new_row["confidence"],
            probabilities=Probabilities(
                Positive=new_row["prob_positive"],
                Neutral=new_row["prob_neutral"],
                Negative=new_row["prob_negative"]
            ),
            created_at=new_row["created_at"]
        )

        summary_data = dataset_store.get_summary()
        summary_resp = DatasetSummaryResponse(
            has_data=summary_data["has_data"],
            total_reviews=summary_data["total_reviews"],
            sentiment_distribution=SentimentDistribution(**summary_data["sentiment_distribution"]),
            sentiment_percentages=SentimentPercentages(**summary_data["sentiment_percentages"]),
            rating_distribution=summary_data["rating_distribution"],
            timeline=[TimelineItem(**item) for item in summary_data["timeline"]],
            positive_themes=[FeedbackTheme(**item) for item in summary_data["positive_themes"]],
            negative_themes=[FeedbackTheme(**item) for item in summary_data["negative_themes"]],
            last_updated=summary_data["last_updated"]
        )

        return SubmitReviewResponse(
            review=review_item,
            summary=summary_resp
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit and store review: {str(e)}"
        )

# -------------------------------------------------------------
# CSV Dataset Upload & Management
# -------------------------------------------------------------

@app.post("/api/upload-csv", response_model=CsvUploadResponse, tags=["CSV Analytics"])
async def upload_csv_to_dataset(
    file: UploadFile = File(...),
    mode: str = Form("replace")  # 'replace' or 'append'
):
    """
    Upload a customer review CSV:
    - Auto-detects review text, rating, customer name, and date columns.
    - Classifies reviews using the trained sentiment model.
    - Persistently stores data into active_reviews.csv (mode: 'replace' or 'append').
    - Returns updated dashboard metrics and preview rows.
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

    mode = mode.lower().strip()
    if mode not in ["replace", "append"]:
        mode = "replace"

    try:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception:
            df = pd.read_csv(io.BytesIO(contents), on_bad_lines='skip')
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV: {str(e)}"
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty."
        )

    try:
        imported_count, text_col = dataset_store.import_dataframe(df, mode=mode)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing error: {str(e)}"
        )

    summary_data = dataset_store.get_summary()
    summary_resp = DatasetSummaryResponse(
        has_data=summary_data["has_data"],
        total_reviews=summary_data["total_reviews"],
        sentiment_distribution=SentimentDistribution(**summary_data["sentiment_distribution"]),
        sentiment_percentages=SentimentPercentages(**summary_data["sentiment_percentages"]),
        rating_distribution=summary_data["rating_distribution"],
        timeline=[TimelineItem(**item) for item in summary_data["timeline"]],
        positive_themes=[FeedbackTheme(**item) for item in summary_data["positive_themes"]],
        negative_themes=[FeedbackTheme(**item) for item in summary_data["negative_themes"]],
        last_updated=summary_data["last_updated"]
    )

    # Preview sample
    preview_reviews = dataset_store.get_reviews(page=1, limit=10)
    sample_items = [
        ReviewItem(
            id=item["id"],
            text=item["text"],
            rating=item["rating"],
            customer_name=item["customer_name"],
            product_name=item["product_name"],
            predicted_sentiment=item["predicted_sentiment"],
            confidence=item["confidence"],
            probabilities=Probabilities(**item["probabilities"]),
            created_at=item["created_at"]
        )
        for item in preview_reviews["items"]
    ]

    return CsvUploadResponse(
        filename=file.filename,
        total_reviews=imported_count,
        text_column_detected=text_col,
        mode=mode,
        sentiment_distribution=SentimentDistribution(**summary_data["sentiment_distribution"]),
        sentiment_percentages=SentimentPercentages(**summary_data["sentiment_percentages"]),
        results_sample=sample_items,
        summary=summary_resp
    )

@app.post("/api/dataset/sample", response_model=DatasetSummaryResponse, tags=["Analytics"])
def load_demo_sample_dataset():
    """
    Populates the dashboard with a curated set of 25 realistic fine-food customer reviews
    spanning 1-star to 5-star experiences for instant demo exploration.
    """
    dataset_store.load_sample_dataset()
    return get_dataset_summary()

@app.post("/api/dataset/clear", response_model=DatasetSummaryResponse, tags=["Analytics"])
def clear_active_dataset():
    """
    Clears the active review dataset, resetting the dashboard to its clean initial state.
    """
    dataset_store.clear()
    return get_dataset_summary()

@app.get("/api/dataset/export", tags=["Analytics"])
def export_active_dataset():
    """
    Downloads the full active customer review dataset with predicted sentiments and confidence.
    """
    df = dataset_store.load_df()
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Active dataset has no records to export."
        )

    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    filename = f"fine_food_insights_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

# -------------------------------------------------------------
# Backward-Compatible Endpoints
# -------------------------------------------------------------

@app.post("/predict", response_model=ReviewResponse, tags=["Inference"])
def predict_sentiment(request: ReviewRequest):
    """Ad-hoc single review prediction without saving."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready."
        )
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text cannot be empty."
        )
    result = predictor.predict_single(text)
    return ReviewResponse(
        text=result["text"],
        cleaned_text=result["cleaned_text"],
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        probabilities=Probabilities(**result["probabilities"])
    )

@app.post("/predict-batch", response_model=BatchReviewResponse, tags=["Inference"])
def predict_batch_sentiment(request: BatchReviewRequest):
    """Ad-hoc batch prediction without saving."""
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

@app.post("/export-csv", tags=["CSV Analytics"])
async def export_analyzed_csv(file: UploadFile = File(...)):
    """Upload CSV file, run sentiment inference, and download the full CSV with predictions appended."""
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not ready.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv format.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {str(e)}")

    text_col = None
    for candidate in ['Text', 'text', 'Review', 'review', 'Reviews', 'reviews', 'Comment', 'comment', 'Summary']:
        if candidate in df.columns:
            text_col = candidate
            break
    if not text_col:
        for col in df.columns:
            if df[col].dtype == 'object':
                text_col = col
                break
    if not text_col:
        raise HTTPException(status_code=400, detail="Could not detect review text column.")

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
