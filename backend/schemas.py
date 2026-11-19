from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Base & Common Schemas
# -------------------------------------------------------------

class Probabilities(BaseModel):
    Negative: float
    Neutral: float
    Positive: float

class SentimentDistribution(BaseModel):
    Positive: int
    Neutral: int
    Negative: int

class SentimentPercentages(BaseModel):
    Positive: float
    Neutral: float
    Negative: float

# -------------------------------------------------------------
# Single Review Prediction (Ad-Hoc / Stateless)
# -------------------------------------------------------------

class ReviewRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        description="The customer review text to analyze."
    )

class ReviewResponse(BaseModel):
    text: str
    cleaned_text: str
    sentiment: str
    confidence: float
    probabilities: Probabilities

class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=5000, 
        description="List of review texts to analyze in a single batch."
    )

class BatchReviewResponse(BaseModel):
    total_reviews: int
    sentiment_distribution: SentimentDistribution
    sentiment_percentages: SentimentPercentages
    results: List[ReviewResponse]

# -------------------------------------------------------------
# Persistent Dataset & Review Submission Schemas
# -------------------------------------------------------------

class SubmitReviewRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=3, 
        description="Customer review feedback text (required)."
    )
    rating: Optional[int] = Field(
        None, 
        ge=1, 
        le=5, 
        description="Optional customer rating score (1 to 5 stars)."
    )
    customer_name: Optional[str] = Field(
        None, 
        max_length=100, 
        description="Optional customer name."
    )
    product_name: Optional[str] = Field(
        None, 
        max_length=100, 
        description="Optional product or item name."
    )

class ReviewItem(BaseModel):
    id: str
    text: str
    rating: Optional[int] = None
    customer_name: Optional[str] = "Customer"
    product_name: Optional[str] = "Food Product"
    predicted_sentiment: str
    confidence: float
    probabilities: Probabilities
    created_at: Optional[str] = ""

class FeedbackTheme(BaseModel):
    theme: str
    mentions: int

class TimelineItem(BaseModel):
    period: str
    Positive: int
    Neutral: int
    Negative: int

class DatasetSummaryResponse(BaseModel):
    has_data: bool
    total_reviews: int
    sentiment_distribution: SentimentDistribution
    sentiment_percentages: SentimentPercentages
    rating_distribution: Dict[str, int]
    timeline: List[TimelineItem]
    positive_themes: List[FeedbackTheme]
    negative_themes: List[FeedbackTheme]
    last_updated: Optional[str] = None

class ReviewsListResponse(BaseModel):
    total: int
    page: int
    pages: int
    items: List[ReviewItem]

class SubmitReviewResponse(BaseModel):
    review: ReviewItem
    summary: DatasetSummaryResponse

# -------------------------------------------------------------
# CSV Upload & Health Schemas
# -------------------------------------------------------------

class CsvReviewItem(BaseModel):
    id: Optional[Any] = None
    original_text: str
    cleaned_text: str
    predicted_sentiment: str
    confidence: float
    probabilities: Probabilities

class CsvUploadResponse(BaseModel):
    filename: str
    total_reviews: int
    text_column_detected: str
    mode: str
    sentiment_distribution: SentimentDistribution
    sentiment_percentages: SentimentPercentages
    results_sample: List[ReviewItem]
    summary: DatasetSummaryResponse

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_directory: str
    vocabulary_size: int
    classes: List[str]
