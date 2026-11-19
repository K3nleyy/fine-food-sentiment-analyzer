from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ReviewRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        description="The customer review text to analyze.",
        example="The dark chocolate almonds arrived fresh, crunchy, and absolutely delicious!"
    )

class Probabilities(BaseModel):
    Negative: float
    Neutral: float
    Positive: float

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

class SentimentDistribution(BaseModel):
    Positive: int
    Neutral: int
    Negative: int

class SentimentPercentages(BaseModel):
    Positive: float
    Neutral: float
    Negative: float

class BatchReviewResponse(BaseModel):
    total_reviews: int
    sentiment_distribution: SentimentDistribution
    sentiment_percentages: SentimentPercentages
    results: List[ReviewResponse]

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
    sentiment_distribution: SentimentDistribution
    sentiment_percentages: SentimentPercentages
    results_sample: List[CsvReviewItem]
    download_available: bool

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_directory: str
    vocabulary_size: int
    classes: List[str]
