"""
Pydantic Schemas for Request & Response Data Validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Tuple


# --- Vision Schemas ---
class FaceRecognitionResponse(BaseModel):
    recognized: bool = Field(..., description="Whether face matched a registered customer")
    status: str = Field(..., description="Customer status (Returning Customer or New Guest)")
    customer_id: str = Field(..., description="Unique customer ID")
    name: str = Field(..., description="Customer full name")
    loyalty_tier: str = Field(..., description="Loyalty tier level (VIP, Platinum, Silver, None)")
    confidence: float = Field(..., description="Recognition confidence percentage")
    distance: float = Field(..., description="Encoding distance score")
    bounding_box: Tuple[int, int, int, int] = Field(..., description="Face bounding box (x, y, w, h)")
    timestamp: str = Field(..., description="ISO 8601 visit timestamp")


class ProductClassificationResponse(BaseModel):
    predicted_category: str = Field(..., description="Predicted product category (shoes, bags, electronics, clothing, groceries)")
    confidence: float = Field(..., description="Classification confidence percentage")
    category_probabilities: Dict[str, float] = Field(..., description="Probabilities across all categories")
    image_dimensions: Dict[str, int] = Field(..., description="Uploaded image width and height")


# --- NLP Schemas ---
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, json_schema_extra={"example": "This handbag is incredible! Super stylish and durable."})


class SentimentResponse(BaseModel):
    text: str = Field(..., description="Input text evaluated")
    sentiment: str = Field(..., description="Classified sentiment: Positive, Neutral, or Negative")
    confidence: float = Field(..., description="Prediction confidence percentage")
    probabilities: Dict[str, float] = Field(..., description="Probability breakdown across categories")
    preprocessing: Dict[str, Any] = Field(..., description="Detailed text preprocessing steps breakdown")


# --- Chatbot Schemas ---
class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, json_schema_extra={"example": "What is your return policy for shoes?"})


class ChatbotResponse(BaseModel):
    matched_by: str = Field(..., description="Matching mechanism: Rule-Based or ML-Classifier")
    intent: str = Field(..., description="Identified intent tag")
    confidence: float = Field(..., description="Intent matching confidence percentage")
    response: str = Field(..., description="Automated chatbot response text")


# --- Dashboard Stats Schemas ---
class DashboardStatsResponse(BaseModel):
    total_visits: int = Field(..., description="Total recorded customer visits")
    returning_customers_count: int = Field(..., description="Count of recognized returning loyalty customers")
    guest_visits_count: int = Field(..., description="Count of unregistered guest visits")
    loyalty_breakdown: Dict[str, int] = Field(..., description="Count of visits per loyalty tier")
    recent_visits: List[Dict[str, Any]] = Field(..., description="List of recent visit logs")
    sentiment_summary: Dict[str, int] = Field(..., description="Distribution of overall sentiment feedback")
