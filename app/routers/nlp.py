"""
NLP API Endpoints Router.
Provides /analyze-sentiment endpoint.
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas import SentimentRequest, SentimentResponse
from pipeline import get_pipeline

router = APIRouter(prefix="", tags=["Natural Language Processing"])


@router.post(
    "/analyze-sentiment",
    response_model=SentimentResponse,
    summary="Analyze Customer Feedback Sentiment",
    description="Evaluates customer review or chat text and classifies sentiment into Positive, Neutral, or Negative with confidence score."
)
async def analyze_sentiment(payload: SentimentRequest):
    try:
        pipeline = get_pipeline()
        result = pipeline.analyze_sentiment(payload.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis processing error: {str(e)}"
        )
