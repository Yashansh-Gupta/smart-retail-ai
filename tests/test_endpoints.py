"""
Automated Pytest Suite for Smart Retail Platform APIs & Services.
"""

import sys
import os
import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from cv_utils import to_grayscale, resize_image, apply_canny_edge_detection, apply_gaussian_blur
from pipeline import get_pipeline

client = TestClient(app)
DEMO_KEY = os.getenv("API_KEY", "demo-key")


def create_dummy_image_bytes(color=(200, 150, 100)) -> bytes:
    """Helper to generate sample test JPEG image bytes."""
    img = Image.new("RGB", (200, 200), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------
# System & Health Endpoint Tests
# ---------------------------------------------------------
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "Smart Retail" in data["service"]


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


# ---------------------------------------------------------
# Computer Vision Endpoints Tests
# ---------------------------------------------------------
def test_recognize_face_endpoint():
    img_bytes = create_dummy_image_bytes()
    files = {"file": ("test_face.jpg", img_bytes, "image/jpeg")}
    headers = {"X-API-Key": DEMO_KEY}
    
    response = client.post("/recognize-face", files=files, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "recognized" in data
    assert "customer_id" in data
    assert "status" in data
    assert "confidence" in data


def test_classify_product_endpoint():
    img_bytes = create_dummy_image_bytes(color=(50, 180, 220))
    files = {"file": ("test_product.jpg", img_bytes, "image/jpeg")}
    headers = {"X-API-Key": DEMO_KEY}
    
    response = client.post("/classify-product", files=files, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert data["predicted_category"] in ["shoes", "bags", "electronics", "clothing", "groceries"]
    assert "confidence" in data
    assert "category_probabilities" in data


# ---------------------------------------------------------
# NLP & Chatbot Endpoints Tests
# ---------------------------------------------------------
def test_analyze_sentiment_positive():
    payload = {"text": "I absolutely love this dress! Magnificent quality and super comfortable."}
    headers = {"X-API-Key": DEMO_KEY}
    response = client.post("/analyze-sentiment", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]
    assert data["confidence"] > 0


def test_analyze_sentiment_negative():
    payload = {"text": "Worst experience ever. Broken item and terrible customer support."}
    headers = {"X-API-Key": DEMO_KEY}
    response = client.post("/analyze-sentiment", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]


def test_chatbot_endpoint_rule_match():
    payload = {"message": "What is your return policy?"}
    headers = {"X-API-Key": DEMO_KEY}
    response = client.post("/chatbot", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "matched_by" in data
    assert "intent" in data
    assert "response" in data
    assert len(data["response"]) > 0


def test_chatbot_endpoint_ml_fallback():
    payload = {"message": "Can I get a student promo code discount for my purchase?"}
    headers = {"X-API-Key": DEMO_KEY}
    response = client.post("/chatbot", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data


def test_dashboard_stats_endpoint():
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_visits" in data
    assert "returning_customers_count" in data
    assert "loyalty_breakdown" in data
    assert "sentiment_summary" in data


# ---------------------------------------------------------
# CV Utils & Pipeline Unit Tests
# ---------------------------------------------------------
def test_cv_utils_preprocessing():
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = to_grayscale(test_img)
    assert len(gray.shape) == 2
    
    resized = resize_image(test_img, width=50)
    assert resized.shape[1] == 50
    
    blurred = apply_gaussian_blur(gray)
    assert blurred.shape == gray.shape
    
    edges = apply_canny_edge_detection(blurred)
    assert edges.shape == gray.shape
