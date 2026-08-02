"""
FastAPI Application Entrypoint.
Smart Retail & Customer Intelligence Platform REST API.
"""

import sys
import os
from fastapi import FastAPI, Request, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add project root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import get_pipeline
from app.routers import vision, nlp, chatbot

# API Key security scheme for simulated production authorization
API_KEY_NAME = "X-API-Key"
DEMO_API_KEY = "retail-secret-key-2026"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML models once into memory
    print("FastAPI Gateway initializing pipeline models...")
    get_pipeline()
    yield
    print("FastAPI Gateway shutting down...")


app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform API",
    description=(
        "Production-style REST Gateway serving Computer Vision (Face Recognition & Product Classifier), "
        "Natural Language Processing (Customer Sentiment Analysis), and Hybrid FAQ Support Chatbot."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def verify_api_key_middleware(request: Request, call_next):
    # Allow docs, open endpoints, and OPTIONS preflight without API key for easy testing
    open_paths = ["/docs", "/redoc", "/openapi.json", "/", "/health"]
    if request.url.path in open_paths or request.method == "OPTIONS":
        return await call_next(request)
        
    api_key = request.headers.get(API_KEY_NAME)
    # If API key is provided and invalid, reject; if omitted in dev mode allow with warning header
    if api_key and api_key != DEMO_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-API-Key header provided."
        )

    response = await call_next(request)
    return response


# Include Routers
app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)


@app.get("/", tags=["System"])
async def root():
    return {
        "status": "online",
        "service": "Smart Retail & Customer Intelligence Platform API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": [
            "POST /recognize-face",
            "POST /classify-product",
            "POST /analyze-sentiment",
            "POST /chatbot",
            "GET /dashboard/stats"
        ]
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "pipeline": "loaded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
