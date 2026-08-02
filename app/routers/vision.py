"""
Computer Vision API Endpoints Router.
Provides /recognize-face and /classify-product endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas import FaceRecognitionResponse, ProductClassificationResponse
from pipeline import get_pipeline

router = APIRouter(prefix="", tags=["Computer Vision"])


@router.post(
    "/recognize-face",
    response_model=FaceRecognitionResponse,
    summary="Recognize Customer Face & Log Visit",
    description="Accepts an uploaded face image, extracts facial encodings, matches against stored customer database, and logs visit timestamp."
)
async def recognize_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image format."
        )
    try:
        contents = await file.read()
        pipeline = get_pipeline()
        result = pipeline.recognize_customer(contents)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Face recognition processing error: {str(e)}"
        )


@router.post(
    "/classify-product",
    response_model=ProductClassificationResponse,
    summary="Classify Product Category",
    description="Accepts an uploaded product image and classifies it into one of 5 retail categories: shoes, bags, electronics, clothing, groceries."
)
async def classify_product(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image format."
        )
    try:
        contents = await file.read()
        pipeline = get_pipeline()
        result = pipeline.classify_product(contents, filename=file.filename or "")
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Product classification processing error: {str(e)}"
        )
