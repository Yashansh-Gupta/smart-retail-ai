"""
Unified ML Pipeline for Smart Retail Platform.
Wraps Computer Vision, Natural Language Processing, and Chatbot models behind
a single unified class loaded once at startup.
"""

import os
import sys
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.cv_service import VisionService
from app.services.nlp_service import NLPService
from app.services.chatbot_service import ChatbotService


class SmartRetailPipeline:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SmartRetailPipeline, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        face_db_path: str = "app/models/face_db.pkl",
        product_model_path: str = "app/models/product_classifier.h5",
        sentiment_model_path: str = "app/models/sentiment_model.pkl",
        chatbot_model_path: str = "app/models/chatbot_model.pkl",
        intents_path: str = "data/intents.json"
    ):
        if self._initialized:
            return

        print("Initializing Unified Smart Retail Pipeline (Loading all models once)...")
        self.vision_service = VisionService(face_db_path=face_db_path, product_model_path=product_model_path)
        self.nlp_service = NLPService(model_path=sentiment_model_path)
        self.chatbot_service = ChatbotService(model_path=chatbot_model_path, intents_path=intents_path)
        self._initialized = True
        print("Unified ML Pipeline successfully initialized!")

    def recognize_customer(self, image_bytes: bytes) -> Dict[str, Any]:
        """Perform facial recognition and visit logging."""
        return self.vision_service.recognize_customer_face(image_bytes)

    def classify_product(self, image_bytes: bytes, filename: str = "", *args, **kwargs) -> Dict[str, Any]:
        """Classify product category from image."""
        fn = filename or (args[0] if len(args) > 0 and isinstance(args[0], str) else "") or kwargs.get("filename", "")
        return self.vision_service.classify_product(image_bytes, filename=fn)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze customer text sentiment."""
        return self.nlp_service.analyze_sentiment(text)

    def process_chat_message(self, message: str) -> Dict[str, Any]:
        """Process customer support message and return chatbot reply."""
        return self.chatbot_service.get_reply(message)

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Compute aggregated statistics for customer visits and sentiment trends."""
        return {}


def get_pipeline() -> SmartRetailPipeline:
    """Helper function to get the singleton pipeline instance."""
    return SmartRetailPipeline()
