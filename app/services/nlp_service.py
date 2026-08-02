"""
Natural Language Processing Service Layer.
Implements text cleaning & preprocessing (lowercasing, punctuation stripping, stopword removal,
tokenization) and sentiment analysis inference (Positive / Neutral / Negative with confidence).
"""

import os
import re
import string
import pickle
from typing import Dict, Any, List


class NLPService:
    def __init__(self, model_path: str = "app/models/sentiment_model.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self.stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
            "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
            "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
            "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
            "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
            "most", "my", "myself", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
            "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
            "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
            "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
            "what", "when", "where", "which", "while", "who", "whom", "why", "with", "you", "your",
            "yours", "yourself", "yourselves"
        }
        self.load_model()

    def load_model(self) -> None:
        """Load trained TF-IDF + Sentiment classifier pipeline."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.pipeline = pickle.load(f)

    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Complete text preprocessing pipeline:
        Lowercasing -> Punctuation Removal -> Stopword Removal -> Tokenization.
        """
        raw_text = text or ""
        # Lowercase
        lowercased = raw_text.lower()
        # Remove punctuation & special characters
        cleaned = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', lowercased)
        # Tokenize
        tokens = cleaned.split()
        # Remove stopwords
        filtered_tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        
        normalized_text = " ".join(filtered_tokens)
        
        return {
            "raw_text": raw_text,
            "lowercased": lowercased,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "cleaned_text": normalized_text
        }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Classify sentiment into Positive, Neutral, or Negative with confidence score."""
        prep_info = self.preprocess_text(text)
        cleaned_text = prep_info["cleaned_text"]
        
        if not cleaned_text.strip():
            return {
                "text": text,
                "sentiment": "Neutral",
                "confidence": 50.0,
                "probabilities": {"Positive": 33.3, "Neutral": 33.4, "Negative": 33.3},
                "preprocessing": prep_info
            }

        if self.pipeline:
            probs = self.pipeline.predict_proba([cleaned_text])[0]
            classes = list(self.pipeline.classes_)
            top_idx = int(np_argmax(probs))
            sentiment = classes[top_idx]
            confidence = float(probs[top_idx]) * 100
            
            prob_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, probs)}
        else:
            # Simple lexicon fallback rule matcher if model not loaded
            pos_words = {"great", "love", "awesome", "good", "excellent", "fast", "best", "perfect", "comfortable", "beautiful"}
            neg_words = {"bad", "poor", "terrible", "worst", "broken", "defective", "slow", "disappointed", "damaged", "cheap"}
            
            tokens = set(prep_info["tokens"])
            pos_count = len(tokens.intersection(pos_words))
            neg_count = len(tokens.intersection(neg_words))
            
            if pos_count > neg_count:
                sentiment = "Positive"
                confidence = 85.0
            elif neg_count > pos_count:
                sentiment = "Negative"
                confidence = 85.0
            else:
                sentiment = "Neutral"
                confidence = 65.0
            
            prob_dict = {"Positive": 33.3, "Neutral": 33.4, "Negative": 33.3}
            prob_dict[sentiment] = round(confidence, 2)

        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "probabilities": prob_dict,
            "preprocessing": prep_info
        }


def np_argmax(arr):
    import numpy as np
    return np.argmax(arr)
