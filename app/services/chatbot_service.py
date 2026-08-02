"""
Chatbot Service Layer.
Implements a Hybrid FAQ & Customer Support Chatbot combining:
1. Rule-based exact/regex intent matching for core retail questions
2. ML-based fallback intent classification using TF-IDF + SGDClassifier
"""

import os
import re
import random
import pickle
import json
from typing import Dict, Any, Optional, List


class ChatbotService:
    def __init__(self, model_path: str = "app/models/chatbot_model.pkl", intents_path: str = "data/intents.json"):
        self.model_path = model_path
        self.intents_path = intents_path
        self.intents_data = None
        self.pipeline = None
        self.rule_patterns: List[Dict[str, Any]] = []
        self.load_intents_and_model()

    def load_intents_and_model(self) -> None:
        """Load intents definition and serialized ML classifier."""
        # Load raw intents JSON
        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r") as f:
                self.intents_data = json.load(f)
                self._compile_rule_patterns()

        # Load ML pipeline
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
                self.pipeline = saved.get("pipeline")
                if not self.intents_data:
                    self.intents_data = saved.get("intents")
                    self._compile_rule_patterns()

    def _compile_rule_patterns(self) -> None:
        """Compile regex rules for fast rule-based matching."""
        self.rule_patterns = []
        if not self.intents_data or "intents" not in self.intents_data:
            return
            
        for intent in self.intents_data["intents"]:
            tag = intent["tag"]
            patterns = intent["patterns"]
            responses = intent["responses"]
            
            # Combine patterns into regex OR list
            regex_str = r'\b(' + '|'.join([re.escape(p) for p in patterns]) + r')\b'
            compiled = re.compile(regex_str, re.IGNORECASE)
            
            self.rule_patterns.append({
                "tag": tag,
                "regex": compiled,
                "responses": responses
            })

    def match_rule_based(self, message: str) -> Optional[Dict[str, Any]]:
        """Try exact / keyword rule-based match first."""
        cleaned_msg = message.strip()
        for rule in self.rule_patterns:
            if rule["regex"].search(cleaned_msg):
                response = random.choice(rule["responses"])
                return {
                    "matched_by": "Rule-Based",
                    "intent": rule["tag"],
                    "confidence": 98.5,
                    "response": response
                }
        return None

    def match_ml_based(self, message: str) -> Dict[str, Any]:
        """ML classifier fallback when no explicit rule triggers."""
        if not self.pipeline or not self.intents_data:
            return {
                "matched_by": "Fallback",
                "intent": "unknown",
                "confidence": 50.0,
                "response": "I'm sorry, I didn't quite understand that. You can ask me about order status, return policies, store hours, shipping fees, or contact human support!"
            }

        probs = self.pipeline.predict_proba([message])[0]
        classes = list(self.pipeline.classes_)
        top_idx = int(np_argmax(probs))
        predicted_tag = classes[top_idx]
        confidence = float(probs[top_idx]) * 100

        # Retrieve response for predicted tag
        responses = ["How can I help you today?"]
        for intent in self.intents_data.get("intents", []):
            if intent["tag"] == predicted_tag:
                responses = intent["responses"]
                break

        if confidence < 20.0:
            return {
                "matched_by": "ML-Fallback-LowConfidence",
                "intent": "unknown",
                "confidence": round(confidence, 2),
                "response": "I'm not completely sure I understood. Are you asking about order tracking, returns, store hours, or shipping?"
            }

        return {
            "matched_by": "ML-Classifier",
            "intent": predicted_tag,
            "confidence": round(confidence, 2),
            "response": random.choice(responses)
        }

    def get_reply(self, message: str) -> Dict[str, Any]:
        """
        Hybrid Chatbot response pipeline:
        1. Attempt rule-based match
        2. Fallback to ML intent classifier
        """
        # Check rule match
        rule_result = self.match_rule_based(message)
        if rule_result:
            return rule_result

        # Check ML classifier match
        return self.match_ml_based(message)


def np_argmax(arr):
    import numpy as np
    return np.argmax(arr)
