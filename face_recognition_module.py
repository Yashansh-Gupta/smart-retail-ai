"""
Face Recognition & Customer Visit Logging Module.
Implements face detection, encoding generation, feature matching against a face database,
and timestamped customer visit logging for retail loyalty analytics.
"""

import cv2
import os
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from cv_utils import detect_faces_haar, to_grayscale, resize_image


class FaceRecognitionSystem:
    def __init__(self, db_path: str = "app/models/face_db.pkl"):
        self.db_path = db_path
        self.customers_db: Dict[str, Dict[str, Any]] = {}
        self.visit_logs: List[Dict[str, Any]] = []
        self.load_database()

    def generate_face_encoding(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Generate a robust 128-dimensional facial encoding vector using
        resized facial ROI spatial color/grayscale features & gradient histograms.
        """
        gray = to_grayscale(face_crop)
        resized = resize_image(gray, width=64, height=64)
        
        # Calculate HOG / Gradient descriptors
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # Downsample and flatten to fixed 128D normalized vector
        spatial_feats = cv2.resize(resized, (8, 8)).flatten() / 255.0
        hist_feats, _ = np.histogram(angle, bins=64, weights=mag, range=(0, 360))
        hist_feats = hist_feats / (np.linalg.norm(hist_feats) + 1e-6)
        
        encoding = np.concatenate([spatial_feats, hist_feats])
        encoding = encoding / (np.linalg.norm(encoding) + 1e-6)
        return encoding

    def register_customer(self, customer_id: str, name: str, face_image: np.ndarray, loyalty_tier: str = "Standard") -> bool:
        """Register a new consenting customer into the face database."""
        faces = detect_faces_haar(face_image)
        if not faces:
            # If full face cascade misses, crop center region as backup face ROI
            h, w = face_image.shape[:2]
            face_roi = face_image[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        else:
            x, y, w, h = faces[0]
            face_roi = face_image[y:y+h, x:x+w]

        encoding = self.generate_face_encoding(face_roi)
        self.customers_db[customer_id] = {
            "customer_id": customer_id,
            "name": name,
            "encoding": encoding,
            "loyalty_tier": loyalty_tier,
            "registered_at": datetime.now().isoformat()
        }
        self.save_database()
        return True

    def recognize_face(self, image: np.ndarray, tolerance: float = 0.65) -> Dict[str, Any]:
        """
        Detect face in image, match against stored customer encodings,
        and log visit timestamp if matched.
        """
        faces = detect_faces_haar(image)
        if not faces:
            # Use whole image as fall-back ROI if Haar cascade doesn't find faces
            h, w = image.shape[:2]
            face_roi = image
            box = (0, 0, w, h)
        else:
            x, y, w, h = faces[0]
            face_roi = image[y:y+h, x:x+w]
            box = (x, y, w, h)

        target_encoding = self.generate_face_encoding(face_roi)
        
        best_match_id = None
        min_distance = float("inf")
        
        for cust_id, data in self.customers_db.items():
            db_encoding = data["encoding"]
            # Cosine distance
            dist = 1.0 - float(np.dot(target_encoding, db_encoding))
            if dist < min_distance:
                min_distance = dist
                best_match_id = cust_id

        confidence = max(0.0, min(100.0, (1.0 - min_distance) * 100))
        
        timestamp = datetime.now().isoformat()
        
        if best_match_id and min_distance < tolerance:
            customer_info = self.customers_db[best_match_id]
            status = "Returning Customer"
            result = {
                "recognized": True,
                "status": status,
                "customer_id": customer_info["customer_id"],
                "name": customer_info["name"],
                "loyalty_tier": customer_info["loyalty_tier"],
                "confidence": round(confidence, 2),
                "distance": round(min_distance, 4),
                "bounding_box": box,
                "timestamp": timestamp
            }
        else:
            result = {
                "recognized": False,
                "status": "New Guest / Unregistered",
                "customer_id": "GUEST_UNKNOWN",
                "name": "Guest",
                "loyalty_tier": "None",
                "confidence": round(confidence, 2),
                "distance": round(min_distance, 4),
                "bounding_box": box,
                "timestamp": timestamp
            }

        # Log visit
        self.visit_logs.append(result)
        self.save_database()
        return result

    def get_visit_logs(self) -> List[Dict[str, Any]]:
        """Return all logged customer visits."""
        return self.visit_logs

    def save_database(self) -> None:
        """Serialize face database and visit logs to pkl file."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = {
            "customers_db": self.customers_db,
            "visit_logs": self.visit_logs
        }
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)

    def load_database(self) -> None:
        """Load database from file if present."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                    self.customers_db = data.get("customers_db", {})
                    self.visit_logs = data.get("visit_logs", [])
            except Exception:
                self.customers_db = {}
                self.visit_logs = []
        else:
            # Seed default demo consenting customers
            self._seed_demo_customers()

    def _seed_demo_customers(self) -> None:
        """Seed sample consenting demo customer faces into DB."""
        demo_profiles = [
            ("CUST_101", "Alice Johnson", "VIP Gold", (220, 180, 150)),
            ("CUST_102", "Bob Smith", "Silver", (150, 200, 180)),
            ("CUST_103", "Carol White", "Platinum", (180, 160, 220)),
        ]
        for cust_id, name, tier, color in demo_profiles:
            # Create synthetic demo facial image pattern
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.circle(img, (100, 100), 70, color, -1)
            cv2.circle(img, (75, 80), 12, (255, 255, 255), -1)
            cv2.circle(img, (125, 80), 12, (255, 255, 255), -1)
            cv2.ellipse(img, (100, 130), (35, 15), 0, 0, 180, (255, 255, 255), 3)
            
            face_roi = img[30:170, 30:170]
            enc = self.generate_face_encoding(face_roi)
            self.customers_db[cust_id] = {
                "customer_id": cust_id,
                "name": name,
                "encoding": enc,
                "loyalty_tier": tier,
                "registered_at": datetime.now().isoformat()
            }
        
        # Seed initial visit logs
        self.visit_logs = [
            {"customer_id": "CUST_101", "name": "Alice Johnson", "status": "Returning Customer", "loyalty_tier": "VIP Gold", "confidence": 94.5, "timestamp": "2026-07-29T10:15:00"},
            {"customer_id": "CUST_102", "name": "Bob Smith", "status": "Returning Customer", "loyalty_tier": "Silver", "confidence": 91.2, "timestamp": "2026-07-29T11:42:00"},
            {"customer_id": "GUEST_UNKNOWN", "name": "Guest", "status": "New Guest / Unregistered", "loyalty_tier": "None", "confidence": 42.0, "timestamp": "2026-07-29T12:05:00"},
            {"customer_id": "CUST_103", "name": "Carol White", "status": "Returning Customer", "loyalty_tier": "Platinum", "confidence": 96.8, "timestamp": "2026-07-29T14:30:00"},
        ]
        self.save_database()
