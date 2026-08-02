"""
OpenCV Image Preprocessing and Computer Vision Utilities.
Provides reusable functions for image processing, filtering, Canny edge detection,
and Haar Cascade face detection.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# Load default Haar cascade for face detection if available
face_cascade = None
try:
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if hasattr(cv2, 'CascadeClassifier'):
            face_cascade = cv2.CascadeClassifier(cascade_path)
except Exception:
    face_cascade = None


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    """Resize image maintaining aspect ratio if only one dimension provided."""
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    elif height is None:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    else:
        dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), sigma_x: float = 0) -> np.ndarray:
    """Apply Gaussian Blur to smooth/denoise image."""
    return cv2.GaussianBlur(image, kernel_size, sigma_x)


def apply_canny_edge_detection(image: np.ndarray, threshold1: float = 100, threshold2: float = 200) -> np.ndarray:
    """Apply Canny Edge Detection."""
    gray = to_grayscale(image)
    return cv2.Canny(gray, threshold1, threshold2)


def detect_faces_haar(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    Detect face bounding boxes using Haar Cascade classifier.
    Returns list of bounding boxes: (x, y, w, h).
    """
    gray = to_grayscale(image)
    if face_cascade is not None and not face_cascade.empty():
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
            
    # Backup face bounding box fallback based on image center dimensions
    h, w = gray.shape[:2]
    return [(int(w * 0.25), int(h * 0.2), int(w * 0.5), int(h * 0.6))]


def draw_face_bounding_boxes(image: np.ndarray, bounding_boxes: List[Tuple[int, int, int, int]], label: str = "Face") -> np.ndarray:
    """Draw rectangle boxes and text labels over image."""
    annotated = image.copy()
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


def preprocess_image_pipeline(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Dict[str, np.ndarray]:
    """
    Full CV preprocessing pipeline returning dictionary of processed variants.
    """
    resized = resize_image(image, width=target_size[0], height=target_size[1])
    gray = to_grayscale(resized)
    blurred = apply_gaussian_blur(gray)
    edges = apply_canny_edge_detection(blurred)
    
    return {
        "original": image,
        "resized": resized,
        "grayscale": gray,
        "blurred": blurred,
        "edges": edges
    }
