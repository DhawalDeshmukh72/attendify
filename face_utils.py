"""
Face recognition utilities for Attendify.

Uses DeepFace with the Facenet512 model, which gives strong accuracy on
standard face-verification benchmarks while still being practical to run
on CPU. Embeddings are 512-d vectors; we compare them with cosine
similarity against a tuned threshold.

If DeepFace / its model weights aren't available in the current
environment (e.g. no network access to download weights on first run),
every function here raises a clear RuntimeError that the Flask routes
turn into a friendly error message, instead of crashing the app.
"""
import base64
import io
import os

import numpy as np
from PIL import Image

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"   # fast, no extra downloads beyond the model itself

# Cosine-similarity threshold for "same person". Facenet512 embeddings are
# well separated; 0.68 is a reasonably strict cutoff that still tolerates
# lighting/angle variation from a webcam.
MATCH_THRESHOLD = 0.68

_deepface = None


def _get_deepface():
    """Lazily import DeepFace so the app can still start (e.g. for the
    dashboard/export pages) even if the ML stack isn't fully installed."""
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


def decode_base64_image(data_url: str) -> np.ndarray:
    """Turn a `data:image/...;base64,...` string from the browser webcam
    into an RGB numpy array."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img)


def get_embedding(image: np.ndarray):
    """Return a 512-d face embedding for the (single, largest) face found
    in `image`. Raises RuntimeError if no face is detected or DeepFace is
    unavailable."""
    DeepFace = _get_deepface()
    try:
        result = DeepFace.represent(
            img_path=image,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
            align=True,
        )
    except ValueError as exc:
        # DeepFace raises ValueError when no face is detected
        raise RuntimeError("No face detected in the image. Try again with better lighting and center your face.") from exc
    except ImportError as exc:
        raise RuntimeError(
            "Face recognition dependencies aren't installed. Run `pip install -r requirements.txt`."
        ) from exc

    if not result:
        raise RuntimeError("No face detected in the image.")

    # DeepFace.represent returns a list (one entry per detected face)
    return np.array(result[0]["embedding"], dtype=np.float32)


def get_multiple_embeddings(images: list):
    """Extract embeddings from multiple image numpy arrays."""
    embeddings = []
    errors = []
    for i, img in enumerate(images):
        try:
            emb = get_embedding(img)
            embeddings.append(emb)
        except Exception as e:
            errors.append(f"Angle #{i+1}: {str(e)}")

    if not embeddings:
        raise RuntimeError("Failed to detect faces in captured images: " + "; ".join(errors))
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_best_match(probe_embedding, students, threshold: float = MATCH_THRESHOLD):
    """Given a probe embedding and a list of Student rows, return
    (best_student_or_None, best_score). Compares against all student embeddings."""
    best_student = None
    best_score = -1.0
    for student in students:
        student_embeddings = student.get_embeddings()
        for emb in student_embeddings:
            score = cosine_similarity(probe_embedding, emb)
            if score > best_score:
                best_score = score
                best_student = student
    if best_score < threshold:
        return None, best_score
    return best_student, best_score

