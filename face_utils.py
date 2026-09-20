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
import threading

import numpy as np
from PIL import Image

# Suppress TF verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"   # fast, lightweight CPU detector

# Cosine-similarity threshold for "same person".
MATCH_THRESHOLD = 0.68

_deepface = None
_model_lock = threading.Lock()
_warmed_up = False


def _get_deepface():
    """Lazily import DeepFace and configure TensorFlow CPU threads."""
    global _deepface
    if _deepface is None:
        import tensorflow as tf
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(2)
        except Exception:
            pass
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


def warmup_model():
    """Pre-build and cache the Facenet512 model into memory."""
    global _warmed_up
    if _warmed_up:
        return
    try:
        DeepFace = _get_deepface()
        with _model_lock:
            # Build and cache model weights
            DeepFace.build_model(MODEL_NAME)
            _warmed_up = True
    except Exception as e:
        print(f"[WARMUP WARNING] Could not pre-warm model: {e}")


def decode_base64_image(data_url: str, max_dimension: int = 640) -> np.ndarray:
    """Turn a `data:image/...;base64,...` string from webcam into an RGB numpy array,
    downsampling large frames to conserve server RAM and boost inference speed."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    
    # Downscale if larger than max_dimension to protect RAM on cloud servers
    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return np.array(img)


def get_embedding(image: np.ndarray):
    """Return a 512-d face embedding for the detected face.
    Thread-locked to prevent concurrent memory spikes on cloud instances."""
    DeepFace = _get_deepface()
    
    # Downsample numpy image if larger than 640px
    if max(image.shape[0], image.shape[1]) > 640:
        pil_img = Image.fromarray(image)
        scale = 640.0 / float(max(image.shape[0], image.shape[1]))
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = np.array(pil_img.resize(new_size, Image.Resampling.LANCZOS))

    try:
        with _model_lock:
            result = DeepFace.represent(
                img_path=image,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
                align=True,
            )
    except ValueError as exc:
        raise RuntimeError("No face detected in the image. Try again with better lighting and center your face.") from exc
    except ImportError as exc:
        raise RuntimeError(
            "Face recognition dependencies aren't installed. Run `pip install -r requirements.txt`."
        ) from exc
    except Exception as exc:
        if "face" in str(exc).lower() or "detect" in str(exc).lower():
            raise RuntimeError("No face detected in the frame. Please look directly at the camera.") from exc
        raise RuntimeError(f"Biometric processing error: {str(exc)}") from exc

    if not result:
        raise RuntimeError("No face detected in the image.")

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

