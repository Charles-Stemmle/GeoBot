"""Settings for the Geobot web app.

Paths and segmentation class IDs live here. Ensemble weights live in
weights.json so you can tune them without touching code.
"""
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# By default the app folder sits inside your project, next to "models/".
MODEL_DIR = Path(os.environ.get("GEOBOT_MODEL_DIR", BASE_DIR.parent / "models"))
WEIGHTS_FILE = Path(os.environ.get("GEOBOT_WEIGHTS_FILE", BASE_DIR / "weights.json"))

# (model file, labels file) -- the names your training scripts save.
MODEL_FILES = {
    "holistic": ("geobot_country_classifier.keras", "country_labels.txt"),
    "road": ("geobot_road_classifier.keras", "road_country_labels.txt"),
    "environment": (
        "geobot_environment_classifier.keras",
        "environment_country_labels.txt",
    ),
}

# Must match the segmentation scripts that produced the training masks,
# otherwise the specialists see masks unlike the ones they were trained on.
SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
ENVIRONMENT_IDS = [4, 9, 16, 17, 29, 66, 72]
ROAD_IDS = [6, 11, 20, 43, 52, 80, 83, 87, 91, 93, 127, 136]

IMAGE_SIZE = (224, 224)  # classifier input size
MAX_SIDE = 640  # uploads are shrunk like the training images (thumbnail 640)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
TOP_K = 5
LOW_COVERAGE = 0.02  # warn when a mask covers less than 2% of the image


def load_settings(path=WEIGHTS_FILE):
    """Read weights.json. Weights are normalised to sum to 1."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    weights = {name: float(raw["weights"][name]) for name in MODEL_FILES}
    if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    total = sum(weights.values())
    weights = {name: w / total for name, w in weights.items()}

    temperature = {
        name: float(raw.get("temperature", {}).get(name, 1.0)) for name in MODEL_FILES
    }
    if any(t <= 0 for t in temperature.values()):
        raise ValueError("temperatures must be positive")

    return {"weights": weights, "temperature": temperature}
