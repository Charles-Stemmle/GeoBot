"""Inference pipeline: segmentation -> three classifiers -> consensus.

Heavy libraries (TensorFlow, PyTorch, transformers) are imported lazily inside
GeoPipeline so the helper functions can be imported and tested without them.
"""
from __future__ import annotations

import base64
import io
import threading
import time

import numpy as np
from PIL import Image, ImageOps

import config

Image.MAX_IMAGE_PIXELS = 40_000_000  # guard against decompression bombs

REGION_COLORS = {"environment": (34, 197, 94), "road": (245, 158, 11)}


class InvalidImage(ValueError):
    """The upload is not a usable JPEG or PNG."""


# --------------------------------------------------------------------------
# Pure helpers (no TensorFlow / PyTorch needed)
# --------------------------------------------------------------------------

def prepare_image(data: bytes) -> Image.Image:
    """Decode an upload and shrink it the same way the training images were."""
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ("JPEG", "PNG"):
            raise InvalidImage("Only JPEG and PNG images are supported.")
        img = ImageOps.exif_transpose(img).convert("RGB")
    except InvalidImage:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImage("That file could not be read as an image.") from exc

    img.thumbnail((config.MAX_SIDE, config.MAX_SIDE))
    return img


def mask_image(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Black out everything outside the mask (what the training masks look like)."""
    out = rgb.copy()
    out[~mask] = 0
    return out


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Equivalent to softmax(logits / T) when given softmax outputs."""
    if temperature == 1.0:
        return probs
    log_p = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()


def combine(predictions: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
    """Weighted average by country name. A model that doesn't know a country adds 0 for it."""
    scores: dict[str, float] = {}
    for model_name, per_country in predictions.items():
        weight = weights[model_name]
        for country, p in per_country.items():
            scores[country] = scores.get(country, 0.0) + weight * float(p)

    total = sum(scores.values())
    if total <= 0:
        raise ValueError("Consensus scores have no positive total.")
    return {country: score / total for country, score in scores.items()}


def top_k(scores: dict[str, float], k: int = config.TOP_K) -> list[dict]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
    return [{"country": c, "probability": float(p)} for c, p in ranked]


def _png_data_url(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def overlay_data_url(mask: np.ndarray, color: tuple[int, int, int]) -> str:
    """Opaque-colour PNG that is transparent outside the mask (opacity is set in the page)."""
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (*color, 255)
    return _png_data_url(rgba)


def photo_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

class GeoPipeline:
    def __init__(self):
        import tensorflow as tf
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        self._tf = tf
        self._torch = torch

        # Let TensorFlow share the GPU with PyTorch instead of grabbing it all.
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading segmenter {config.SEG_MODEL_NAME} on {self.device} ...")
        self.processor = AutoImageProcessor.from_pretrained(config.SEG_MODEL_NAME)
        self.segmenter = (
            AutoModelForSemanticSegmentation.from_pretrained(config.SEG_MODEL_NAME)
            .to(self.device)
            .eval()
        )
        self._print_class_names()

        self.models = {}
        self.labels = {}
        for name, (model_file, labels_file) in config.MODEL_FILES.items():
            model = tf.keras.models.load_model(config.MODEL_DIR / model_file)
            with open(config.MODEL_DIR / labels_file, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            n_outputs = model.output_shape[-1]
            if n_outputs != len(labels):
                raise RuntimeError(
                    f"{name}: model outputs {n_outputs} classes but {labels_file} "
                    f"has {len(labels)} labels."
                )
            self.models[name] = model
            self.labels[name] = labels
            print(f"Loaded {name} model ({len(labels)} countries)")

        self.lock = threading.Lock()  # one request on the models at a time
        self._settings_mtime = config.WEIGHTS_FILE.stat().st_mtime
        self._settings = config.load_settings()

    # -- startup sanity check ------------------------------------------------
    def _print_class_names(self):
        id2label = self.segmenter.config.id2label
        env = [id2label.get(i, "?") for i in config.ENVIRONMENT_IDS]
        road = [id2label.get(i, "?") for i in config.ROAD_IDS]
        print("Environment classes:", ", ".join(env))
        print("Road classes:", ", ".join(road))

    # -- settings (weights.json is re-read when it changes) ---------------------
    def _current_settings(self):
        try:
            mtime = config.WEIGHTS_FILE.stat().st_mtime
            if mtime != self._settings_mtime:
                self._settings_mtime = mtime
                self._settings = config.load_settings()
                print("Reloaded weights:", self._settings["weights"])
        except (OSError, ValueError, KeyError) as exc:
            print(f"[warn] could not reload weights.json ({exc}); keeping previous values")
        return self._settings

    # -- model steps --------------------------------------------------------
    def _segment(self, image: Image.Image) -> np.ndarray:
        """Same steps as the segmentation scripts: resize, upsample logits, argmax."""
        torch = self._torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.segmenter(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            return logits.argmax(dim=1)[0].cpu().numpy()

    def _classify(self, name: str, rgb: np.ndarray, temperature: float) -> dict[str, float]:
        tf = self._tf
        x = tf.image.resize(rgb, config.IMAGE_SIZE)
        # Same preprocessing the training scripts used (scales to [-1, 1]).
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        probs = self.models[name](tf.expand_dims(x, 0), training=False).numpy()[0]
        probs = apply_temperature(probs.astype(np.float64), temperature)
        return dict(zip(self.labels[name], map(float, probs)))

    # -- public entry point ----------------------------------------------------
    def predict(self, data: bytes) -> dict:
        started = time.perf_counter()
        image = prepare_image(data)
        rgb = np.asarray(image)
        settings = self._current_settings()

        with self.lock:
            classes = self._segment(image)
            masks = {
                "environment": np.isin(classes, config.ENVIRONMENT_IDS),
                "road": np.isin(classes, config.ROAD_IDS),
            }
            inputs = {
                "holistic": rgb,
                "road": mask_image(rgb, masks["road"]),
                "environment": mask_image(rgb, masks["environment"]),
            }
            predictions = {
                name: self._classify(name, arr, settings["temperature"][name])
                for name, arr in inputs.items()
            }

        consensus = combine(predictions, settings["weights"])
        coverage = {region: float(mask.mean()) for region, mask in masks.items()}

        warnings = []
        for region, fraction in coverage.items():
            if fraction < config.LOW_COVERAGE:
                warnings.append(
                    f"Only {fraction:.1%} of the image was labelled {region}, so the "
                    f"{region} model is mostly looking at a blank image and its guess is unreliable."
                )

        return {
            "width": image.width,
            "height": image.height,
            "image": photo_data_url(image),
            "regions": {
                region: {
                    "coverage": coverage[region],
                    "overlay": overlay_data_url(masks[region], REGION_COLORS[region]),
                }
                for region in masks
            },
            "predictions": {
                **{name: top_k(scores) for name, scores in predictions.items()},
                "consensus": top_k(consensus),
            },
            "weights": settings["weights"],
            "warnings": warnings,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
