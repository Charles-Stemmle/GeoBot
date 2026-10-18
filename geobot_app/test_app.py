"""Light tests that need no TensorFlow, PyTorch or model files.

Run with:  python -m pytest test_app.py     (needs pytest and httpx)
"""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
import pipeline
from pipeline import InvalidImage


def png_bytes(size=(64, 48), color=(120, 80, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_combine_ranks_by_weight_and_normalises():
    predictions = {
        "holistic": {"France": 0.7, "Spain": 0.3},
        "road": {"France": 0.2, "Spain": 0.8},
        "environment": {"France": 0.5, "Spain": 0.5},
    }
    weights = {"holistic": 0.5, "road": 0.25, "environment": 0.25}
    scores = pipeline.combine(predictions, weights)
    assert abs(sum(scores.values()) - 1.0) < 1e-9
    assert pipeline.top_k(scores, 1)[0]["country"] == "France"


def test_combine_handles_country_missing_from_one_model():
    predictions = {
        "holistic": {"France": 0.6, "Chile": 0.4},
        "road": {"France": 1.0},
        "environment": {"France": 1.0},
    }
    scores = pipeline.combine(predictions, {"holistic": 0.5, "road": 0.25, "environment": 0.25})
    assert set(scores) == {"France", "Chile"}


def test_temperature_above_one_flattens():
    probs = np.array([0.8, 0.15, 0.05])
    flat = pipeline.apply_temperature(probs, 2.0)
    assert flat[0] < probs[0]
    assert abs(flat.sum() - 1.0) < 1e-9


def test_prepare_image_accepts_png_and_rejects_garbage():
    img = pipeline.prepare_image(png_bytes((2000, 1000)))
    assert max(img.size) == 640
    with pytest.raises(InvalidImage):
        pipeline.prepare_image(b"definitely not an image")


def test_overlay_is_transparent_outside_mask():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    url = pipeline.overlay_data_url(mask, (1, 2, 3))
    assert url.startswith("data:image/png;base64,")
    import base64
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    rgba = np.asarray(decoded)
    assert rgba[0, 0, 3] == 0 and rgba[1, 1, 3] == 255


class StubPipeline:
    def predict(self, data):
        pipeline.prepare_image(data)  # raises InvalidImage on junk
        return {"ok": True}


@pytest.fixture
def client():
    main.app.state.pipeline = StubPipeline()
    return TestClient(main.app)  # no `with`: skips model loading in lifespan


def test_endpoint_happy_path(client):
    r = client.post("/api/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_endpoint_rejects_wrong_type(client):
    r = client.post("/api/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_endpoint_rejects_corrupt_image(client):
    r = client.post("/api/predict", files={"file": ("a.png", b"not a png", "image/png")})
    assert r.status_code == 400


def test_endpoint_rejects_oversize(client, monkeypatch):
    monkeypatch.setattr(main.config, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/api/predict", files={"file": ("a.png", png_bytes((200, 200)), "image/png")})
    assert r.status_code == 413


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Geobot" in r.text
