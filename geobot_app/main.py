"""FastAPI app: serves the page and the /api/predict endpoint.

Run with:  uvicorn main:app
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import GeoPipeline, InvalidImage

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models load once at startup, not per request.
    if getattr(app.state, "pipeline", None) is None:
        app.state.pipeline = GeoPipeline()
    yield


app = FastAPI(title="Geobot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(request: Request, file: UploadFile = File(...)):
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Please upload a JPEG or PNG image.")

    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        limit_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Image is larger than {limit_mb} MB.")

    try:
        return request.app.state.pipeline.predict(data)
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc))
