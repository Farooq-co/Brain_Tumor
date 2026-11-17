import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import inference
from .model_loader import PROJECT_DIR, store
from .schemas import AnalysisResponse, Health

log = logging.getLogger("brain_tumor_app")
MAX_BYTES = 15 * 1024 * 1024
ALLOWED_EXT = (".jpg", ".jpeg", ".png")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        store.load()
    except Exception:
        log.exception("Model loading failed; /api/health will report the error.")
    yield


app = FastAPI(title="Brain Tumor AI Analysis System", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _read_upload(file: UploadFile) -> bytes:
    if not store.ready:
        raise HTTPException(503, f"Models are not loaded. {store.error or ''}".strip())
    name = (file.filename or "").lower()
    if not name.endswith(ALLOWED_EXT):
        raise HTTPException(400, "Unsupported file type. Upload a JPG, JPEG or PNG image.")
    data = file.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (limit 15 MB).")
    return data


def _run(file: UploadFile, classify: bool, segment: bool):
    data = _read_upload(file)
    try:
        return inference.run(store, data, classify=classify, segment=segment)
    except inference.InvalidImage as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.exception("Inference failed")
        raise HTTPException(500, f"Inference failed: {exc}")


@app.get("/api/health", response_model=Health)
def health():
    return Health(
        status="ok" if store.ready else "unavailable",
        models_loaded=store.ready,
        segmentation_loaded=bool(store.ready and store.predictor.segmentation_model is not None),
        error=store.error,
    )


@app.get("/api/model-info")
def model_info():
    if not store.ready:
        raise HTTPException(503, "Models are not loaded.")
    return store.info()


@app.post("/api/predict/classify", response_model=AnalysisResponse, response_model_exclude_none=True)
def classify(file: UploadFile = File(...)):
    return _run(file, True, False)


@app.post("/api/predict/segment", response_model=AnalysisResponse, response_model_exclude_none=True)
def segment(file: UploadFile = File(...)):
    return _run(file, False, True)


@app.post("/api/predict/analyze", response_model=AnalysisResponse, response_model_exclude_none=True)
def analyze(file: UploadFile = File(...)):
    return _run(file, True, True)


app.mount("/", StaticFiles(directory=str(PROJECT_DIR / "frontend"), html=True), name="frontend")
