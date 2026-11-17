import base64
import io
import time

import numpy as np
from PIL import Image

from .segmentation import render_gradcam, render_segmentation

NO_TUMOR_LABELS = {"no_tumor", "normal", "notumor"}


class InvalidImage(ValueError):
    pass


def _png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _preprocess(store, data: bytes):
    # Exactly test_model.py: RGB -> resize(image_size, LANCZOS) -> float32 (0-255, no scaling).
    img_array, img_pil = store.predictor.preprocess_image(io.BytesIO(data))
    if img_array is None:
        raise InvalidImage("The file could not be read as an image.")
    return img_array, img_pil


def _classify(store, img_array):
    p = store.predictor
    batch = np.expand_dims(img_array, axis=0)
    with store.lock:
        probs, heatmap = p._predict_with_gradcam(batch)
    idx = int(np.argmax(probs))
    return {
        "predicted_class": p.class_names[idx],
        "predicted_index": idx,
        "confidence": float(probs[idx]),
        "probabilities": [
            {"label": p.class_names[i], "index": i, "probability": float(probs[i])}
            for i in range(len(p.class_names))
        ],
    }, heatmap


def _segment(store, img_array):
    p = store.predictor
    if p.segmentation_model is None:
        return None
    with store.lock:
        return p.predict_tumor_mask(img_array)


def run(store, data: bytes, classify=True, segment=True):
    t0 = time.perf_counter()
    img_array, img_pil = _preprocess(store, data)
    image_rgb = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)
    out = {"status": "completed", "image_size": store.predictor.image_size,
           "original_image": _png(image_rgb)}

    heatmap = None
    if classify:
        out["classification"], heatmap = _classify(store, img_array)
        out["gradcam_image"] = _png(render_gradcam(image_rgb, heatmap))

    if segment:
        prob = _segment(store, img_array)
        if prob is None:
            out["segmentation"] = {"available": False, "message": "segmentation.keras was not loaded."}
        else:
            mask, overlay, stats = render_segmentation(image_rgb, prob)
            out["segmentation"] = {"available": True, "mask_image": _png(mask),
                                   "overlay_image": _png(overlay), **stats}

    if classify and segment and out["segmentation"].get("available"):
        cls = out["classification"]["predicted_class"].lower()
        has_mask = out["segmentation"]["mask_pixels"] > 0
        if cls in NO_TUMOR_LABELS and has_mask:
            out["interpretation"] = ("The classifier predicted no tumor, but the segmentation model still marked a region. "
                                     "The two models can disagree; a mask is a model output, not proof of a tumor.")
        elif cls not in NO_TUMOR_LABELS and not has_mask:
            out["interpretation"] = ("The classifier predicted a tumor class, but the segmentation model marked no region "
                                     "at the 0.5 threshold. The two models can disagree.")
        else:
            out["interpretation"] = ("Both outputs are experimental model predictions and are not a medical diagnosis. "
                                     "A normal prediction does not rule out disease.")
    out["processing_time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
