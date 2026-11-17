"""Rendering of the segmentation probability map, matching test_model.py's display (Reds, threshold 0.5, alpha 0.55)."""
import numpy as np
from matplotlib import colormaps

THRESHOLD = 0.5
ALPHA = 0.55


def render_segmentation(image_rgb: np.ndarray, prob: np.ndarray):
    """image_rgb: HxWx3 uint8, prob: HxW float in [0,1]. Returns (mask_rgb, overlay_rgb, stats)."""
    binary = prob >= THRESHOLD
    mask_rgb = np.repeat((binary * 255).astype(np.uint8)[..., None], 3, axis=2)

    scaled = np.clip((prob - THRESHOLD) / (1 - THRESHOLD), 0, 1)
    red = (colormaps["Reds"](scaled)[..., :3] * 255).astype(np.float32)
    base = image_rgb.astype(np.float32)
    blended = np.where(binary[..., None], base * (1 - ALPHA) + red * ALPHA, base)
    overlay = np.clip(blended, 0, 255).astype(np.uint8)

    pixels = int(binary.sum())
    stats = {
        "threshold": THRESHOLD,
        "mask_pixels": pixels,
        "total_pixels": int(binary.size),
        "mask_area_percent": round(100.0 * pixels / binary.size, 3),
        "max_probability": float(prob.max()),
        "note": "Area is relative to the model-input image (pixels). No physical scale (mm) is available.",
    }
    return mask_rgb, overlay, stats


def render_gradcam(image_rgb: np.ndarray, heatmap: np.ndarray):
    """Grad-CAM overlay (jet, alpha 0.45) as in test_model.py. Shows classifier focus, not a tumor mask."""
    color = (colormaps["jet"](np.clip(heatmap, 0, 1))[..., :3] * 255).astype(np.float32)
    out = image_rgb.astype(np.float32) * 0.55 + color * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)
