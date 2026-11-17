"""Loads both models once, reusing the loading + inference code from results/test_model.py."""
import os
import sys
import threading
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(
    os.environ.get("BT_RESULTS_DIR", PROJECT_DIR / "results")
)


class ModelStore:
    def __init__(self):
        self.predictor = None
        self.lock = threading.Lock()  # Keras models are not guaranteed thread-safe
        self.error = None

    def load(self):
        try:
            sys.path.insert(0, str(RESULTS_DIR))
            from test_model import BrainTumorPredictor  # the original pipeline

            config = RESULTS_DIR / "preprocessing_config.json"  # optional, as in test_model.py
            self.predictor = BrainTumorPredictor(
                str(RESULTS_DIR / "classifier.keras"),
                str(config) if config.is_file() else None,
                str(RESULTS_DIR / "class_names.json"),
            )
        except Exception as exc:  # surfaced via /api/health
            self.error = f"{type(exc).__name__}: {exc}"
            raise

    @property
    def ready(self):
        return self.predictor is not None

    def info(self):
        p = self.predictor
        seg = p.segmentation_model
        return {
            "classifier_file": "classifier.keras",
            "segmentation_file": "segmentation.keras",
            "class_names": p.class_names,
            "image_size": p.image_size,
            "classifier_input_shape": [str(d) for d in p.model.input_shape] if not isinstance(p.model.input_shape, list) else [str(s) for s in p.model.input_shape],
            "segmentation_loaded": seg is not None,
            "segmentation_input_shape": [str(d) for d in seg.input_shape] if seg is not None else None,
            "segmentation_threshold": 0.5,
        }


store = ModelStore()
