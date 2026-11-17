# Brain Tumor AI Analysis System

The app is in `website/`. Put `classifier.keras`, `segmentation.keras`,
`class_names.json`, and `test_model.py` in `website/results/`. You can override
that location with `BT_RESULTS_DIR`.

```
cd F:\ai_and_data_science\brain_tumors\website
python -m pip install -r backend\requirements.txt
python -m uvicorn backend.main:app --port 8000
```
Open http://127.0.0.1:8000

The backend imports `BrainTumorPredictor` from `results/test_model.py`, so loading,
preprocessing (RGB, Lanczos resize, raw 0-255), softmax probabilities, Grad-CAM and the
segmentation map are the original code. Mask threshold 0.5 and Reds overlay (alpha 0.55) match its display.
