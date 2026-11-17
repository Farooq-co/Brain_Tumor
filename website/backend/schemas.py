from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Health(BaseModel):
    status: str
    models_loaded: bool
    segmentation_loaded: bool
    error: Optional[str] = None


class ClassProbability(BaseModel):
    label: str
    index: int
    probability: float


class Classification(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: List[ClassProbability]


class AnalysisResponse(BaseModel):
    status: str
    image_size: int
    original_image: str
    classification: Optional[Classification] = None
    gradcam_image: Optional[str] = None
    segmentation: Optional[Dict[str, Any]] = None
    interpretation: Optional[str] = None
    processing_time_ms: float
    disclaimer: str = ("Experimental research output. Not a medical diagnosis; "
                       "consult a qualified clinician.")
