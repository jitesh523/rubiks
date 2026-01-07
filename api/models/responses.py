"""Pydantic models for API responses"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SolutionResponse(BaseModel):
    """Cube solution response"""

    success: bool = Field(..., description="Whether solving was successful")
    solution: Optional[List[str]] = Field(default=None, description="List of moves to solve the cube")
    move_count: int = Field(..., description="Number of moves in solution")
    error: Optional[str] = Field(default=None, description="Error message if solving failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ColorPredictionResponse(BaseModel):
    """ML color prediction response"""

    color: str = Field(..., description="Predicted color name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    used_ml: bool = Field(..., description="Whether ML model was used (vs HSV fallback)")


class ModelInfoResponse(BaseModel):
    """ML model information"""

    is_trained: bool = Field(..., description="Whether model is trained and ready")
    accuracy: Optional[float] = Field(default=None, description="Model accuracy on test set")
    last_trained: Optional[str] = Field(default=None, description="Timestamp of last training")
    confidence_threshold: float = Field(..., description="Current confidence threshold")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional model metadata")


class TrainModelResponse(BaseModel):
    """Model training response"""

    success: bool = Field(..., description="Whether training was successful")
    accuracy: Optional[float] = Field(default=None, description="Model accuracy on test set")
    n_train: Optional[int] = Field(default=None, description="Number of training samples")
    n_test: Optional[int] = Field(default=None, description="Number of test samples")
    error: Optional[str] = Field(default=None, description="Error message if training failed")


class MoveExplanationResponse(BaseModel):
    """Move explanation response"""

    move: str = Field(..., description="The move in notation (e.g., R, U', F2)")
    explanation: str = Field(..., description="Human-readable explanation")
    direction_hint: Optional[str] = Field(default=None, description="Visual direction hint")
