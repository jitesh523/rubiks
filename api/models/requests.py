"""Pydantic models for API requests"""

from pydantic import BaseModel, Field
from typing import List, Optional


class CubeStateRequest(BaseModel):
    """Request to solve a cube from string notation"""

    cube_string: str = Field(..., min_length=54, max_length=54, description="54-character cube state in URFDLB notation")
    use_ml_detection: bool = Field(default=False, description="Whether to use ML color detection")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cube_string": "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB",
                    "use_ml_detection": False
                }
            ]
        }
    }


class CubeFacesRequest(BaseModel):
    """Request to solve cube from face arrays"""

    faces: List[List[List[str]]] = Field(..., description="6 faces, each 3x3 grid of colors")
    use_ml_detection: bool = Field(default=True, description="Whether to use ML color detection")


class ColorPredictionRequest(BaseModel):
    """Request to predict color from RGB values"""

    rgb: List[int] = Field(..., min_length=3, max_length=3, description="RGB values [R, G, B]")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rgb": [200, 30, 30]
                }
            ]
        }
   }


class TrainModelRequest(BaseModel):
    """Request to train ML model"""

    training_data: List[dict] = Field(..., description="Training samples with rgb and color fields")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence threshold for predictions")
