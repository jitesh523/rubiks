"""
ML Model API routes

Endpoints for ML color detection model management.
"""

from fastapi import APIRouter, HTTPException

from api.models.requests import ColorPredictionRequest, TrainModelRequest
from api.models.responses import (
    ColorPredictionResponse,
    ModelInfoResponse,
    TrainModelResponse,
)
from api.services.ml_service import ml_service

router = APIRouter()


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get ML model status and metadata"""
    info = ml_service.get_model_info()
    return ModelInfoResponse(
        is_trained=info["is_trained"],
        accuracy=info.get("accuracy"),
        last_trained=info.get("last_trained"),
        confidence_threshold=info["confidence_threshold"],
        metadata=info.get("metadata", {}),
    )


@router.post("/predict", response_model=ColorPredictionResponse)
async def predict_color(request: ColorPredictionRequest):
    """Predict color from RGB values using ML model"""
    try:
        result = await ml_service.predict_color(tuple(request.rgb))
        return ColorPredictionResponse(
            color=result["color"],
            confidence=result["confidence"],
            used_ml=result["used_ml"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/train", response_model=TrainModelResponse)
async def train_model(request: TrainModelRequest):
    """Train ML model with provided training data"""
    try:
        result = await ml_service.train_model(request.training_data, request.confidence_threshold)

        return TrainModelResponse(
            success=True,
            accuracy=result.get("accuracy"),
            n_train=result.get("n_train"),
            n_test=result.get("n_test"),
            error=None,
        )
    except Exception as e:
        return TrainModelResponse(
            success=False, accuracy=None, n_train=None, n_test=None, error=str(e)
        )
