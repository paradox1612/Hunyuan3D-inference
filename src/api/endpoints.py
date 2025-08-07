from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, BackgroundTasks
from typing import List, Optional, Dict, Any
import uuid
import os
from datetime import datetime, timezone
import json
import logging

from ..config import settings
from ..worker.tasks import process_prediction_task, get_prediction_status
from .schemas import (
    CreatePredictionRequest,
    PredictionResponse,
    Status,
    ModelType,
    PredictionInput
)

router = APIRouter(prefix="/v1", tags=["predictions"])
logger = logging.getLogger(__name__)

# In-memory store for predictions (in production, use Redis or a database)
predictions_store: Dict[str, Dict[str, Any]] = {}

@router.post(
    "/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new prediction"
)
async def create_prediction(
    request: CreatePredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new prediction job for 3D model generation.
    
    This endpoint follows the Synexa API specification for compatibility.
    """
    try:
        # Generate a unique prediction ID
        prediction_id = f"prediction-{str(uuid.uuid4())}"
        created_at = datetime.now(timezone.utc)
        
        # Determine model to use
        model = request.model
        
        # Create initial prediction record
        prediction = {
            "id": prediction_id,
            "model": model,
            "version": None,  # Could be set from model config
            "input": request.input.dict(),
            "status": Status.PROCESSING,
            "created_at": created_at.isoformat(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "output": None,
            "error": None,
            "logs": None,
            "metrics": None,
            "webhook": request.webhook,
            "webhook_events_filter": request.webhook_events_filter
        }
        
        # Store the prediction
        predictions_store[prediction_id] = prediction
        
        # Queue the background task - pass both model and input data
        task_data = {
            "model": model,
            "input": request.input.dict()
        }
        background_tasks.add_task(
            process_prediction_task,
            prediction_id=prediction_id,
            input_data=task_data
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error creating prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating prediction: {str(e)}"
        )

@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionResponse,
    summary="Get prediction status",
    responses={
        200: {"description": "Successful Response", "model": PredictionResponse},
        404: {"description": "Prediction not found"}
    }
)
async def get_prediction(prediction_id: str):
    """
    Get the status and results of a prediction by its ID.
    
    This endpoint follows the Synexa API specification for checking prediction status.
    Returns the current status, output (if completed), and any error messages.
    """
    # First check in-memory store (for development)
    prediction = predictions_store.get(prediction_id)
    
    # If not found, try Redis
    if not prediction:
        prediction = get_prediction_status(prediction_id)
        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Prediction {prediction_id} not found",
                        "type": "not_found",
                        "details": "The specified prediction ID does not exist or has expired"
                    }
                }
            )
    
    # Convert Redis hash to dict if needed
    if isinstance(prediction, dict) and not isinstance(prediction, dict):
        prediction = dict(prediction)
    
    # Ensure all datetime fields are properly formatted
    for time_field in ["created_at", "started_at", "completed_at"]:
        if time_field in prediction and prediction[time_field]:
            if not isinstance(prediction[time_field], str):
                prediction[time_field] = prediction[time_field].isoformat()
    
    # Handle output formatting
    if prediction.get("output") and isinstance(prediction["output"], str):
        try:
            prediction["output"] = json.loads(prediction["output"])
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse output for prediction {prediction_id}")
    
    return prediction

@router.get(
    "/models",
    summary="List available models"
)
async def list_models():
    """
    List all available models and their capabilities.
    """
    return {
        "models": [
            {
                "url": f"/v1/models/{model.value}",
                "id": model.value,
                "name": "Hunyuan3D 2.0" if "2.0" in model.value else "Hunyuan3D 2.1",
                "description": "Tencent's Hunyuan3D model for 3D generation",
                "capabilities": {
                    "multimodal_input": True,
                    "output_types": ["3d_model"],
                    "input_modes": ["text", "image"]
                },
                "settings": {
                    "defaults": {
                        "num_inference_steps": 50,
                        "guidance_scale": 7.5,
                        "height": 512,
                        "width": 512,
                        "num_images": 1
                    },
                    "constraints": {
                        "min_inference_steps": 10,
                        "max_inference_steps": 100,
                        "min_guidance_scale": 1.0,
                        "max_guidance_scale": 20.0,
                        "min_resolution": 256,
                        "max_resolution": 1024
                    }
                }
            } for model in ModelType
        ]
    }
