import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from celery import current_task
import redis

from .celery_app import app
from ..models.model_loader import load_model, get_model_config
from ..models.hunyuan3d import Hunyuan3DGenerator
from ..config import settings
from ..api.schemas import Status, ModelType, PredictionResponse

logger = logging.getLogger(__name__)

# Global model instances
models = {}

# Redis connection
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True
)

def get_model(model_id: str) -> Hunyuan3DGenerator:
    """Get or load the specified Hunyuan3D model"""
    global models
    
    if model_id not in models:
        logger.info(f"Loading model: {model_id}")
        model_path = os.path.join(settings.MODEL_CACHE_DIR, model_id.replace("/", "_"))
        models[model_id] = load_model(model_path, model_id=model_id)
        
    return models[model_id]

def update_prediction_status(
    prediction_id: str, 
    status: Status, 
    output: Optional[Dict] = None, 
    error: Optional[str] = None,
    metrics: Optional[Dict] = None
) -> None:
    """Update the status of a prediction in Redis"""
    try:
        prediction_key = f"prediction:{prediction_id}"
        current_time = datetime.now(timezone.utc).isoformat()
        
        update_data = {
            "status": status,
            "updated_at": current_time
        }
        
        if status == Status.SUCCEEDED and output:
            update_data["output"] = output
            update_data["completed_at"] = current_time
        elif status == Status.FAILED and error:
            update_data["error"] = error
            update_data["completed_at"] = current_time
        
        if metrics:
            update_data["metrics"] = metrics
        
        # Update in Redis
        redis_client.hmset(prediction_key, update_data)
        
        # Set TTL for the prediction (7 days)
        redis_client.expire(prediction_key, 60 * 60 * 24 * 7)
        
        logger.info(f"Updated prediction {prediction_id} status to {status}")
        
    except Exception as e:
        logger.error(f"Error updating prediction status: {str(e)}", exc_info=True)

def get_prediction_status(prediction_id: str) -> Optional[Dict]:
    """Get the status of a prediction from Redis"""
    try:
        prediction = redis_client.hgetall(f"prediction:{prediction_id}")
        return prediction if prediction else None
    except Exception as e:
        logger.error(f"Error getting prediction status: {str(e)}", exc_info=True)
        return None

@app.task(bind=True, name="process_prediction_task")
def process_prediction_task(self, prediction_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a prediction task"""
    task_id = self.request.id
    logger.info(f"Starting prediction task {task_id} for prediction {prediction_id}")
    
    try:
        # Update status to processing
        update_prediction_status(
            prediction_id=prediction_id,
            status=Status.PROCESSING,
            metrics={"started_at": datetime.now(timezone.utc).isoformat()}
        )
        
        # Get model info from input data - now model is at root level
        model_id = input_data.get("model", ModelType.HUNYUAN3D_2_1)
        
        # If model is still nested in input, get it from there
        if isinstance(input_data.get("input"), dict) and "model" in input_data["input"]:
            model_id = input_data["input"]["model"]
            
        model_config = get_model_config(model_id)
        
        # Load the model
        model = get_model(model_id)
        
        # Prepare output directory
        output_dir = os.path.join(settings.OUTPUT_DIR, prediction_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate 3D model
        start_time = time.time()
        
        # TODO: Implement actual model inference
        # For now, just save a placeholder
        output_files = []
        output_path = os.path.join(output_dir, "output.glb")
        with open(output_path, "w") as f:
            f.write("PLACEHOLDER - 3D MODEL OUTPUT")
        output_files.append(output_path)
        
        # Upload to R2 if configured
        output_urls = []
        if settings.R2_BUCKET_NAME:
            from ..utils.storage import upload_to_r2
            for i, file_path in enumerate(output_files):
                file_name = os.path.basename(file_path)
                url = upload_to_r2(
                    local_path=file_path,
                    key=f"generations/{prediction_id}/{file_name}",
                    content_type="model/gltf-binary"
                )
                if url:
                    output_urls.append(url)
        
        # Calculate metrics
        duration = time.time() - start_time
        metrics = {
            "inference_time": duration,
            "model": model_id,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Prepare output
        output = {
            "output": output_urls,
            "metadata": {
                "model": model_id,
                "inference_time": duration,
                "output_format": "glb"
            }
        }
        
        # Update status to completed
        update_prediction_status(
            prediction_id=prediction_id,
            status=Status.SUCCEEDED,
            output=output,
            metrics=metrics
        )
        
        return {
            "status": "succeeded",
            "prediction_id": prediction_id,
            "output": output,
            "metrics": metrics
        }
        
    except Exception as e:
        error_msg = f"Error in prediction task: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update status to failed
        update_prediction_status(
            prediction_id=prediction_id,
            status=Status.FAILED,
            error=error_msg,
            metrics={
                "error": str(e),
                "failed_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        return {
            "status": "failed",
            "prediction_id": prediction_id,
            "error": error_msg
        }
