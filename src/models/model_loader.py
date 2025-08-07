import os
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from .hunyuan3d import Hunyuan3DGenerator

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "tencent/hunyuan3d-2.0": {
        "name": "Hunyuan3D 2.0",
        "description": "Multiview model for 3D generation from multiple images",
        "supports": ["multiview"],
        "default_params": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
        }
    },
    "tencent/hunyuan3d-2.1": {
        "name": "Hunyuan3D 2.1",
        "description": "Single view model for 3D generation from text or single image",
        "supports": ["text-to-3d", "image-to-3d"],
        "default_params": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
        }
    }
}

def get_model_config(model_id: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model ID: {model_id}. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_id]

def list_available_models() -> Dict[str, Dict[str, Any]]:
    """List all available models and their configurations."""
    return MODEL_CONFIGS

def load_model(model_path: str, model_id: Optional[str] = None, device: str = None) -> Hunyuan3DGenerator:
    """
    Load a Hunyuan3D model from the specified path.
    
    Args:
        model_path: Path to the model directory
        model_id: Optional model ID (e.g., 'tencent/hunyuan3d-2.1')
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        An instance of Hunyuan3DGenerator
    """
    # Set default device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Verify model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    # If model_id is provided, validate against known configurations
    if model_id and model_id in MODEL_CONFIGS:
        logger.info(f"Loading {MODEL_CONFIGS[model_id]['name']} from {model_path}")
    
    # Initialize and load the model
    generator = Hunyuan3DGenerator(model_path, model_id=model_id, device=device)
    generator.load_model()
    
    return generator

def download_model(model_id: str, cache_dir: str) -> str:
    """
    Download a pretrained model from the Hugging Face Hub.
    
    Args:
        model_id: Model identifier (e.g., 'tencent/hunyuan3d-2.1')
        cache_dir: Directory to cache the downloaded model
        
    Returns:
        Path to the downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is required for model downloading. Install it with: pip install huggingface_hub")
        raise
    
    model_config = get_model_config(model_id)
    model_dir = os.path.join(cache_dir, model_id.replace("/", "_"))
    
    # Skip if already downloaded
    if os.path.exists(model_dir):
        logger.info(f"Using cached model at {model_dir}")
        return model_dir
    
    # Download the model
    logger.info(f"Downloading {model_id} to {model_dir}...")
    try:
        # Map to actual HuggingFace repo
        actual_repo = "tencent/Hunyuan3D-2" if "hunyuan3d" in model_id.lower() else model_id
        
        snapshot_download(
            repo_id=actual_repo,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", "*.txt", "*.png", "*.jpg"],  # Skip documentation files
            cache_dir=cache_dir
        )
        logger.info(f"Successfully downloaded {model_id} to {model_dir}")
        return model_dir
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {str(e)}")
        raise
