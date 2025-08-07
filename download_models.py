#!/usr/bin/env python3
"""
Model download script for Hunyuan3D models.
This script downloads the required Hunyuan3D models and caches them locally.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional
import shutil

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install huggingface_hub transformers torch")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "tencent/hunyuan3d-2.0": {
        "name": "Hunyuan3D 2.0",
        "description": "Multiview model for 3D generation from multiple images",
        "repo_id": "tencent/Hunyuan3D-2",  # Actual HF repo
        "supports": ["multiview"],
        "required_files": [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt"
        ],
        "size_gb": 12.5
    },
    "tencent/hunyuan3d-2.1": {
        "name": "Hunyuan3D 2.1", 
        "description": "Single view model for 3D generation from text or single image",
        "repo_id": "tencent/Hunyuan3D-2",  # Using same repo for now
        "supports": ["text-to-3d", "image-to-3d"],
        "required_files": [
            "config.json",
            "model.safetensors", 
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt"
        ],
        "size_gb": 15.2
    }
}

def get_cache_dir() -> Path:
    """Get the model cache directory"""
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/runpod-volume/models")
    if not os.path.exists(cache_dir):
        # Fallback to local directory if runpod volume doesn't exist
        cache_dir = "./models"
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path

def check_disk_space(required_gb: float) -> bool:
    """Check if there's enough disk space"""
    cache_dir = get_cache_dir()
    stats = shutil.disk_usage(cache_dir)
    free_gb = stats.free / (1024**3)
    
    logger.info(f"Available disk space: {free_gb:.1f} GB")
    logger.info(f"Required space: {required_gb:.1f} GB")
    
    if free_gb < required_gb:
        logger.error(f"Insufficient disk space. Need {required_gb:.1f} GB, have {free_gb:.1f} GB")
        return False
    return True

def download_model(model_id: str, force_download: bool = False) -> bool:
    """
    Download a single model
    
    Args:
        model_id: Model identifier (e.g., 'tencent/hunyuan3d-2.1')
        force_download: Whether to re-download even if model exists
        
    Returns:
        True if successful, False otherwise
    """
    if model_id not in MODEL_CONFIGS:
        logger.error(f"Unknown model ID: {model_id}")
        logger.info(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return False
    
    config = MODEL_CONFIGS[model_id]
    cache_dir = get_cache_dir()
    model_dir = cache_dir / model_id.replace("/", "_")
    
    # Check if model already exists
    if model_dir.exists() and not force_download:
        logger.info(f"Model {model_id} already exists at {model_dir}")
        if verify_model_files(model_dir, config["required_files"]):
            logger.info(f"Model {model_id} is complete and ready to use")
            return True
        else:
            logger.warning(f"Model {model_id} is incomplete, re-downloading...")
            shutil.rmtree(model_dir)
    
    # Check disk space
    if not check_disk_space(config["size_gb"] + 2):  # +2GB buffer
        return False
    
    try:
        logger.info(f"Downloading {config['name']} ({config['size_gb']:.1f} GB)...")
        logger.info(f"Repository: {config['repo_id']}")
        logger.info(f"Destination: {model_dir}")
        
        # Download the model
        snapshot_download(
            repo_id=config["repo_id"],
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            cache_dir=str(cache_dir / ".cache"),
            ignore_patterns=[
                "*.md", "*.txt", "*.png", "*.jpg", "*.jpeg", 
                "*.git*", "**/.*", "**/__pycache__/**"
            ]
        )
        
        # Verify download
        if verify_model_files(model_dir, config["required_files"]):
            logger.info(f"✅ Successfully downloaded and verified {model_id}")
            
            # Create a model info file
            create_model_info(model_dir, model_id, config)
            return True
        else:
            logger.error(f"❌ Model download verification failed for {model_id}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error downloading {model_id}: {str(e)}")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        return False

def verify_model_files(model_dir: Path, required_files: List[str]) -> bool:
    """Verify that all required model files exist"""
    missing_files = []
    
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.warning(f"Missing files in {model_dir}: {missing_files}")
        return False
    
    return True

def create_model_info(model_dir: Path, model_id: str, config: dict) -> None:
    """Create a model info file for debugging"""
    import json
    from datetime import datetime
    
    info = {
        "model_id": model_id,
        "name": config["name"],
        "description": config["description"],
        "supports": config["supports"],
        "repo_id": config["repo_id"],
        "downloaded_at": datetime.utcnow().isoformat(),
        "size_gb": config["size_gb"],
        "files": [f.name for f in model_dir.iterdir() if f.is_file()]
    }
    
    info_file = model_dir / "model_info.json"
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)

def download_all_models(force_download: bool = False) -> bool:
    """Download all configured models"""
    logger.info("Starting download of all Hunyuan3D models...")
    
    total_size = sum(config["size_gb"] for config in MODEL_CONFIGS.values())
    if not check_disk_space(total_size + 5):  # +5GB buffer
        return False
    
    success_count = 0
    for model_id in MODEL_CONFIGS.keys():
        if download_model(model_id, force_download):
            success_count += 1
        else:
            logger.error(f"Failed to download {model_id}")
    
    logger.info(f"Downloaded {success_count}/{len(MODEL_CONFIGS)} models successfully")
    return success_count == len(MODEL_CONFIGS)

def list_models() -> None:
    """List available models and their status"""
    cache_dir = get_cache_dir()
    
    print("\n📦 Available Hunyuan3D Models:")
    print("=" * 60)
    
    for model_id, config in MODEL_CONFIGS.items():
        model_dir = cache_dir / model_id.replace("/", "_")
        status = "✅ Downloaded" if model_dir.exists() else "❌ Not Downloaded"
        
        print(f"Model: {config['name']}")
        print(f"  ID: {model_id}")
        print(f"  Size: {config['size_gb']:.1f} GB")
        print(f"  Status: {status}")
        print(f"  Description: {config['description']}")
        print(f"  Supports: {', '.join(config['supports'])}")
        if model_dir.exists():
            print(f"  Path: {model_dir}")
        print()

def cleanup_incomplete_downloads() -> None:
    """Clean up any incomplete or corrupted model downloads"""
    cache_dir = get_cache_dir()
    
    for model_id, config in MODEL_CONFIGS.items():
        model_dir = cache_dir / model_id.replace("/", "_")
        if model_dir.exists():
            if not verify_model_files(model_dir, config["required_files"]):
                logger.info(f"Cleaning up incomplete download: {model_dir}")
                shutil.rmtree(model_dir)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download Hunyuan3D models")
    parser.add_argument(
        "models", 
        nargs="*", 
        help="Specific models to download (default: all models)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List available models and their status"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--cleanup", 
        action="store_true", 
        help="Clean up incomplete downloads"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Override cache directory"
    )
    
    args = parser.parse_args()
    
    # Override cache directory if provided
    if args.cache_dir:
        os.environ["MODEL_CACHE_DIR"] = args.cache_dir
    
    if args.list:
        list_models()
        return
    
    if args.cleanup:
        cleanup_incomplete_downloads()
        return
    
    # Download models
    if args.models:
        # Download specific models
        success = True
        for model_id in args.models:
            if not download_model(model_id, args.force):
                success = False
        
        if not success:
            sys.exit(1)
    else:
        # Download all models
        if not download_all_models(args.force):
            sys.exit(1)
    
    logger.info("✅ Model download completed successfully!")
    logger.info("Models are ready for inference.")

if __name__ == "__main__":
    main()