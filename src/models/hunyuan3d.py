import torch
import numpy as np
from typing import Optional, List, Dict, Any, Union
import os
from pathlib import Path
import logging
from datetime import datetime
import trimesh
import tempfile
from PIL import Image

logger = logging.getLogger(__name__)

class Hunyuan3DGenerator:
    """Wrapper class for Hunyuan3D model inference"""
    
    def __init__(self, model_path: str, model_id: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Hunyuan3D generator.
        
        Args:
            model_path: Path to the pretrained model
            model_id: Model identifier (e.g., 'tencent/hunyuan3d-2.1')
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.model_id = model_id or "tencent/hunyuan3d-2.1"
        self.device = device
        self.pipeline = None
        self.rmbg_worker = None
        self.is_multiview = "2.0" in self.model_id
        
    def load_model(self):
        """Load the Hunyuan3D model"""
        try:
            logger.info(f"Loading Hunyuan3D model {self.model_id} from {self.model_path}")
            
            # Import required modules
            from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
            from hy3dshape.rembg import BackgroundRemover
            
            # Load background remover
            self.rmbg_worker = BackgroundRemover()
            
            # Determine subfolder based on model
            if "2.0" in self.model_id:
                subfolder = "hunyuan3d-dit-v2-0-mv"
            elif "2.1" in self.model_id:
                subfolder = "hunyuan3d-dit-v2-1"
            else:
                subfolder = "hunyuan3d-dit-v2-1"  # Default
            
            # Load the pipeline
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.model_path,
                subfolder=subfolder,
                use_safetensors=False,
                device=self.device,
            )
            
            logger.info(f"Hunyuan3D model loaded successfully: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Error loading Hunyuan3D model: {str(e)}")
            raise
    
    def generate(
        self,
        image: Union[Image.Image, str, Dict[str, Image.Image]] = None,
        seed: int = 1234,
        steps: int = 30,
        guidance_scale: float = 7.5,
        octree_resolution: int = 256,
        check_box_rembg: bool = True,
        num_chunks: int = 200000,
        caption: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate 3D assets using the Hunyuan3D model.
        
        Args:
            image: Input image(s) - PIL Image, path string, or dict for multiview
            seed: Random seed for reproducibility
            steps: Number of inference steps
            guidance_scale: Guidance scale for diffusion
            octree_resolution: Resolution for octree representation
            check_box_rembg: Whether to remove background
            num_chunks: Number of chunks for processing
            caption: Text caption (for text-to-3D, not implemented)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing the generated 3D assets and metadata
        """
        if self.pipeline is None:
            self.load_model()
            
        if image is None and caption is None:
            raise ValueError("Either image or caption must be provided")
            
        try:
            # Handle image input
            processed_image = self._process_input_image(image, check_box_rembg)
            
            # Set up generator
            generator = torch.Generator(device=self.device)
            generator = generator.manual_seed(int(seed))
            
            # Generate 3D model
            with torch.inference_mode():
                outputs = self.pipeline(
                    image=processed_image,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    output_type='mesh'
                )
            
            return {
                "status": "success",
                "outputs": outputs,
                "model": self.model_id,
                "parameters": {
                    "seed": seed,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "octree_resolution": octree_resolution,
                    "check_box_rembg": check_box_rembg,
                    "num_chunks": num_chunks,
                    "caption": caption
                }
            }
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _process_input_image(self, image, remove_bg=True):
        """Process input image(s) for model inference"""
        if isinstance(image, str):
            # Load image from path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, dict):
            # Multi-view mode - process each view
            processed_images = {}
            for view_name, img in image.items():
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                if remove_bg:
                    img = self.rmbg_worker(img.convert('RGB'))
                processed_images[view_name] = img
            return processed_images
        
        # Single image processing
        if remove_bg and (image.mode == "RGB" or remove_bg):
            image = self.rmbg_worker(image.convert('RGB'))
            
        return image
    
    def save_outputs(self, outputs: Dict[str, Any], output_dir: str) -> List[str]:
        """
        Save the generated 3D assets to disk.
        
        Args:
            outputs: Output dictionary from generate()
            output_dir: Directory to save the outputs
            
        Returns:
            List of saved file paths
        """
        if outputs.get("status") != "success":
            raise ValueError("Cannot save failed generation outputs")
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        try:
            # Import export function
            from hy3dshape.pipelines import export_to_trimesh
            
            # Convert to trimesh
            meshes = export_to_trimesh(outputs["outputs"])
            
            # Save each mesh
            for i, mesh in enumerate(meshes):
                output_path = os.path.join(output_dir, f"output_{i}.glb")
                mesh.export(output_path, file_format="glb")
                saved_files.append(output_path)
            
            # Save metadata
            import json
            metadata = {
                "model": outputs["model"],
                "parameters": outputs["parameters"],
                "generated_at": datetime.utcnow().isoformat(),
                "saved_files": saved_files,
                "mesh_stats": {
                    "num_faces": mesh.faces.shape[0] if len(meshes) > 0 else 0,
                    "num_vertices": mesh.vertices.shape[0] if len(meshes) > 0 else 0
                }
            }
            
            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving outputs: {str(e)}")
            raise
    
    def export_mesh(self, outputs: Dict[str, Any], output_path: str, file_format: str = "glb") -> str:
        """
        Export a single mesh to specified path and format.
        
        Args:
            outputs: Output dictionary from generate()
            output_path: Full path for the output file
            file_format: File format ('glb', 'obj', 'ply', 'stl')
            
        Returns:
            Path to the exported file
        """
        if outputs.get("status") != "success":
            raise ValueError("Cannot export failed generation outputs")
            
        try:
            from hy3dshape.pipelines import export_to_trimesh
            
            # Convert to trimesh
            meshes = export_to_trimesh(outputs["outputs"])
            
            if not meshes:
                raise ValueError("No meshes to export")
                
            # Export the first mesh
            mesh = meshes[0]
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export with appropriate settings
            if file_format.lower() in ['glb', 'obj']:
                mesh.export(output_path, include_normals=True)
            else:
                mesh.export(output_path)
                
            logger.info(f"Mesh exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting mesh: {str(e)}")
            raise
    
    def get_mesh_stats(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the generated mesh"""
        if outputs.get("status") != "success":
            return {"error": "No successful generation to analyze"}
            
        try:
            from hy3dshape.pipelines import export_to_trimesh
            meshes = export_to_trimesh(outputs["outputs"])
            
            if not meshes:
                return {"error": "No meshes found"}
                
            mesh = meshes[0]
            return {
                "num_vertices": len(mesh.vertices),
                "num_faces": len(mesh.faces),
                "watertight": mesh.is_watertight,
                "volume": float(mesh.volume),
                "surface_area": float(mesh.area),
                "bounds": mesh.bounds.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error getting mesh stats: {str(e)}")
            return {"error": str(e)}
