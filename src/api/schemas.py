from pydantic import BaseModel, Field, AnyHttpUrl
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

class ModelType(str, Enum):
    HUNYUAN3D_2_0 = "tencent/hunyuan3d-2.0"
    HUNYUAN3D_2_1 = "tencent/hunyuan3d-2.1"

class Status(str, Enum):
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

class PredictionInput(BaseModel):
    seed: int = 1234
    image: Optional[Union[str, AnyHttpUrl]] = None
    steps: int = 5
    caption: str = ""
    shape_only: bool = True
    guidance_scale: float = 5.5
    check_box_rembg: bool = True
    octree_resolution: str = "256"
    multiple_views: List[Union[str, AnyHttpUrl]] = []

class PredictionOutput(BaseModel):
    output: List[AnyHttpUrl]
    metadata: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    id: str
    model: str
    version: Optional[str] = None
    input: Dict[str, Any]
    output: Optional[PredictionOutput] = None
    status: Status
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    logs: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    webhook: Optional[AnyHttpUrl] = None
    webhook_events_filter: Optional[List[str]] = None

class CreatePredictionRequest(BaseModel):
    model: ModelType = ModelType.HUNYUAN3D_2_1
    input: PredictionInput
    webhook: Optional[AnyHttpUrl] = None
    webhook_events_filter: Optional[List[str]] = None
