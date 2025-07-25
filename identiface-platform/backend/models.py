from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Face(BaseModel):
    id: str
    path: str
    similarity: float = Field(ge=0.0, le=1.0)
    cluster_id: int
    original_image: str
    bounding_box: List[int] = Field(min_items=4, max_items=4)
    landmarks: List[List[float]] = []

class Cluster(BaseModel):
    id: int
    faces: List[Face]
    representative_face: Optional[Face]
    size: int

class Stats(BaseModel):
    totalFaces: int
    totalClusters: int
    totalPhotos: int
    lastUpdated: str
    status: str = "active"

class SearchRequest(BaseModel):
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)

class UploadResponse(BaseModel):
    message: str
    faces_detected: int
    status: str = "success"
    timestamp: str

class ProcessResponse(BaseModel):
    message: str
    total_faces: int
    new_clusters: int
    status: str = "success"
    timestamp: str
