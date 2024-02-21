from typing import List, Optional

from pydantic import BaseModel


class Landmark(BaseModel):
    x: float
    y: float
    type: str


class Face(BaseModel):
    bounding_box: List[float]
    landmarks: List[Landmark]
    confidence: float


class DetectorResponse(BaseModel):
    faces_count: int
    faces: List[Face]


class FaceDetectionRequest(BaseModel):
    images_url: Optional[List[str]]
    images_base64: Optional[List[str]]
    images_binary: Optional[List[bytes]]


class FaceDetectionResponse(BaseModel):
    result: List[DetectorResponse]
    time_taken: float
