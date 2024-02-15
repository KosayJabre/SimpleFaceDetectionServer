from pydantic import BaseModel
from typing import List


class Landmark(BaseModel):
    x: float
    y: float


class Face(BaseModel):
    bounding_box: List[float]
    landmarks: List[Landmark]
    confidence: float


class DetectorResponse(BaseModel):
    faces_detected: int
    faces: List[Face]


class FaceDetectionRequest(BaseModel):
    image_url: str


class FaceDetectionResponse(BaseModel):
    result: DetectorResponse
    time_taken: float
