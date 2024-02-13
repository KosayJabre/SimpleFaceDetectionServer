from pydantic import BaseModel
from typing import List


class Face(BaseModel):
    bounding_box: List[float]
    confidence: float


class DetectorResponse(BaseModel):
    faces_detected: int
    bounding_boxes: List[Face]


class FaceDetectionRequest(BaseModel):
    image_url: str


class FaceDetectionResponse(BaseModel):
    result: DetectorResponse
    time_taken: float
