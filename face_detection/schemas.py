from typing import List

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
    image_url: str


class FaceDetectionResponse(BaseModel):
    result: DetectorResponse
    time_taken: float
