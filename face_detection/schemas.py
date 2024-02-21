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


class UrlFaceDetectionRequest(BaseModel):
    images_url: List[str]


class Base64FaceDetectionRequest(BaseModel):
    images_base64: List[str]


class FaceDetectionResponse(BaseModel):
    result: List[DetectorResponse]
    time_taken: float
