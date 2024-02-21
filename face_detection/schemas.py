from enum import Enum
from typing import List

from pydantic import BaseModel


class LandmarkType(Enum):
    left_eye = "left_eye"
    right_eye = "right_eye"
    nose = "nose"
    left_mouth = "left_mouth"
    right_mouth = "right_mouth"


class Landmark(BaseModel):
    x: float
    y: float
    type: LandmarkType


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
