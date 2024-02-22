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
    position: List[float]
    type: LandmarkType


class Face(BaseModel):
    bounding_box: List[float]
    landmarks: List[Landmark]
    confidence: float


class DetectorResponse(BaseModel):
    faces_count: int
    faces: List[Face]


class FaceDetectionRequest(BaseModel):
    confidence_threshold: float = 0.5
    clip_boxes: bool = True


class UrlFaceDetectionRequest(FaceDetectionRequest):
    images_url: List[str]


class Base64FaceDetectionRequest(FaceDetectionRequest):
    images_base64: List[str]


class FaceDetectionResponse(BaseModel):
    result: List[DetectorResponse]
