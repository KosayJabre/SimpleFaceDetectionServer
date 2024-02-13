from src.face_detection.detect import RetinaNetDetector
from src.face_detection.utils import get_device

import numpy as np
import cv2
import requests

from .schemas import DetectorResponse, Face


detector = RetinaNetDetector(
    confidence_threshold=0.5,
    nms_iou_threshold=0.3,
    device=get_device(),
    max_resolution=None,
    fp16_inference=False,
    clip_boxes=False,
)


def download_image(image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        return image
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
        return None


def detect_faces(image: np.ndarray) -> DetectorResponse:
    results = detector.detect(image[:, :, ::-1])
    faces = []
    for result in results:
        bounding_box = result[:4]
        confidence = result[4]
        face = Face(bounding_box=bounding_box, confidence=confidence)
        faces.append(face)
    return DetectorResponse(faces_detected=len(faces), bounding_boxes=faces)
