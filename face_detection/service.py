import base64
from io import BytesIO
from typing import List

import numpy as np
import requests
from PIL import Image

from face_detection.detector.detector import RetinaNetDetector

from .schemas import DetectorResponse, Face, Landmark, LandmarkType


detector = RetinaNetDetector(
    nms_iou_threshold=0.3,
)


def download_image(image_url: str) -> Image.Image:
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
        return image
    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
        return None


def base64_to_image(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    image_file = BytesIO(image_bytes)
    image = Image.open(image_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def binary_to_image(binary: bytes) -> Image.Image:
    image_file = BytesIO(binary)
    image = Image.open(image_file)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def detect_faces(images: List[Image.Image], confidence_threshold=0.5, clip_boxes=True) -> List[DetectorResponse]:
    images = np.array([np.array(image.convert("RGB"), dtype=np.float32) for image in images])

    batch_boxes, batch_landmarks = detector.detect(images, confidence_threshold=confidence_threshold, clip_boxes=clip_boxes)

    detector_responses = []
    for image_boxes, image_landmarks in zip(batch_boxes, batch_landmarks):
        faces = []
        for box, landmarks in zip(image_boxes, image_landmarks):
            face = Face(
                bounding_box=box[:4],
                confidence=box[4],
                landmarks=[Landmark(position=(landmark[0], landmark[1]), type=list(LandmarkType)[i]) for i, landmark in enumerate(landmarks)],
            )
            faces.append(face)
        detector_responses.append(DetectorResponse(faces_count=len(faces), faces=faces))

    return detector_responses
