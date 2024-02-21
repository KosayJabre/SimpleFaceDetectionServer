from typing import List

import cv2
import numpy as np
import requests
from PIL import Image

from face_detection.model.detector import RetinaNetDetector
from face_detection.model.utils import get_device

from .schemas import DetectorResponse, Face, Landmark


detector = RetinaNetDetector(
    nms_iou_threshold=0.3,
    device=get_device(),
    max_resolution=None,
    fp16_inference=False,
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


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    # OpenCV reads images in BGR format for some reason
    return image[:, :, ::-1]


def detect_faces(images: List[Image.Image]) -> List[DetectorResponse]:
    images_np = np.array([np.array(image.convert('RGB'), dtype=np.float32) for image in images])

    batch_boxes, batch_landmarks = detector.detect(images_np)

    detector_responses = []
    for image_boxes, image_landmarks in zip(batch_boxes, batch_landmarks):
        faces = []
        for box, landmarks in zip(image_boxes, image_landmarks):
            bounding_box = box[:4]
            confidence = box[4]
            face = Face(
                bounding_box=bounding_box,
                confidence=confidence,
                landmarks=[
                    Landmark(x=landmark[0], y=landmark[1], type="landmark")
                    for landmark in landmarks
                ],
            )
            faces.append(face)
        detector_responses.append(DetectorResponse(faces_count=len(faces), faces=faces))

    return detector_responses
