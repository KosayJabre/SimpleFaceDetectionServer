from typing import List

import cv2
import numpy as np
import requests

from face_detection.model.detector import RetinaNetDetector
from face_detection.model.utils import get_device

from .schemas import DetectorResponse, Face, Landmark


detector = RetinaNetDetector(
    nms_iou_threshold=0.3,
    device=get_device(),
    max_resolution=None,
    fp16_inference=False,
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


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    # OpenCV reads images in BGR format for some reason
    return image[:, :, ::-1]


def detect_faces(images: List[np.ndarray]) -> List[DetectorResponse]:
    images = np.array([bgr_to_rgb(image) for image in images]) 
    
    batch_boxes, batch_landmarks = detector.detect(images)

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
