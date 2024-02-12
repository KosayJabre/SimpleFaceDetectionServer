from .face_detection import build_detector

import numpy as np
import cv2
import requests

from .schemas import DetectorResponse, Face


detector = build_detector(
    "RetinaNetMobileNetV1",
    max_resolution=1080
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


# # For testing
# def draw_faces(im, bboxes):
#     for bbox in bboxes:
#         x0, y0, x1, y1 = [int(_) for _ in bbox]
#         cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

# image_url = "https://i.imgur.com/o4FejFz.jpeg"
# cv2_image = download_image(image_url)
# results = detect_faces(cv2_image)
# draw_faces(cv2_image, [face.bounding_box for face in results.bounding_boxes])
# cv2.imshow("Image", cv2_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
