from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import FastAPI, File, Form, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from .schemas import Base64FaceDetectionRequest, FaceDetectionResponse, UrlFaceDetectionRequest
from .service import base64_to_image, binary_to_image, detect_faces, download_image


app = FastAPI(
    title="Simple Face Detection",
    description="Quick and easy face detection using RetinaNet backed by ResNet50. Supports URL, base64, and binary images as input.",
    version="0.1.0",
    contact={"name": "Kosay Jabre", "url": "https://kosayjabre.com"},
    license_info={"name": "Unlicense", "url": "https://unlicense.org"},
)


limiter = Limiter(key_func=get_remote_address)


@limiter.limit("10/second")
@app.post("/detect_faces_from_urls/", response_model=FaceDetectionResponse)
async def detect_faces_from_urls(request: Request, face_detection_request: UrlFaceDetectionRequest):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(download_image, face_detection_request.images_url))

    results = detect_faces(images, confidence_threshold=face_detection_request.confidence_threshold, clip_boxes=face_detection_request.clip_boxes)

    return FaceDetectionResponse(result=results).model_dump()


@limiter.limit("10/second")
@app.post("/detect_faces_from_base64/", response_model=FaceDetectionResponse)
async def detect_faces_from_base64(request: Request, face_detection_request: Base64FaceDetectionRequest):
    images = [base64_to_image(image) for image in face_detection_request.images_base64]

    results = detect_faces(images, confidence_threshold=face_detection_request.confidence_threshold, clip_boxes=face_detection_request.clip_boxes)

    return FaceDetectionResponse(result=results).model_dump()


@limiter.limit("10/second")
@app.post("/detect_faces_from_binary/")
async def detect_faces_from_binary(
    request: Request,
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Form(0.5),
    clip_boxes: bool = Form(True),
):
    images = []
    for file in files:
        image_data = await file.read()
        image = binary_to_image(image_data)
        images.append(image)
        await file.close()

    results = detect_faces(images, confidence_threshold=confidence_threshold, clip_boxes=clip_boxes)

    return FaceDetectionResponse(result=results).model_dump()


@limiter.limit("10/second")
@app.get("/ready/")
async def ready(request: Request):
    return "Ready"
