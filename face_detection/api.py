import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from .schemas import Base64FaceDetectionRequest, FaceDetectionResponse, UrlFaceDetectionRequest
from .service import base64_to_image, detect_faces, download_image

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)


@limiter.limit("10/second")
@app.post("/detect_faces_from_urls/", response_model=FaceDetectionResponse)
def detect_faces_from_urls(request: Request, face_detection_request: UrlFaceDetectionRequest):
    start = time.perf_counter()

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(download_image, face_detection_request.images_url))

    results = detect_faces(images)

    time_taken = time.perf_counter() - start

    return FaceDetectionResponse(result=results, time_taken=time_taken).model_dump()


@limiter.limit("10/second")
@app.post("/detect_faces_from_base64/", response_model=FaceDetectionResponse)
def detect_faces_from_base64(request: Request, face_detection_request: Base64FaceDetectionRequest):
    start = time.perf_counter()

    images = [base64_to_image(image) for image in face_detection_request.images_base64]
    results = detect_faces(images)

    time_taken = time.perf_counter() - start

    return FaceDetectionResponse(result=results, time_taken=time_taken).model_dump()


@limiter.limit("10/second")
@app.get("/ready/")
def ready(request: Request):
    return "Ready"
