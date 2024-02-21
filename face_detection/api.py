import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from .schemas import FaceDetectionRequest, FaceDetectionResponse
from .service import detect_faces, download_image

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)


@limiter.limit("60/minute")
@limiter.limit("5/second")
@app.post("/detect_faces/", response_model=FaceDetectionResponse)
def detect_faces_post(request: Request, face_detection_request: FaceDetectionRequest):
    start = time.perf_counter()

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(download_image, face_detection_request.images_url))

    results = detect_faces(images)

    time_taken = time.perf_counter() - start

    return FaceDetectionResponse(result=results, time_taken=time_taken).model_dump()


@limiter.limit("10/second")
@app.get("/ready/")
def ready(request: Request):
    return "Ready"
