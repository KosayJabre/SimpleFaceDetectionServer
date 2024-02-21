import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import FastAPI, File, Request, UploadFile
from slowapi import Limiter
from slowapi.util import get_remote_address

from .schemas import Base64FaceDetectionRequest, FaceDetectionResponse, UrlFaceDetectionRequest
from .service import base64_to_image, binary_to_image, detect_faces, download_image


app = FastAPI()
limiter = Limiter(key_func=get_remote_address)


@limiter.limit("10/second")
@app.post("/detect_faces_from_urls/", response_model=FaceDetectionResponse)
async def detect_faces_from_urls(request: Request, face_detection_request: UrlFaceDetectionRequest):
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(download_image, face_detection_request.images_url))

    results = detect_faces(images)

    return FaceDetectionResponse(result=results).model_dump()


@limiter.limit("10/second")
@app.post("/detect_faces_from_base64/", response_model=FaceDetectionResponse)
async def detect_faces_from_base64(request: Request, face_detection_request: Base64FaceDetectionRequest):
    images = [base64_to_image(image) for image in face_detection_request.images_base64]
    
    results = detect_faces(images)

    return FaceDetectionResponse(result=results).model_dump()


@limiter.limit("10/second")
@app.post("/detect_faces_from_binary/")
async def detect_faces_from_binary(request: Request, files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        image_data = await file.read()
        image = binary_to_image(image_data)
        images.append(image)
        await file.close()

    results = detect_faces(images)

    return FaceDetectionResponse(resul=results).model_dump()


@limiter.limit("10/second")
@app.get("/ready/")
async def ready(request: Request):
    return "Ready"
