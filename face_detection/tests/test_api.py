import base64
import io
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from face_detection.api import app

from . import IMAGE_DIR


client = TestClient(app)


@pytest.fixture(scope="session")
def test_image():
    image = Image.open(os.path.join(IMAGE_DIR, "many_faces.jpg"))
    return image


@pytest.mark.asyncio
async def test_detect_faces_from_files(test_image):
    image_bytes = io.BytesIO()
    test_image.save(image_bytes, format="JPEG")

    response = client.post(
        "/detect_faces_from_files/",
        files={"files": ("single_face.gif", image_bytes, "image/gif")},
    )

    assert response.status_code == 200
    assert response.json()["result"][0]["faces_count"] == 29


@pytest.mark.asyncio
async def test_detect_faces_from_base64(test_image):
    image_bytes = io.BytesIO()
    test_image.save(image_bytes, format="JPEG")
    image_base64_str = base64.b64encode(image_bytes.getvalue()).decode("utf-8")

    response = client.post(
        "/detect_faces_from_base64/",
        json={"images_base64": [image_base64_str], "confidence_threshold": 0.5, "clip_boxes": True},
    )

    assert response.status_code == 200
    assert response.json()["result"][0]["faces_count"] == 29


@pytest.mark.asyncio
@patch("face_detection.api.download_image")
async def test_detect_faces_from_urls(mock_download, test_image):
    mock_download.return_value = test_image

    response = client.post(
        "/detect_faces_from_urls/",
        json={"images_url": ["http://example.com/some_image.jpg"], "confidence_threshold": 0.5, "clip_boxes": True},
    )
    assert response.status_code == 200
    assert response.json()["result"][0]["faces_count"] == 29
