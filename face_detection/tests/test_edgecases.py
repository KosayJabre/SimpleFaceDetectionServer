import os

from PIL import Image

from face_detection.service import detect_faces

from . import IMAGE_DIR


def test_detect_faces_cartoon():
    image = Image.open(os.path.join(IMAGE_DIR, "cartoon.jpg"))
    results = detect_faces([image])
    assert results[0].faces_count == 12


def test_detect_faces_surgical_mask():
    image = Image.open(os.path.join(IMAGE_DIR, "surgical_mask.jpg"))
    results = detect_faces([image])
    assert results[0].faces_count == 1


def test_detect_faces_mask():
    image = Image.open(os.path.join(IMAGE_DIR, "mask.jpg"))
    results = detect_faces([image])
    assert results[0].faces_count == 1
