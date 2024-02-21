import os

from PIL import Image

from face_detection.service import detect_faces

from . import IMAGE_DIR


def test_detect_faces_single_face():
    image = Image.open(os.path.join(IMAGE_DIR, "single_face.gif"))
    results = detect_faces([image])
    assert results[0].faces_count == 1


def test_detect_faces_many_faces():
    image = Image.open(os.path.join(IMAGE_DIR, "many_faces.jpg"))
    results = detect_faces([image])
    assert results[0].faces_count == 29


def test_detect_faces_no_faces():
    image = Image.open(os.path.join(IMAGE_DIR, "no_faces.jpg"))
    results = detect_faces([image])
    assert results[0].faces_count == 0


def test_detect_faces_diverse_group():
    image = Image.open(os.path.join(IMAGE_DIR, "diverse_group.jpg"))
    results = detect_faces([image])
    assert results[0].faces_count == 10
