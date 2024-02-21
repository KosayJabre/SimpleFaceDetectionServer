import pytest
from PIL import Image
from ..service import detect_faces


def test_detect_faces_single_face():
    image = Image.open("single_face.gif")
    results = detect_faces([image])
    assert len(results) == 1
    assert results[0].faces_count == 1


def test_detect_faces_multiple_faces():
    image = Image.open("many_faces.jpg")
    results = detect_faces([image])
    assert len(results) == 1
    assert results[0].faces_count == 29


def test_detect_faces_no_faces():
    image = Image.open("no_faces.jpg")
    results = detect_faces([image])
    assert len(results) == 1
    assert results[0].faces_count == 0