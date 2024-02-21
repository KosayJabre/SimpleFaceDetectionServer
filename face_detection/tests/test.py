from PIL import Image

from ..service import detect_faces


def test_detect_faces_single_face():
    image = Image.open("single_face.gif")
    results = detect_faces([image])
    assert results[0].faces_count == 1


def test_detect_faces_many_faces():
    image = Image.open("many_faces.jpg")
    results = detect_faces([image])
    assert results[0].faces_count == 29


def test_detect_faces_no_faces():
    image = Image.open("no_faces.jpg")
    results = detect_faces([image])
    assert results[0].faces_count == 0


def test_detect_faces_diverse_group():
    image = Image.open("diverse_group.jpg")
    results = detect_faces([image])
    assert results[0].faces_count == 10


def test_detect_faces_cartoon():
    image = Image.open("cartoon.jpg")
    results = detect_faces([image])
    assert results[0].faces_count == 12


def test_detect_faces_surgical_mask():
    image = Image.open("surgical_mask.jpg")
    results = detect_faces([image])
    assert results[0].faces_count == 1


def test_detect_faces_mask():
    image = Image.open("mask.jpg")
    results = detect_faces([image])
    assert results[0].faces_count == 1
