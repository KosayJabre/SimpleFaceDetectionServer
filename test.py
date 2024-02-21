from face_detection.service import detect_faces, download_image

from PIL import ImageDraw, ImageFont


def draw_faces(image, bboxes):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)


def draw_landmarks(image, faces_landmarks):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for landmarks in faces_landmarks:
        for landmark in landmarks:
            x, y = int(landmark.x), int(landmark.y)
            landmark_type = landmark.type.name
            radius = 2
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="lime")
            text_position = (x + radius + 2, y - radius)
            draw.text(text_position, landmark_type, fill="white", font=font)


if __name__ == "__main__":                
    image_url = "https://i.imgur.com/o4FejFz.jpeg"
    image = download_image(image_url)
    results = detect_faces([image, image])
    for result in results:
        draw_faces(image, [face.bounding_box for face in result.faces])
        draw_landmarks(image, [face.landmarks for face in result.faces])
        image.show()