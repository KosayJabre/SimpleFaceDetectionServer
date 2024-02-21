from face_detection.service import detect_faces, download_image

import cv2


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


def draw_landmarks(im, faces_landmarks):
    for landmarks in faces_landmarks:
        for landmark in landmarks:
            x, y = int(landmark.x), int(landmark.y)
            cv2.circle(im, (x, y), 2, (0, 255, 0), -1)


if __name__ == "__main__":                
    image_url = "https://i.imgur.com/o4FejFz.jpeg"
    cv2_image = download_image(image_url)
    results = detect_faces(cv2_image)
    draw_faces(cv2_image, [face.bounding_box for face in results.faces])
    draw_landmarks(cv2_image, [face.landmarks for face in results.faces])
    cv2.imshow("Image", cv2_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()