# Simple Face Detection Server

I needed to do some face detection locally (no internet access for APIs) and the available options were surprisingly difficult to setup. So in this repo, I've put together a very simple face detection service using FastAPI based off of RetinaFace and backed by ResNet50.

To run:
`docker compose up --build`

To try it out:
Navigate to `localhost:8000/docs` and use the GUI to call the API.
