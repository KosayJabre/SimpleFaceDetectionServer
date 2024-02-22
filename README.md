# Simple Face Detection Server
![Example bounding boxes around faces](/example.png)

An easy to use face detection API based on the [RetinaFace architecture](https://arxiv.org/abs/1905.00641) with a ResNet50 backbone. 

Influenced by [biubug6's implementation](https://github.com/biubug6/Pytorch_Retinaface) and optimized for inference and simplicity of integration.

## Setup
### Docker Server
All the dependencies have been packaged into a Dockerfile. To run the server inside Docker, you can:
1) `docker build . -t face_detection` to build the image.
2) `docker run face_detection` to run the image and start the server.

Alternatively, just run `docker compose up --build`.

### Local Server
If you want to run it locally instead, you can:
1) `pip install -r requirements.txt` to install dependencies. Recommend using venv or Conda here.
2) `uvicorn face_detection.api:app --host 0.0.0.0 --port 8080` to start the server.


## Performance 
### (CPU) Average Time Per Image *
| Batch Size / Image Size | 256x256 | 512x512 | 1024x1024 | 2048x2048 |
|-------------------------|---------|---------|-----------|-----------|
| 1                       | 44ms    | 94ms    | 303ms     | 378ms     |
| 4                       | 23ms    | 69ms    | 267ms     | 344ms     |
| 16                      | 17ms    | 63ms    | 261ms     | 342ms     |

### (GPU) Average Time Per Image *
| Batch Size / Image Size 	| 256x256 	| 512x512 	| 1024x1024 	| 2048x2048 	|
|-------------------------	|---------	|---------	|-----------	|-----------	|
| 1                       	| 10ms    	| 17ms    	| 38ms      	| 100ms     	|
| 4                       	| 4ms     	| 8ms     	| 23ms      	| 83ms      	|
| 16                      	| 3ms     	| 6ms     	| 22ms      	| 80ms      	|

\* On a machine with a Intel 14900K CPU, Nvidia 4090 GPU, and 32GB of DDR5 RAM.

## Accuracy
| Benchmark Name      	| Accuracy 	|
|---------------------	|----------	|
| WIDER FACE (Easy)   	| 95%      	|
| WIDER FACE (Medium) 	| 94%      	|
| WIDER FACE (Hard)   	| 84%      	|

[Source](https://arxiv.org/pdf/1905.00641.pdf)


## Usage
When you start the server, it will listen on port 8080 by default. You can navigate to `http://localhost:8080/docs` and use the GUI to send test requests to the API.

The following endpoints are available:

- `/detect_faces_from_urls/`: Provide a list of image URLs and the server will try to download them.
- `/detect_faces_from_base64/`: Provide images encoded as base64 strings and the server will decode them/
- `/detect_faces_from_files/`: Provide images directly as form-data.
- `/ready/`: Returns HTTP 200 if the server is initialized and ready. 

An example response looks like this:
```
{
  "result": [
    {
      "faces_count": 1,
      "faces": [
        {
          "bounding_box": [
            287.1794128417969,
            147.63372802734375,
            771.133056640625,
            771.7412719726562
          ],
          "landmarks": [
            {
              "position": [
                451.84375,
                378.625
              ],
              "type": "left_eye"
            },
            {
              "position": [
                666.25,
                376.875
              ],
              "type": "right_eye"
            },
            {
              "position": [
                585.8125,
                518.03125
              ],
              "type": "nose"
            },
            {
              "position": [
                465.09375,
                605.375
              ],
              "type": "left_mouth"
            },
            {
              "position": [
                647.25,
                602.1875
              ],
              "type": "right_mouth"
            }
          ],
          "confidence": 0.9999823570251465
        }
      ]
    }
  ]
}
```

The bounding box is represented by four numbers `[x1, y1, x2, y2]` which are the coordinates of the top-left and bottom-right vertices.

## As a Library
If you don't need a server but want to do inference directly in your own code, you can also use this repo as a library:
```
from PIL import Image

from face_detection.service import detect_faces

image = Image.open("some_image.jpg")
results = detect_faces([image])
```

## Tests
There are some basic tests under `face_detection/tests`. 

They can be run with `python -m pytest .` from inside the directory.
