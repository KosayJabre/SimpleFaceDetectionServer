# Simple Face Detection Server

## Performance Benchmark
The following benchmark was carried out on a machine with the following hardware:
- **CPU:** 14900K
- **GPU:** 4090
- **RAM:** 32GB DDR5-7200

### (CPU) Average Time Per Image
| Batch Size / Image Size | 256x256 | 512x512 | 1024x1024 | 2048x2048 |
|-------------------------|---------|---------|-----------|-----------|
| 1                       | 44ms    | 94ms    | 303ms     | 378ms     |
| 4                       | 23ms    | 69ms    | 267ms     | 344ms     |
| 16                      | 17ms    | 63ms    | 261ms     | 342ms     |

### (GPU) Average Time Per Image
| Batch Size / Image Size 	| 256x256 	| 512x512 	| 1024x1024 	| 2048x2048 	|
|-------------------------	|---------	|---------	|-----------	|-----------	|
| 1                       	| 10ms    	| 17ms    	| 38ms      	| 100ms     	|
| 4                       	| 4ms     	| 8ms     	| 23ms      	| 83ms      	|
| 16                      	| 3ms     	| 6ms     	| 22ms      	| 80ms      	|

## Usage
To run:
`docker compose up --build`

To try it out:
Navigate to `localhost:8000/docs` and use the GUI to call the API.
