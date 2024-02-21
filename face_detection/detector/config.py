RESNET50_CONFIG = {
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "clip": False,
    "variance": [0.1, 0.2],
    "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
    "in_channel": 256,
    "out_channel": 256,
}
