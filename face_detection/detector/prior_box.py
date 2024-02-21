from math import ceil

import numpy as np
import torch


def generate_prior_box(feature_maps, image_size, steps, min_sizes):
    n_anchors = sum(f[0] * f[1] * len(min_sizes[0]) for f in feature_maps)
    anchors = np.empty((n_anchors, 4), dtype=np.float32)
    idx_anchor = 0
    for k, f in enumerate(feature_maps):
        for i in range(f[0]):
            for j in range(f[1]):
                for min_size in min_sizes[k]:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    cx = (j + 0.5) * steps[k] / image_size[1]
                    cy = (i + 0.5) * steps[k] / image_size[0]
                    anchors[idx_anchor] = [cx, cy, s_kx, s_ky]
                    idx_anchor += 1
    return anchors


class PriorBox:
    def __init__(self, cfg, image_size=None):
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def forward(self):
        anchors = generate_prior_box(
            self.feature_maps, self.image_size, self.steps, self.min_sizes
        )
        output = torch.from_numpy(anchors).float()
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
