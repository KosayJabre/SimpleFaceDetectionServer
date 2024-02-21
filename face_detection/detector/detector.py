import io
import os
import zipfile

import numpy as np
import torch
from pydantic import BaseModel, Field
from torchvision.ops import nms

from .config import RESNET50_CONFIG
from .network import RetinaFace
from .prior_box import PriorBox
from .utils import batched_decode, decode_landm, get_device


def load_model_weights():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_file_directory, "RetinaNetResNet50.zip")
    with zipfile.ZipFile(model_path, "r") as zip_ref:
        with zip_ref.open("RetinaNetResNet50") as model_file:
            model_bytes = model_file.read()
    return io.BytesIO(model_bytes)


class RetinaNetDetectorRequestParameters(BaseModel):
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    clip_boxes: bool = True
    max_resolution: int = None


class RetinaNetDetector:
    def __init__(
        self,
        nms_iou_threshold: float,
        max_resolution: int,
        fp16_inference: bool,
    ):
        self.nms_iou_threshold = nms_iou_threshold
        self.max_resolution = max_resolution
        self.fp16_inference = fp16_inference
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.cfg = RESNET50_CONFIG
        self.prior_box_cache = {}

        net = RetinaFace(cfg=self.cfg)
        net.eval()
        state_dict = torch.load(load_model_weights(), map_location=get_device())
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)

        self.net = net.to(get_device())

    def detect(
        self,
        images: np.ndarray,
        params: RetinaNetDetectorRequestParameters = RetinaNetDetectorRequestParameters(),
    ) -> tuple[np.ndarray, np.ndarray]:
        processed_images = self._pre_process(images)
        orig_shape = images.shape[1:3]
        boxes, landms = self._detect(processed_images, return_landmarks=True)

        final_output_box = []
        final_output_landmarks = []
        for i, (boxes_i, landms_i, scores_i) in enumerate(
            zip(boxes, landms, boxes[:, :, -1])
        ):
            keep = scores_i >= params.confidence_threshold
            boxes_i, landms_i, scores_i = boxes_i[keep], landms_i[keep], scores_i[keep]

            # Separate box coordinates from confidence scores before NMS
            box_coords = boxes_i[:, :4]  # Extract only the coordinates
            keep = nms(box_coords, scores_i, self.nms_iou_threshold)

            boxes_i, landms_i, scores_i = boxes_i[keep], landms_i[keep], scores_i[keep]
            if params.clip_boxes:
                boxes_i = boxes_i.clamp(0, 1)
            boxes_i[:, [0, 2]] *= orig_shape[1]
            boxes_i[:, [1, 3]] *= orig_shape[0]
            landms_i = landms_i.reshape(-1, 5, 2)
            landms_i[:, :, 0] *= orig_shape[1]
            landms_i[:, :, 1] *= orig_shape[0]
            final_output_box.append(
                torch.cat((boxes_i, scores_i.unsqueeze(-1)), dim=1).cpu().numpy()
            )
            final_output_landmarks.append(landms_i.cpu().numpy())

        return final_output_box, final_output_landmarks

    def _pre_process(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) - self.mean
        image = np.moveaxis(image, -1, 1)
        image = torch.from_numpy(image)
        if self.max_resolution:
            shrink = min(self.max_resolution / max(image.shape[2:]), 1)
            size = (int(image.shape[2] * shrink), int(image.shape[3] * shrink))
            image = torch.nn.functional.interpolate(
                image[None], size=size, mode="bilinear", align_corners=False
            )
        return image.to(get_device())

    @torch.no_grad()
    def _detect(self, image: torch.Tensor, return_landmarks=False):
        image = image.flip(-3)  # BGR to RGB
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            loc, conf, landms = self.net(image)
            scores = conf[:, :, 1:]
            height, width = image.shape[2:]
            priors = self._get_prior_boxes((height, width))
            boxes = batched_decode(loc, priors.data, self.cfg["variance"])
            boxes = torch.cat((boxes, scores), dim=-1)
            if return_landmarks:
                landms = decode_landm(landms, priors.data, self.cfg["variance"])
                return boxes, landms
        return boxes

    def _get_prior_boxes(self, image_size):
        """Cache or generate prior boxes."""
        if image_size in self.prior_box_cache:
            return self.prior_box_cache[image_size]
        priorbox = PriorBox(self.cfg, image_size=image_size)
        priors = priorbox.forward()
        self.prior_box_cache[image_size] = priors
        return priors.to(get_device())
