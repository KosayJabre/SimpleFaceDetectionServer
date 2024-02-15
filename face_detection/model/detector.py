import torch
import numpy as np
from . import utils
import typing
from .network import RetinaFace
from .utils import batched_decode
from .utils import decode_landm
from .config import RESNET50_CONFIG
from .prior_box import PriorBox
from torchvision.ops import nms
import os
import zipfile
import io
import typing
from abc import ABC, abstractmethod

import numpy as np
import torch
from torchvision.ops import nms

from .utils import check_image, scale_boxes


def load_model_weights():
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_file_directory, "RetinaNetResNet50.zip")

    with zipfile.ZipFile(model_path, "r") as zip_ref:
        with zip_ref.open("RetinaNetResNet50") as model_file:
            model_bytes = model_file.read()

    return io.BytesIO(model_bytes)


class RetinaNetDetector():
    def __init__(
        self,
        confidence_threshold: float,
        nms_iou_threshold: float,
        device: torch.device,
        max_resolution: int,
        fp16_inference: bool,
        clip_boxes: bool,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.max_resolution = max_resolution
        self.fp16_inference = fp16_inference
        self.clip_boxes = clip_boxes
        self.mean = np.array([123, 117, 104], dtype=np.float32).reshape(1, 1, 1, 3)

        net = RetinaFace(cfg=RESNET50_CONFIG)
        net.eval()

        state_dict = torch.load(load_model_weights(), map_location=torch.device("cpu"))
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)

        self.cfg = RESNET50_CONFIG
        self.net = net.to(self.device)
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.prior_box_cache = {}

    def filter_boxes(self, boxes: torch.Tensor) -> typing.List[np.ndarray]:
        """Performs NMS and score thresholding

        Args:
            boxes (torch.Tensor): shape [N, B, 5] with (xmin, ymin, xmax, ymax, score)
        Returns:
            list: N np.ndarray of shape [B, 5]
        """
        final_output = []
        for i in range(len(boxes)):
            scores = boxes[i, :, 4]
            keep_idx = scores >= self.confidence_threshold
            boxes_ = boxes[i, keep_idx, :-1]
            scores = scores[keep_idx]
            if scores.dim() == 0:
                final_output.append(torch.empty(0, 5))
                continue
            keep_idx = nms(boxes_, scores, self.nms_iou_threshold)
            scores = scores[keep_idx].view(-1, 1)
            boxes_ = boxes_[keep_idx].view(-1, 4)
            output = torch.cat((boxes_, scores), dim=-1)
            final_output.append(output)
        return final_output

    @torch.no_grad()
    def resize(self, image, shrink: float):
        if self.max_resolution is None and shrink == 1:
            return image
        height, width = image.shape[2:4]
        shrink_factor = self.max_resolution / max((height, width))
        if shrink_factor <= shrink:
            shrink = shrink_factor
        size = (int(height * shrink), int(width * shrink))
        image = torch.nn.functional.interpolate(image, size=size)
        return image

    def _pre_process(self, image: np.ndarray, shrink: float) -> torch.Tensor:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [N, height, width, 3]
        Returns:
            torch.Tensor: shape [N, 3, height, width]
        """
        assert image.dtype == np.uint8
        height, width = image.shape[1:3]
        image = image.astype(np.float32) - self.mean
        image = np.moveaxis(image, -1, 1)
        image = torch.from_numpy(image)
        image = self.resize(image, shrink)
        image = image.to(self.device)
        image = image.to(self.device)
        return image

    def _batched_detect(self, image: np.ndarray) -> typing.List[np.ndarray]:
        boxes = self._detect(image)
        boxes = self.filter_boxes(boxes)
        if self.clip_boxes:
            boxes = [box.clamp(0, 1) for box in boxes]
        return boxes

    @torch.no_grad()
    def batched_detect(self, image: np.ndarray, shrink=1.0) -> typing.List[np.ndarray]:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [N, height, width, 3]
        Returns:
            np.ndarray: a list with N set of bounding boxes of
                shape [B, 5] with (xmin, ymin, xmax, ymax, score)
        """
        check_image(image)
        height, width = image.shape[1:3]
        image = self._pre_process(image, shrink)
        boxes = self._batched_detect(image)
        boxes = [scale_boxes((height, width), box).cpu().numpy() for box in boxes]
        self.validate_detections(boxes)
        return boxes

    def validate_detections(self, boxes: typing.List[np.ndarray]):
        for box in boxes:
            assert np.all(box[:, 4] <= 1) and np.all(
                box[:, 4] >= 0
            ), f"Confidence values not valid: {box}"

    def batched_detect_with_landmarks(
        self, image: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Takes N images and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [N, height, width, 3]
        Returns:
            np.ndarray: shape [N, 5] with (xmin, ymin, xmax, ymax, score)
            np.ndarray: shape [N, 5, 2] with 5 landmarks with (x, y)
        """
        image = image.astype(np.float32) - self.mean
        image = np.moveaxis(image, -1, 1)
        image = torch.from_numpy(image)
        orig_shape = image.shape[2:]
        image = self.resize(image, 1).to(self.device)
        boxes, landms = self._detect(image, return_landmarks=True)
        scores = boxes[:, :, -1]
        boxes = boxes[:, :, :-1]
        final_output_box = []
        final_output_landmarks = []
        for i in range(len(boxes)):
            boxes_ = boxes[i]
            landms_ = landms[i]
            scores_ = scores[i]
            # Confidence thresholding
            keep_idx = scores_ >= self.confidence_threshold
            boxes_ = boxes_[keep_idx]
            scores_ = scores_[keep_idx]
            landms_ = landms_[keep_idx]
            # Non maxima suppression
            keep_idx = nms(boxes_, scores_, self.nms_iou_threshold)
            boxes_ = boxes_[keep_idx]
            scores_ = scores_[keep_idx]
            landms_ = landms_[keep_idx]
            # Scale boxes
            height, width = orig_shape
            if self.clip_boxes:
                boxes_ = boxes_.clamp(0, 1)
            boxes_[:, [0, 2]] *= width
            boxes_[:, [1, 3]] *= height

            # Scale landmarks
            landms_ = landms_.cpu().numpy().reshape(-1, 5, 2)
            landms_[:, :, 0] *= width
            landms_[:, :, 1] *= height
            dets = torch.cat((boxes_, scores_.view(-1, 1)), dim=1).cpu().numpy()
            final_output_box.append(dets)
            final_output_landmarks.append(landms_)
        return final_output_box, final_output_landmarks

    @torch.no_grad()
    def _detect(self, image: np.ndarray, return_landmarks=False) -> np.ndarray:
        """Batched detect
        Args:
            image (np.ndarray): shape [N, H, W, 3]
        Returns:
            boxes: list of length N with shape [num_boxes, 5] per element
        """
        image = image[:, [2, 1, 0]]
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            loc, conf, landms = self.net(image)  # forward pass
            scores = conf[:, :, 1:]
            height, width = image.shape[2:]
            if image.shape[2:] in self.prior_box_cache:
                priors = self.prior_box_cache[image.shape[2:]]
            else:
                priorbox = PriorBox(self.cfg, image_size=(height, width))
                priors = priorbox.forward()
                self.prior_box_cache[image.shape[2:]] = priors
            priors = utils.to_cuda(priors, self.device)
            prior_data = priors.data
            boxes = batched_decode(loc, prior_data, self.cfg["variance"])
            boxes = torch.cat((boxes, scores), dim=-1)
        if return_landmarks:
            landms = decode_landm(landms, prior_data, self.cfg["variance"])
            return boxes, landms
        return boxes
