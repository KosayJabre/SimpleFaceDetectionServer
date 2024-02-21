import io
import os
import zipfile

import numpy as np
import torch
from PIL import Image
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


class RetinaNetDetector:
    def __init__(
        self,
        nms_iou_threshold: float,
    ):
        self.nms_iou_threshold = nms_iou_threshold
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.cfg = RESNET50_CONFIG

        net = RetinaFace(cfg=self.cfg)
        net.eval()
        state_dict = torch.load(load_model_weights(), map_location=get_device())
        net.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})

        self.net = net.to(get_device())

    def detect(self, images: np.ndarray, confidence_threshold=0.5, clip_boxes=True) -> tuple[np.ndarray, np.ndarray]:
        original_shapes = [img.shape[:2] for img in images]  # Keep track of shapes so we can scale bounding boxes back to original size
        processed_images = self._pre_process(images)

        # Feed-forward
        boxes, landms = self._detect(processed_images)

        final_output_box = []
        final_output_landmarks = []
        for i, (boxes_i, landms_i, scores_i) in enumerate(zip(boxes, landms, boxes[:, :, -1])):
            # Filter out low confidence detections
            keep = scores_i >= confidence_threshold
            boxes_i, landms_i, scores_i = boxes_i[keep], landms_i[keep], scores_i[keep]

            # Filter out low confidence overlapping boxes
            box_coords = boxes_i[:, :4]
            keep = nms(box_coords, scores_i, self.nms_iou_threshold)

            boxes_i, landms_i, scores_i = boxes_i[keep], landms_i[keep], scores_i[keep]

            # Clipping makes sure the bounding boxes are within the image
            if clip_boxes:
                boxes_i = boxes_i.clamp(0, 1)

            # Convert bounding boxes and landmarks back to original size
            boxes_i[:, [0, 2]] *= original_shapes[i][1]
            boxes_i[:, [1, 3]] *= original_shapes[i][0]
            landms_i = landms_i.reshape(-1, 5, 2)
            landms_i[:, :, 0] *= original_shapes[i][1]
            landms_i[:, :, 1] *= original_shapes[i][0]

            final_output_box.append(torch.cat((boxes_i, scores_i.unsqueeze(-1)), dim=1).cpu().numpy())
            final_output_landmarks.append(landms_i.cpu().numpy())

        return final_output_box, final_output_landmarks

    def _pre_process(self, images: np.ndarray) -> torch.Tensor:
        """
        Resize and normalize images. We have to resize because the model was trained on smaller images.
        The bounding boxes are converted back to the original size after detection.
        """
        max_resolution = (1080, 1080)
        resized_images = []
        for image in images:
            if image.shape[0] > max_resolution[0] or image.shape[1] > max_resolution[1]:
                img = Image.fromarray(image.astype("uint8"), "RGB")
                img.thumbnail(max_resolution, Image.LANCZOS)
                image = np.array(img)
            resized_images.append(image)
        images = np.array(resized_images)
        images = images.astype(np.float32) - self.mean
        images = np.moveaxis(images, -1, 1)
        images = torch.from_numpy(images)
        return images.to(get_device())

    @torch.no_grad()
    def _detect(self, image: torch.Tensor):
        locations, confidences, landmarks = self.net(image)
        scores = confidences[:, :, 1:]
        height, width = image.shape[2:]
        priors = self._get_prior_boxes((height, width))
        boxes = batched_decode(locations, priors.data, self.cfg["variance"])
        boxes = torch.cat((boxes, scores), dim=-1)
        landmarks = decode_landm(landmarks, priors.data, self.cfg["variance"])
        return boxes, landmarks

    def _get_prior_boxes(self, image_size):
        priorbox = PriorBox(self.cfg, image_size=image_size)
        priors = priorbox.forward()
        return priors.to(get_device())
