import numpy as np
import torch


def check_image(im: np.ndarray):
    assert im.dtype == np.uint8, f"Expect image to have dtype np.uint8. Was: {im.dtype}"
    assert len(im.shape) == 4, f"Expected image to have 4 dimensions. got: {im.shape}"
    assert (
        im.shape[-1] == 3
    ), f"Expected image to be RGB, got: {im.shape[-1]} color channels"


def to_cuda(elements, device):
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.to(device) for x in elements]
        return elements.to(device)
    return elements


def get_device():
    return torch.device("cpu")


def batched_decode(loc, priors, variances, to_XYXY=True):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [N, num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    priors = priors[None]
    boxes = torch.cat(
        (
            priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
        ),
        dim=2,
    )
    if to_XYXY:
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def scale_boxes(imshape, boxes):
    height, width = imshape
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [N, num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    priors = priors[None]
    landms = torch.cat(
        (
            priors[:, :, :2] + pre[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, :2] + pre[:, :, 2:4] * variances[0] * priors[:, :, 2:],
            priors[:, :, :2] + pre[:, :, 4:6] * variances[0] * priors[:, :, 2:],
            priors[:, :, :2] + pre[:, :, 6:8] * variances[0] * priors[:, :, 2:],
            priors[:, :, :2] + pre[:, :, 8:10] * variances[0] * priors[:, :, 2:],
        ),
        dim=2,
    )
    return landms
