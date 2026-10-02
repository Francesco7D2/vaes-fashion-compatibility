"""Device selection and conversions between image arrays and model tensors."""

import numpy as np
import torch

from src.utils.setup_utilities import DEVICES


def resolve_device(name="auto"):
    """`auto` picks CUDA, then Apple's Metal GPU (MPS), then CPU."""
    if name not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}")
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if name == "auto":
        name = next(device for device, ok in available.items() if ok)
    elif not available[name]:
        raise ValueError(f"Device {name!r} is not available on this machine")
    return torch.device(name)


def to_model_input(images, device):
    """N x H x W x 3 images (uint8, or floats in [0, 1]) -> N x 3 x H x W float32.

    Accepts arrays or tensors (possibly already on `device`). The result is
    contiguous: channels-last strides make MPS convolution gradients far slower.
    """
    if not torch.is_tensor(images):
        array = np.asarray(images)
        if array.dtype != np.uint8:
            array = array.astype(np.float32, copy=False)  # MPS has no float64
        images = torch.from_numpy(np.ascontiguousarray(array))
    tensor = images.to(device).permute(0, 3, 1, 2).contiguous()
    return tensor.float().div_(255) if tensor.dtype == torch.uint8 else tensor.float()


def to_images(tensor):
    """N x 3 x H x W tensor -> N x H x W x 3 float NumPy array."""
    return tensor.detach().permute(0, 2, 3, 1).cpu().numpy()
