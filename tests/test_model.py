import numpy as np
import pytest
import torch

from src.models.vae256 import Conv_VAE
from src.utils import torch_utilities
from src.utils.torch_utilities import resolve_device, to_images, to_model_input
from src.utils.train_utilities import compute_loss

ACCELERATORS = [
    name
    for name, available in (
        ("mps", torch.backends.mps.is_available()),
        ("cuda", torch.cuda.is_available()),
    )
    if available
]


def test_architecture():
    model = Conv_VAE()
    assert sum(p.numel() for p in model.parameters()) == 47_514_479
    assert all(not bias.any() for name, bias in model.named_parameters() if "bias" in name)
    x = torch.rand(2, 3, 128, 960)
    mean, logvar = model.encode(x)
    assert mean.shape == logvar.shape == (2, 256)
    assert (mean >= 0).all()  # final encoder ReLU
    assert model(x).shape == model.decode(mean).shape == (2, 3, 128, 960)
    assert 0 <= model.sample(torch.zeros(1, 256)).min() <= 1


def test_devices(monkeypatch):
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device().type in ("cpu", "mps", "cuda")
    with pytest.raises(ValueError, match="one of"):
        resolve_device("tpu")
    monkeypatch.setattr(torch_utilities.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="not available"):
        resolve_device("cuda")


def test_image_layout_conversions():
    images = np.zeros((2, 4, 6, 3), np.uint8)
    images[..., 1] = 255
    tensor = to_model_input(images, "cpu")
    assert tensor.shape == (2, 3, 4, 6) and tensor.dtype == torch.float32
    assert tensor.is_contiguous()
    assert tensor[:, 1].eq(1).all() and tensor[:, 0].eq(0).all()
    np.testing.assert_array_equal(to_images(tensor), images / 255)
    assert to_model_input(images / 255.0, "cpu").dtype == torch.float32  # float64 in


@pytest.mark.skipif(not ACCELERATORS, reason="no GPU")
@pytest.mark.parametrize("device", ACCELERATORS)
def test_gpu_gradients_match_cpu(device):
    """The accelerator must reproduce the CPU's loss and gradients.

    A unit whose pre-activation sits at a ReLU kink can flip between devices,
    hence the loose per-tensor bound.
    """
    torch.manual_seed(0)
    cpu = Conv_VAE()
    gpu = Conv_VAE().to(device)
    gpu.load_state_dict(cpu.state_dict())
    x = torch.rand(4, 3, 128, 960)
    losses = []
    for model, data in ((cpu, x), (gpu, x.to(device))):
        loss = compute_loss(lambda batch: model.decode(model.encode(batch)[0]), data)
        loss.backward()
        losses.append(loss.item())
    assert losses[1] == pytest.approx(losses[0], rel=1e-5)
    for (name, a), b in zip(cpu.named_parameters(), gpu.parameters()):
        error = (b.grad.cpu() - a.grad).norm() / a.grad.norm().clamp_min(1e-12)
        assert error < 1e-2, name

