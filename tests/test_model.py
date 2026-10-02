import numpy as np
import pytest
import torch

from src.models.vae256 import Conv_VAE, load_trained_model, save_model
from src.utils import torch_utilities
from src.utils.torch_utilities import resolve_device, to_images, to_model_input
from src.utils.train_utilities import compute_loss, fit

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
    assert (Conv_VAE(linear_head=False).encode(x)[0] >= 0).all()  # final encoder ReLU
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



def test_linear_head_and_saved_architecture(tmp_path):
    torch.manual_seed(0)
    model = Conv_VAE()
    mean, _ = model.encode(torch.rand(2, 3, 128, 960))
    assert (mean < 0).any()  # linear head by default
    save_model(Conv_VAE(linear_head=False), tmp_path / "relu.pt")
    assert not load_trained_model(tmp_path / "relu.pt", "cpu").linear_head
    save_model(model, tmp_path / "model.pt")
    restored = load_trained_model(tmp_path / "model.pt", "cpu")
    assert restored.linear_head
    x = torch.rand(1, 3, 128, 960)
    assert torch.allclose(restored.encode(x)[0], model.encode(x)[0])


class _Constant(torch.nn.Module):
    """Encoder output fixed by `mean`/`logvar`; decoder returns a parameter."""

    def __init__(self, mean, logvar):
        super().__init__()
        self.mean, self.logvar = mean, logvar
        self.logits = torch.nn.Parameter(torch.zeros(1, 3, 2, 2))

    def encode(self, x):
        return self.mean.expand(len(x), -1), self.logvar.expand(len(x), -1)

    def reparameterize(self, mean, logvar):
        return mean

    def decode(self, z):
        return self.logits.expand(len(z), -1, -1, -1)

    def forward(self, x):
        return self.decode(self.encode(x)[0])


def test_loss_options():
    x = torch.full((2, 3, 2, 2), 1.0)
    x[:, :, 0, 0] = 0.0  # one dark (product) pixel per image
    prior = _Constant(torch.zeros(1, 4), torch.zeros(1, 4))
    plain = compute_loss(prior, x)
    assert compute_loss(prior, x, kl_weight=1.0).item() == pytest.approx(plain.item())  # KL is 0
    shifted = _Constant(torch.ones(1, 4), torch.zeros(1, 4))
    kl = 0.5 * 4  # 0.5 * sum(mean^2) per sample
    assert compute_loss(shifted, x, kl_weight=1.0).item() == pytest.approx(plain.item() + kl / x[0].numel())
    # All logits are zero, so every pixel has the same loss and weighting changes nothing...
    assert compute_loss(prior, x, foreground_weight=5.0).item() == pytest.approx(plain.item())
    prior.logits.data[:, :, 0, 0] = 3.0  # ...until the dark pixel is reconstructed badly.
    assert compute_loss(prior, x, foreground_weight=5.0) > compute_loss(prior, x)


def test_fit_shuffles_and_keeps_the_best_epoch():
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 3, 1)
    dataset = torch.randint(0, 256, (10, 4, 4, 3), dtype=torch.uint8)
    states = []
    history = fit(
        model,
        dataset,
        epochs=4,
        batch_size=3,
        learning_rate=0.5,
        progress=False,
        on_epoch=lambda epoch, history: states.append(
            {k: v.clone() for k, v in model.state_dict().items()}
        ),
    )
    assert len(history) == len(states) == 4
    best = states[int(np.argmin(history))]
    assert all(torch.equal(best[k], v) for k, v in model.state_dict().items())
