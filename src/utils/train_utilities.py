"""Loss, optimizer, training step and reconstruction previews."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.utils.torch_utilities import to_images, to_model_input


def compute_loss(model, x):
    """Mean pixel binary cross entropy between reconstruction logits and `x`."""
    return F.binary_cross_entropy_with_logits(model(x), x)


def make_optimizer(model, learning_rate):
    """Adam; on GPUs the fused kernel updates all parameters in one launch."""
    device = next(model.parameters()).device
    return torch.optim.Adam(model.parameters(), learning_rate, fused=device.type != "cpu")


def train_step(model, x, optimizer):
    """One update; the loss stays on the device so no step waits for the GPU."""
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, x)
    loss.backward()
    optimizer.step()
    return loss.detach()


@torch.inference_mode()
def generate_and_save_images(model, epoch, test_sample, path):
    """Save deterministic reconstructions of any nonempty uint8 sample batch."""
    count = len(test_sample)
    if not count:
        return
    device = next(model.parameters()).device
    mean, _ = model.encode(to_model_input(test_sample, device))
    predictions = to_images(model.decode(mean, apply_sigmoid=True))
    Path(path).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(count * 2, 1, figsize=(10, count * 2), squeeze=False)
    for index, (original, reconstructed) in enumerate(zip(test_sample, predictions)):
        for axis, image in zip(
            axes[2 * index : 2 * index + 2, 0], (original, reconstructed)
        ):
            axis.imshow(image)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(Path(path) / f"image_at_epoch_{epoch:04d}.png")
    plt.close(fig)
