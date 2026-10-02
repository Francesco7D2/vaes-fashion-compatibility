"""Loss, optimizer, training step and reconstruction previews."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.torch_utilities import to_images, to_model_input

# Product photos have light grey or beige backgrounds (about 0.9) and empty
# slots are white; darker pixels are treated as product pixels.
FOREGROUND_LEVEL = 0.85


def compute_loss(model, x, kl_weight=0.0, foreground_weight=1.0):
    """Mean pixel binary cross entropy between reconstruction logits and `x`.

    `foreground_weight` multiplies the weight of product pixels. With
    `kl_weight` > 0 the KL divergence to a standard normal prior is added, per
    pixel, so that `kl_weight=1` gives the negative ELBO divided by the number
    of pixels.
    """
    if kl_weight:
        mean, logvar = model.encode(x)
        logits = model.decode(model.reparameterize(mean, logvar))
    else:
        logits = model(x)
    pixels = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    if foreground_weight == 1:
        loss = pixels.mean()
    else:
        weight = 1 + (foreground_weight - 1) * (x < FOREGROUND_LEVEL).any(dim=1, keepdim=True)
        loss = (pixels * weight).sum() / weight.expand_as(pixels).sum()
    if kl_weight:
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(dim=1).mean()
        loss = loss + kl_weight * kl / x[0].numel()
    return loss


def make_optimizer(model, learning_rate):
    """Adam; on GPUs the fused kernel updates all parameters in one launch."""
    device = next(model.parameters()).device
    return torch.optim.Adam(model.parameters(), learning_rate, fused=device.type != "cpu")


def train_step(model, x, optimizer, **loss_options):
    """One update; the loss stays on the device so no step waits for the GPU."""
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, x, **loss_options)
    loss.backward()
    optimizer.step()
    return loss.detach()


def fit(
    model,
    dataset,
    epochs,
    batch_size,
    learning_rate,
    shuffle=True,
    keep_best=True,
    on_epoch=None,
    progress=True,
    **loss_options,
):
    """Train on a uint8 N x H x W x 3 tensor; return the mean loss of every epoch.

    `shuffle` visits the outfits in a new random order every epoch. With
    `keep_best` the model ends with the weights of its lowest-loss epoch.
    `on_epoch(epoch, history)` runs after every epoch, e.g. for checkpoints.
    """
    device = next(model.parameters()).device
    optimizer = make_optimizer(model, learning_rate)
    history, best = [], None
    for epoch in range(1, epochs + 1):
        start = time.monotonic()
        order = torch.randperm(len(dataset), device=dataset.device) if shuffle else None
        total = torch.zeros((), device=device)
        batches = tqdm(
            range(0, len(dataset), batch_size),
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            leave=False,
            disable=not progress,
        )
        for index, first in enumerate(batches, start=1):
            rows = slice(first, first + batch_size) if order is None else order[first : first + batch_size]
            batch = to_model_input(dataset[rows], device)
            total += train_step(model, batch, optimizer, **loss_options) * len(batch)
            if index % 10 == 0:  # reading the loss waits for the GPU; do it rarely
                seen = min(first + batch_size, len(dataset))
                batches.set_postfix(loss=f"{total.item() / seen:.5f}")
        batches.close()
        history.append(total.item() / len(dataset))
        if progress:
            tqdm.write(
                f"Epoch {epoch}/{epochs}: loss={history[-1]:.6f}; "
                f"seconds={time.monotonic() - start:.1f}"
            )
        if keep_best and history[-1] == min(history):
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if on_epoch is not None:
            on_epoch(epoch, history)
    if best is not None:
        model.load_state_dict(best)
    return history


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
