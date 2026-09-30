"""Train the original reconstruction objective using the shared model."""

import math
import pickle
import time

import torch
from tqdm import tqdm

from src.data_processing.image_processing import load_strips
from src.models.vae256 import Conv_VAE
from src.utils.setup_utilities import ensure_parent, load_config
from src.utils.torch_utilities import resolve_device, to_model_input
from src.utils.train_utilities import (
    generate_and_save_images,
    make_optimizer,
    train_step,
)


@torch.inference_mode()
def encode_dataset(model, dataset, batch_size):
    """Posterior means of every strip, in dataset order."""
    means = [
        model.encode(to_model_input(dataset[start : start + batch_size], dataset.device))[0]
        for start in range(0, len(dataset), batch_size)
    ]
    return torch.cat(means).cpu().numpy()


def train_outfit_embeddings(config_path=None, overrides=None):
    config = load_config(config_path, overrides)
    data, training = config["data"], config["training"]
    for key in ("epochs", "batch_size", "checkpoint_every"):
        value = training.get(key, 20 if key == "checkpoint_every" else 0)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (data["image_height"], data["image_width"]) != (128, 96):
        raise ValueError("The original model requires 128 x 96 product images")
    torch.manual_seed(training.get("seed", 42))
    device = resolve_device(training.get("device", "auto"))
    strips = load_strips(training["train_dataset_path"])
    if not len(strips):
        raise ValueError("Training dataset is empty")
    if strips.shape[1:] != (128, 960, 3):
        raise ValueError(f"Expected 128 x 960 x 3 strips, found {strips.shape[1:]}")
    # Decoded once; the uint8 strips (about 370 KB each) stay on the device.
    dataset = torch.from_numpy(strips).to(device)
    del strips
    samples = load_strips(training["train_sample_path"])
    batch_size = training["batch_size"]
    num_batches = math.ceil(len(dataset) / batch_size)
    model = Conv_VAE().to(device)
    optimizer = make_optimizer(model, training.get("learning_rate", 1e-4))
    history = []
    print(f"Training on {device.type}; {num_batches} batches per epoch")
    for epoch in range(1, training["epochs"] + 1):
        start = time.monotonic()
        total = torch.zeros((), device=device)
        progress = tqdm(
            range(0, len(dataset), batch_size),
            desc=f"Epoch {epoch}/{training['epochs']}",
            unit="batch",
            leave=False,
        )
        for index, first in enumerate(progress, start=1):
            batch = to_model_input(dataset[first : first + batch_size], device)
            total += train_step(model, batch, optimizer) * len(batch)
            if index % 10 == 0:  # reading the loss waits for the GPU; do it rarely
                seen = min(first + batch_size, len(dataset))
                progress.set_postfix(loss=f"{total.item() / seen:.5f}")
        progress.close()
        history.append(total.item() / len(dataset))
        tqdm.write(
            f"Epoch {epoch}/{training['epochs']}: reconstruction_loss={history[-1]:.6f}; "
            f"seconds={time.monotonic() - start:.1f}"
        )
        if (
            epoch % training.get("checkpoint_every", 20) == 0
            or epoch == training["epochs"]
        ):
            torch.save(model.state_dict(), ensure_parent(training["model_weights_path"]))
            with open(ensure_parent(training["model_data_path"]), "wb") as stream:
                pickle.dump(
                    {
                        "latent_space": encode_dataset(model, dataset, batch_size),
                        "reconstruction_loss": history,
                        "embedding_type": "posterior_mean",
                    },
                    stream,
                )
            generate_and_save_images(model, epoch, samples, training["epoch_images_path"])
    return model, history


if __name__ == "__main__":
    train_outfit_embeddings()
