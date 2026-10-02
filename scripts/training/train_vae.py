"""Train the outfit autoencoder; export weights, embeddings and previews."""

import math
import pickle

import torch

from src.data_processing.image_processing import load_strips
from src.generation.embeddings import encode_strips
from src.models.vae256 import Conv_VAE, save_model
from src.utils.setup_utilities import ensure_parent, load_config
from src.utils.torch_utilities import resolve_device
from src.utils.train_utilities import fit, generate_and_save_images

LOSS_OPTIONS = ("kl_weight", "foreground_weight")


def training_options(training):
    """Keyword arguments of `fit` taken from the `training` config section."""
    options = {
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
        "learning_rate": training.get("learning_rate", 1e-4),
        "shuffle": training.get("shuffle", True),
        "keep_best": training.get("keep_best", True),
    }
    options.update({key: training[key] for key in LOSS_OPTIONS if key in training})
    return options


def train_outfit_embeddings(config_path=None, overrides=None):
    config = load_config(config_path, overrides)
    data, training = config["data"], config["training"]
    for key in ("epochs", "batch_size", "checkpoint_every"):
        value = training.get(key, 20 if key == "checkpoint_every" else 0)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (data["image_height"], data["image_width"]) != (128, 96):
        raise ValueError("The model requires 128 x 96 product images")
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
    model = Conv_VAE(linear_head=training.get("linear_head", True)).to(device)
    print(
        f"Training on {device.type}; "
        f"{math.ceil(len(dataset) / training['batch_size'])} batches per epoch"
    )

    def export(epoch, history):
        save_model(model, ensure_parent(training["model_weights_path"]))
        with open(ensure_parent(training["model_data_path"]), "wb") as stream:
            pickle.dump(
                {
                    "latent_space": encode_strips(model, dataset, training["batch_size"]),
                    "loss": history,
                },
                stream,
            )
        generate_and_save_images(model, epoch, samples, training["epoch_images_path"])

    def checkpoint(epoch, history):
        if epoch % training.get("checkpoint_every", 20) == 0 and epoch < training["epochs"]:
            export(epoch, history)

    history = fit(model, dataset, on_epoch=checkpoint, **training_options(training))
    export(training["epochs"], history)  # with keep_best, the lowest-loss weights
    return model, history


if __name__ == "__main__":
    train_outfit_embeddings()
