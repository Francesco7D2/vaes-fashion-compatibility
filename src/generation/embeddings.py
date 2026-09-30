"""Encode outfit tables with a trained model, matching the training pipeline."""

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tqdm import tqdm

from src.data_processing.image_processing import (
    build_outfit_strip,
    jpeg_round_trip,
    load_and_preprocess_image,
)
from src.utils.torch_utilities import to_model_input


class ProductImages:
    """Lazy product-image loader; caches uint8 pixels (about 37 KB per product)."""

    def __init__(self, image_paths, image_width=96, image_height=128):
        self.image_paths = image_paths
        self.width, self.height = image_width, image_height
        self._cache = {}

    def __call__(self, product_id):
        """Return an H x W x 3 uint8 image."""
        if product_id not in self._cache:
            self._cache[product_id] = load_and_preprocess_image(
                self.image_paths.loc[product_id], self.width, self.height
            )
        return self._cache[product_id]


def encode_outfits(
    model, outfits, images, batch_size=32, match_training_jpeg=True, progress=True
):
    """Posterior-mean embeddings (len(outfits) x latent_dim) for slot-wide rows.

    Training strips pass through JPEG inside the strip archives; with
    `match_training_jpeg` candidates take the same round trip so their
    embeddings are comparable to the exported training ones. Strips are built
    in threads while the model encodes the previous batch on its device.
    """
    device = next(model.parameters()).device
    rows = outfits.to_dict("records")

    def strip(row):
        image = build_outfit_strip(row, images, images.width, images.height)
        return jpeg_round_trip(image) if match_training_jpeg else image

    def build(start):
        return np.stack(list(pool.map(strip, rows[start : start + batch_size])))

    starts = range(0, len(rows), batch_size)
    embeddings = []
    with ThreadPoolExecutor(os.cpu_count()) as pool, torch.inference_mode():
        batch = build(0) if rows else None
        for index in tqdm(range(len(starts)), disable=not progress, desc="Encoding"):
            mean = model.encode(to_model_input(batch, device))[0]
            if index + 1 < len(starts):
                batch = build(starts[index + 1])  # overlaps the queued GPU work
            embeddings.append(mean.cpu().numpy())
    if not embeddings:
        return np.zeros((0, model.latent_dim), dtype=np.float32)
    return np.concatenate(embeddings)


def load_training_embeddings(path):
    """Read the training export (trusted pickle written by train_vae)."""
    with open(path, "rb") as stream:
        exported = pickle.load(stream)
    if exported.get("embedding_type") != "posterior_mean":
        raise ValueError(
            "Legacy embedding export (sampled z); rebuild it by retraining with "
            "scripts.training.train_vae"
        )
    return exported
