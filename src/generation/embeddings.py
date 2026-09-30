"""Encode outfit tables with a trained model, matching the training pipeline."""

import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.data_processing.image_processing import (
    build_outfit_strip,
    load_and_preprocess_image,
    parse_tfrecord_fn,
    serialize_image,
)


class ProductImages:
    """Lazy product-image loader; caches uint8 pixels (about 37 KB per product)."""

    def __init__(self, image_paths, image_width=96, image_height=128):
        self.image_paths = image_paths
        self.width, self.height = image_width, image_height
        self._cache = {}

    def __call__(self, product_id):
        """Return a 1 x H x W x 3 float32 tensor in [0, 1]."""
        if product_id not in self._cache:
            image = load_and_preprocess_image(
                self.image_paths.loc[product_id], self.width, self.height
            )
            self._cache[product_id] = tf.image.convert_image_dtype(
                image, tf.uint8, saturate=True
            ).numpy()
        return tf.image.convert_image_dtype(self._cache[product_id], tf.float32)


def encode_outfits(
    model, outfits, images, batch_size=32, match_training_jpeg=True, progress=True
):
    """Posterior-mean embeddings (len(outfits) x latent_dim) for slot-wide rows.

    Training strips pass through JPEG inside the TFRecords; with
    `match_training_jpeg` candidates take the same round trip so their
    embeddings are comparable to the exported training ones.
    """
    encode = tf.function(lambda x: model.encode(x)[0], reduce_retracing=True)
    embeddings = []
    for start in tqdm(
        range(0, len(outfits), batch_size), disable=not progress, desc="Encoding"
    ):
        batch = []
        for row in outfits.iloc[start : start + batch_size].to_dict("records"):
            strip = build_outfit_strip(row, images, images.width, images.height)
            if match_training_jpeg:
                batch.append(parse_tfrecord_fn(serialize_image(strip)))
            else:
                batch.append(strip[0])
        embeddings.append(encode(tf.stack(batch)).numpy())
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
