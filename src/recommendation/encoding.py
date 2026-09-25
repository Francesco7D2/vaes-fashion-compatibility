"""Encode candidates with exactly the training slot layout and JPEG round trip."""

from functools import lru_cache

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.data_processing.image_processing import (
    create_white_image,
    load_and_preprocess_image,
    parse_tfrecord_fn,
    serialize_image,
)
from src.data_processing.schema import PRODUCT_CLASSES


class OutfitImageLoader:
    """Bound image memory independently of the number of candidate outfits."""

    def __init__(self, paths, width=96, height=128, cache_size=256):
        if cache_size < 0:
            raise ValueError("image_cache_size cannot be negative")
        self.white = create_white_image(width, height)
        self.load = lru_cache(maxsize=cache_size)(
            lambda product: load_and_preprocess_image(paths[product], width, height)
        )

    def __call__(self, row):
        return tf.concat(
            [
                self.white
                if pd.isna(row[slot]) or row[slot] == ""
                else self.load(str(row[slot]))
                for slot in PRODUCT_CLASSES
            ],
            axis=2,
        )


def encode_candidates(model, candidates, paths, batch_size=32, cache_size=256):
    if batch_size < 1 or candidates.empty:
        raise ValueError("Encoding requires candidates and a positive batch size")
    loader = OutfitImageLoader(paths, cache_size=cache_size)
    embeddings = []
    for start in tqdm(
        range(0, len(candidates), batch_size), desc="Encoding candidates"
    ):
        images = [
            parse_tfrecord_fn(serialize_image(loader(row)))
            for row in candidates.iloc[start : start + batch_size].to_dict("records")
        ]
        mean, _ = model.encode(tf.stack(images))
        embeddings.append(mean.numpy())
    return np.concatenate(embeddings)


def encode_reference(model, record_path, batch_size=32):
    dataset = tf.data.TFRecordDataset(record_path).map(parse_tfrecord_fn)
    chunks = [model.encode(batch)[0].numpy() for batch in dataset.batch(batch_size)]
    if not chunks:
        raise ValueError("Training TFRecord is empty")
    return np.concatenate(chunks)
