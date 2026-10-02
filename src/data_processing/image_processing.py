"""uint8 H x W x 3 images in memory; JPEG bytes in the strip archives (.npz)."""

import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from src.data_processing.schema import PRODUCT_CLASSES

JPEG_QUALITY = 95


def to_uint8(image):
    """Floats in [0, 1] -> uint8 pixels (rounded, saturated)."""
    return np.clip(np.rint(np.asarray(image) * 255), 0, 255).astype(np.uint8)


def load_and_preprocess_image(image_path, image_width, image_height):
    """Decode as RGB and resize to H x W x 3 uint8.

    Bilinear with half-pixel centres and no antialiasing.
    """
    if hasattr(image_path, "iloc"):
        image_path = image_path.iloc[0]
    with Image.open(image_path) as image:
        pixels = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32))
    resized = F.interpolate(
        pixels.permute(2, 0, 1)[None] / 255,
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    return to_uint8(resized[0].permute(1, 2, 0).numpy())


def create_white_image(width, height):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def build_outfit_strip(outfit, load_image, image_width, image_height):
    """Concatenate one image per slot (white if missing) into an H x 10W x 3 strip.

    `outfit` maps slot name to product ID (or NaN); `load_image` maps a product
    ID to an H x W x 3 uint8 image. Shared by training-data and inference code.
    """
    white = create_white_image(image_width, image_height)
    images = [
        white if pd.isna(outfit[slot]) else load_image(outfit[slot])
        for slot in PRODUCT_CLASSES
    ]
    return np.concatenate(images, axis=1)


def encode_jpeg(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, "JPEG", quality=JPEG_QUALITY, subsampling="4:2:0")
    return buffer.getvalue()


def decode_jpeg(data):
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"))


def jpeg_round_trip(image):
    """What a strip looks like after being stored in and read from an archive."""
    return decode_jpeg(encode_jpeg(image))


def save_strips(path, encoded):
    """Write JPEG-encoded strips as one flat byte buffer plus offsets (no pickle)."""
    lengths = np.array([len(item) for item in encoded], dtype=np.int64)
    with open(path, "wb") as stream:  # a path would get ".npz" appended
        np.savez(
            stream,
            jpeg=np.frombuffer(b"".join(encoded), dtype=np.uint8),
            offsets=np.concatenate([[0], np.cumsum(lengths)]),
        )


def load_strips(path, workers=None):
    """Decode every strip of an archive into an N x H x W x 3 uint8 array.

    Decoding runs in threads (Pillow releases the GIL); a few seconds for the
    full training set, which then stays in memory for the whole run.
    """
    with np.load(path) as archive:
        buffer, offsets = archive["jpeg"].tobytes(), archive["offsets"]
    encoded = [buffer[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
    with ThreadPoolExecutor(workers or os.cpu_count()) as pool:
        images = list(pool.map(decode_jpeg, encoded))
    if not images:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(images)
