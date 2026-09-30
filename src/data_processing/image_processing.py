"""Float32 images in [0, 1] in memory; JPEG bytes in TFRecords."""

import pandas as pd
import tensorflow as tf

from src.data_processing.schema import PRODUCT_CLASSES


def load_and_preprocess_image(image_path, image_width, image_height):
    if hasattr(image_path, "iloc"):
        image_path = image_path.iloc[0]
    image = tf.io.read_file(str(image_path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [image_height, image_width])
    return tf.expand_dims(image, 0)


def create_white_image(width, height):
    return tf.ones([1, height, width, 3], dtype=tf.float32)


def build_outfit_strip(outfit, load_image, image_width, image_height):
    """Concatenate one image per slot (white if missing) into a 1 x H x 10W x 3 strip.

    `outfit` maps slot name to product ID (or NaN); `load_image` maps a product
    ID to a 1 x H x W x 3 float tensor. Shared by training-data and inference code.
    """
    white = create_white_image(image_width, image_height)
    images = [
        white if pd.isna(outfit[slot]) else load_image(outfit[slot])
        for slot in PRODUCT_CLASSES
    ]
    return tf.concat(images, axis=2)


def _bytes_feature(value):
    """Encode normalized floats (or uint8 pixels) without a second division."""
    value = tf.convert_to_tensor(value)
    value = tf.image.convert_image_dtype(value, tf.uint8, saturate=True)
    if value.shape.rank == 4:
        value = tf.squeeze(value, axis=0)
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def serialize_image(image):
    features = tf.train.Features(feature={"image": _bytes_feature(image)})
    return tf.train.Example(features=features).SerializeToString()


def parse_tfrecord_fn(example_proto):
    description = {"image": tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example_proto, description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    return tf.image.convert_image_dtype(image, tf.float32)
