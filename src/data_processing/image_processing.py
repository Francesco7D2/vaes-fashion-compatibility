# -*- coding: utf-8 -*-

import tensorflow as tf

def load_and_preprocess_image(image_path, image_width, image_height):
    """
    Loads and preprocesses an image from the specified path.

    Parameters:
    - image_path (str): The file path of the image.
    - image_width (int): Target width for resizing the image.
    - image_height (int): Target height for resizing the image.

    Returns:
    - tf.Tensor: Preprocessed image as a TensorFlow tensor.

    Example:
    load_and_preprocess_image('path/to/image.jpg', 224, 224)
    """
    img = tf.io.read_file(image_path.iloc[0])
    img = tf.image.decode_image(img, channels=3)  
    img = tf.image.resize(img, [image_height, image_width])
    img = img / 255.0
    img = tf.expand_dims(img, 0)
    return img


def create_white_image(width, height):
    """
    Creates a white image with the specified width and height.

    Parameters:
    - width (int): Width of the white image.
    - height (int): Height of the white image.

    Returns:
    - tf.Tensor: White image as a TensorFlow tensor.

    Example:
    create_white_image(224, 224)
    """
    white_tensor = tf.ones([height, width, 3], dtype=tf.float32)
    white_image = white_tensor
    white_image = tf.expand_dims(white_image, 0)
    return white_image

def _bytes_feature(value):
    # Ensure that the image data is of type uint8
    value = tf.image.convert_image_dtype(value, dtype=tf.uint8)

    # Remove the singleton batch dimension if it exists
    value = tf.squeeze(value, axis=0) if len(value.shape) == 4 else value

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def parse_tfrecord_fn(example_proto):
    feature_description = {'image': tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.io.decode_jpeg(example['image'], channels=3)
    return img

