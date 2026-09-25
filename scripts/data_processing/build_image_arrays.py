"""Build full and preview TFRecords in one pass."""

import argparse
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.data_processing.image_processing import (
    create_white_image,
    load_and_preprocess_image,
    serialize_image,
)
from src.data_processing.schema import PRODUCT_CLASSES
from src.utils.setup_utilities import ensure_parent, load_config


def build_image_arrays(compact=False, config_path=None):
    if compact:
        raise ValueError(
            "Channel-stacked images are incompatible with JPEG and this model"
        )
    config = load_config(config_path)
    data, training = config["data"], config["training"]
    products = pd.read_csv(data["valid_outfits_path"], dtype={"cod_modelo_color": str})
    outfits = pd.read_csv(
        data["outfits_w_products_path"], dtype={slot: str for slot in PRODUCT_CLASSES}
    )
    paths = products.drop_duplicates("cod_modelo_color").set_index("cod_modelo_color")[
        "des_filename"
    ]
    width, height = data["image_width"], data["image_height"]
    white = create_white_image(width, height)
    samples = set(training["train_sample_indices"])
    if any(index < 0 or index >= len(outfits) for index in samples):
        raise ValueError("train_sample_indices must refer to existing outfit rows")
    dataset_path = Path(ensure_parent(training["train_dataset_path"]))
    sample_path = Path(ensure_parent(training["train_sample_path"]))
    temporary_dataset = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    temporary_sample = sample_path.with_suffix(sample_path.suffix + ".tmp")
    try:
        with (
            tf.io.TFRecordWriter(str(temporary_dataset)) as writer,
            tf.io.TFRecordWriter(str(temporary_sample)) as sample_writer,
        ):
            for index, (_, row) in enumerate(
                tqdm(outfits.iterrows(), total=len(outfits))
            ):
                images = []
                for slot in PRODUCT_CLASSES:
                    product = row[slot]
                    images.append(
                        white
                        if pd.isna(product)
                        else load_and_preprocess_image(
                            paths.loc[product], width, height
                        )
                    )
                record = serialize_image(tf.concat(images, axis=2))
                writer.write(record)
                if index in samples:
                    sample_writer.write(record)
        os.replace(temporary_dataset, dataset_path)
        os.replace(temporary_sample, sample_path)
    finally:
        temporary_dataset.unlink(missing_ok=True)
        temporary_sample.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    build_image_arrays(config_path=parser.parse_args().config)
