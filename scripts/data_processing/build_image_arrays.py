"""Build full and preview strip archives in one pass."""

from functools import lru_cache

import pandas as pd
from tqdm import tqdm

from src.data_processing.image_processing import (
    build_outfit_strip,
    encode_jpeg,
    load_and_preprocess_image,
    save_strips,
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
    samples = list(dict.fromkeys(training["train_sample_indices"]))
    if any(index < 0 or index >= len(outfits) for index in samples):
        raise ValueError("train_sample_indices must refer to existing outfit rows")

    @lru_cache(maxsize=None)
    def load_image(product):
        return load_and_preprocess_image(paths.loc[product], width, height)

    records = [
        encode_jpeg(build_outfit_strip(row, load_image, width, height))
        for _, row in tqdm(outfits.iterrows(), total=len(outfits))
    ]
    save_strips(ensure_parent(training["train_dataset_path"]), records)
    # Previews follow the configured order, so titles can use the same list.
    save_strips(
        ensure_parent(training["train_sample_path"]), [records[i] for i in samples]
    )


if __name__ == "__main__":
    build_image_arrays()
