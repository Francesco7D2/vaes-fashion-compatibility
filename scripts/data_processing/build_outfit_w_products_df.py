"""Create one row per outfit with one product ID per fixed image slot."""

import argparse

import pandas as pd

from src.data_processing.schema import PRODUCT_CLASSES
from src.utils.setup_utilities import ensure_parent, load_config


def build_outfit_w_products(config_path=None):
    data = load_config(config_path)["data"]
    frame = pd.read_csv(data["valid_outfits_path"], dtype={"cod_modelo_color": str})
    # Reject duplicate classes within an outfit instead of overwriting products.
    outfits = frame.pivot(
        index="cod_outfit", columns="des_product_class", values="cod_modelo_color"
    )
    outfits = outfits.reindex(columns=PRODUCT_CLASSES).reset_index()
    outfits.to_csv(ensure_parent(data["outfits_w_products_path"]), index=False)
    return outfits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    build_outfit_w_products(parser.parse_args().config)
