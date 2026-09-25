"""Join product metadata and retain outfits matching the configured rules."""

from pathlib import Path

import pandas as pd

from src.data_processing.group_manipulation import (
    create_configurations,
    select_valid_outfits,
)
from src.data_processing.product_processing import get_des_product_class
from src.utils.setup_utilities import ensure_parent, load_config


def build_valid_outfits_df(config_path=None):
    data = load_config(config_path)["data"]
    outfits = pd.read_csv(data["outfits_path"], dtype={"cod_modelo_color": str})
    products = pd.read_csv(data["products_path"], dtype={"cod_modelo_color": str})
    merged = outfits.merge(
        products,
        on="cod_modelo_color",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("Some outfit products have no product metadata")
    merged["des_product_class"] = merged.apply(get_des_product_class, axis=1)
    configurations = create_configurations(
        data["configurations_base"], data["optional_products"]
    )
    selected, _ = select_valid_outfits(
        merged, "des_product_class", "cod_outfit", "outfit", configurations
    )
    data_root = Path(data["outfits_path"]).parent.parent

    def image_path(value):
        path = Path(value)
        if path.parts and path.parts[0] == "datathon":
            path = data_root.joinpath(*path.parts[1:])
        elif not path.is_absolute():
            path = data_root.parent / path
        return str(path.resolve())

    selected["des_filename"] = selected["des_filename"].map(image_path)
    selected.to_csv(ensure_parent(data["valid_outfits_path"]), index=False)
    return selected


if __name__ == "__main__":
    build_valid_outfits_df()
