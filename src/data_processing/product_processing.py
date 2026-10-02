from pathlib import Path

import pandas as pd

from src.data_processing.schema import PRODUCT_CLASSES


def get_des_product_class(row: pd.Series) -> pd.Series:
    """
    Extracts the desired product class based on specific conditions.

    Parameters:
    - row (pd.Series): The input row from a Pandas DataFrame containing
                      information about a product.

    Returns:
    - pd.Series: The extracted product class information.

    Conditions:
    - If 'des_product_category' is 'Accessories, Swim and Intimate' and
      'des_product_family' is not 'Jewellery', the function returns the
      'des_product_family'.
    - If 'des_product_family' is 'Jewellery', the function returns the
      'des_product_type'.
    - Otherwise, the function returns 'des_product_category'.
    """
    if (
        row["des_product_category"] == "Accesories, Swim and Intimate"
        and row["des_product_family"] != "Jewellery"
    ):
        return row["des_product_family"]
    elif row["des_product_family"] == "Jewellery":
        return row["des_product_type"]
    else:
        return row["des_product_category"]


def resolve_image_path(value, data_root):
    """Map `datathon/...` to the data directory; other relative paths to its parent."""
    path = Path(value)
    if path.parts and path.parts[0] == "datathon":
        path = Path(data_root).joinpath(*path.parts[1:])
    elif not path.is_absolute():
        path = Path(data_root).parent / path
    return str(path.resolve())


def build_product_catalog(products, data_root, require_images=False):
    """One row per product of a modelled class, with a resolved image path.

    Columns: cod_modelo_color, des_product_class, des_filename.
    """
    catalog = products.copy()
    catalog["des_product_class"] = catalog.apply(get_des_product_class, axis=1)
    catalog = catalog[catalog["des_product_class"].isin(PRODUCT_CLASSES)]
    catalog = catalog.drop_duplicates("cod_modelo_color")
    catalog["des_filename"] = catalog["des_filename"].map(
        lambda value: resolve_image_path(value, data_root)
    )
    if require_images:
        catalog = catalog[catalog["des_filename"].map(lambda f: Path(f).is_file())]
    return catalog[["cod_modelo_color", "des_product_class", "des_filename"]].reset_index(
        drop=True
    )
