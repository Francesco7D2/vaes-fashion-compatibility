"""Frequency-weighted candidate sampling recovered from the prediction notebook."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.group_manipulation import create_configurations
from src.data_processing.product_processing import get_des_product_class
from src.data_processing.schema import PRODUCT_CLASSES


def resolve_image_path(value, data_root):
    path = Path(value)
    if path.parts and path.parts[0] == "datathon":
        path = Path(data_root).joinpath(*path.parts[1:])
    elif not path.is_absolute():
        path = Path(data_root).parent / path
    return str(path.resolve())


def read_catalog(path):
    products = pd.read_csv(path, dtype={"cod_modelo_color": str})
    if (
        products.cod_modelo_color.isna().any()
        or products.cod_modelo_color.duplicated().any()
    ):
        raise ValueError("Catalog requires unique, nonempty product IDs")
    products["des_product_class"] = products.apply(get_des_product_class, axis=1)
    return products


def read_outfits(path):
    frame = pd.read_csv(
        path, dtype={"cod_outfit": str, **dict.fromkeys(PRODUCT_CLASSES, str)}
    )
    if (
        frame.empty
        or frame.cod_outfit.isna().any()
        or frame.cod_outfit.duplicated().any()
    ):
        raise ValueError("Training outfits require unique IDs and at least one row")
    return frame[["cod_outfit", *PRODUCT_CLASSES]]


def sample_candidates(training, products, data_config, count=100000, seed=42):
    """Sample with replacement, as in the report; preserve every draw and its ID.

    Configuration probabilities are empirical frequencies in the training table.
    Products are uniform within class and exclude all training product IDs.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("num_candidates must be a positive integer")
    configurations = create_configurations(
        data_config["configurations_base"], data_config["optional_products"]
    )
    keys = [frozenset(c) for c in configurations]
    if any(
        len(c) != len(set(c)) or not set(c) <= set(PRODUCT_CLASSES)
        for c in configurations
    ) or len(set(keys)) != len(keys):
        raise ValueError(
            "Configurations must be unique sets of supported product classes"
        )
    positions = {key: i for i, key in enumerate(keys)}
    counts = np.zeros(len(configurations), dtype=np.int64)
    catalog = products.set_index("cod_modelo_color")["des_product_class"].to_dict()
    seen = set()
    for _, row in training.iterrows():
        present = [slot for slot in PRODUCT_CLASSES if pd.notna(row[slot])]
        if frozenset(present) not in positions:
            raise ValueError("Training table contains an unconfigured outfit")
        for slot in present:
            product = str(row[slot])
            if catalog.get(product) != slot:
                raise ValueError(
                    f"Training product {product} has missing or inconsistent class metadata"
                )
            seen.add(product)
        counts[positions[frozenset(present)]] += 1
    if not counts.sum():
        raise ValueError("Training outfits are empty")
    unseen = products[~products.cod_modelo_color.isin(seen)]
    pools = {
        slot: sorted(
            unseen.loc[unseen.des_product_class == slot, "cod_modelo_color"].tolist()
        )
        for slot in PRODUCT_CLASSES
    }
    needed = set().union(*(set(configurations[i]) for i in np.flatnonzero(counts)))
    missing = sorted(slot for slot in needed if not pools[slot])
    if missing:
        raise ValueError(
            f"No unseen products for observed classes: {missing}. Sampling probabilities were not changed."
        )
    probabilities = counts / counts.sum()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(configurations), size=count, p=probabilities)
    candidates = pd.DataFrame("", index=range(count), columns=PRODUCT_CLASSES)
    for conf_id in np.unique(sampled):
        rows = np.flatnonzero(sampled == conf_id)
        for slot in configurations[conf_id]:
            candidates.loc[rows, slot] = rng.choice(pools[slot], size=len(rows))
    candidates.insert(0, "configuration_id", sampled)
    candidates.insert(0, "candidate_id", [f"candidate_{i:06d}" for i in range(count)])
    candidates["signature"] = [
        hashlib.sha256(
            json.dumps(list(row), separators=(",", ":")).encode()
        ).hexdigest()
        for row in candidates[list(PRODUCT_CLASSES)].itertuples(index=False, name=None)
    ]
    frequencies = pd.DataFrame(
        {
            "configuration_id": range(len(configurations)),
            "classes": [json.dumps(c) for c in configurations],
            "training_count": counts,
            "probability": probabilities,
            "sampled_count": np.bincount(sampled, minlength=len(configurations)),
        }
    )
    return candidates, frequencies
