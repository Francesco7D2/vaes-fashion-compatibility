"""Random candidate outfits that mimic the class mix of the training outfits."""

import warnings

import numpy as np
import pandas as pd

from src.data_processing.schema import PRODUCT_CLASSES

POOLS = ("unseen", "seen", "all")


def configuration_probabilities(outfits):
    """Share of training outfits per occupied-slot pattern.

    `outfits` is the wide table (one column per slot, NaN when empty). Returns
    a DataFrame with a boolean tuple `pattern` (aligned with PRODUCT_CLASSES),
    `count` and `probability`.
    """
    if outfits.empty:
        raise ValueError("No training outfits to derive configurations from")
    patterns = pd.Series(
        list(map(tuple, outfits[list(PRODUCT_CLASSES)].notna().to_numpy()))
    )
    counts = patterns.value_counts().rename_axis("pattern").reset_index(name="count")
    counts["probability"] = counts["count"] / counts["count"].sum()
    return counts


def product_pools(catalog, outfits, pool="unseen"):
    """Map each class to the product IDs candidates may draw from.

    `unseen` products never appear in a training outfit (the report's setting),
    `seen` only those that do, `all` every catalog product.
    """
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}")
    used = set(outfits[list(PRODUCT_CLASSES)].stack().astype(str))
    in_training = catalog["cod_modelo_color"].astype(str).isin(used)
    selected = {"unseen": ~in_training, "seen": in_training, "all": in_training | ~in_training}
    catalog = catalog[selected[pool]]
    return {
        slot: catalog.loc[catalog["des_product_class"] == slot, "cod_modelo_color"]
        .astype(str)
        .to_numpy()
        for slot in PRODUCT_CLASSES
    }


def sample_candidates(pools, configurations, num_candidates, seed=42):
    """Draw random outfits: a pattern by training frequency, then a product per slot.

    Patterns needing a class whose pool is empty are dropped and probabilities
    renormalised. Returns `candidate_id` plus one column per slot.
    """
    feasible = configurations[
        configurations["pattern"].map(
            lambda pattern: all(
                len(pools[slot]) for slot, used in zip(PRODUCT_CLASSES, pattern) if used
            )
        )
    ]
    if feasible.empty:
        raise ValueError("Product pools cannot fill any training configuration")
    if len(feasible) < len(configurations):
        warnings.warn(
            f"Dropped {len(configurations) - len(feasible)} configurations whose "
            "classes have no products in the pool"
        )
    rng = np.random.default_rng(seed)
    probabilities = feasible["probability"].to_numpy()
    chosen = rng.choice(
        len(feasible), size=num_candidates, p=probabilities / probabilities.sum()
    )
    masks = np.array(feasible["pattern"].tolist(), dtype=bool)[chosen]
    candidates = pd.DataFrame(index=range(num_candidates), columns=PRODUCT_CLASSES, dtype=object)
    for column, slot in enumerate(PRODUCT_CLASSES):
        rows = np.flatnonzero(masks[:, column])
        if len(rows):
            candidates.loc[rows, slot] = rng.choice(pools[slot], size=len(rows))
    candidates.insert(0, "candidate_id", range(num_candidates))
    return candidates
