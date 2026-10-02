"""Random candidate outfits that mimic the class mix of the training outfits."""

import re
import warnings
from pathlib import Path

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

    `unseen` products never appear in a training outfit, `seen` only those
    that do, `all` every catalog product.
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


def image_year(path):
    """Year at the start of an image file name (`2019_...jpg`), or None."""
    match = re.match(r"(\d{4})_", Path(path).name)
    return int(match.group(1)) if match else None


def catalogues(catalog, outfits):
    """Map each product ID to its catalogue.

    Products are grouped by the year of their image; years that appear together
    in a training outfit belong to the same catalogue. Products without a year
    form one catalogue, `unknown`.
    """
    years = {
        product: image_year(path)
        for product, path in zip(catalog["cod_modelo_color"], catalog["des_filename"])
    }
    parent = {year: year for year in set(years.values()) if year is not None}

    def root(year):
        while parent[year] != year:
            year = parent[year]
        return year

    for row in outfits[list(PRODUCT_CLASSES)].itertuples(index=False):
        found = sorted({years.get(p) for p in row if isinstance(p, str)} - {None})
        for year in found[1:]:
            parent[root(year)] = root(found[0])
    members = {}
    for year in parent:
        members.setdefault(root(year), []).append(year)
    names = {year: f"{min(group)}-{max(group)}" for group in members.values() for year in group}
    return pd.Series({product: names.get(year, "unknown") for product, year in years.items()})


def outfit_catalogues(outfits, catalogue):
    """Catalogue of each outfit (all its products share one)."""
    return outfits[list(PRODUCT_CLASSES)].apply(
        lambda row: catalogue.get(row.dropna().iloc[0], "unknown"), axis=1
    )


def sample_candidates_by_catalogue(catalog, outfits, pool, num_candidates, seed=42):
    """Candidates whose products all come from one catalogue.

    Each catalogue receives candidates in proportion to its training outfits
    and follows its own configuration frequencies.
    """
    catalogue = catalogues(catalog, outfits)
    of_outfit = outfit_catalogues(outfits, catalogue)
    shares = of_outfit.value_counts(normalize=True)
    counts = np.random.default_rng(seed).multinomial(num_candidates, shares.to_numpy())
    frames = []
    for offset, (name, count) in enumerate(zip(shares.index, counts), start=1):
        if count:
            members = catalog[catalog["cod_modelo_color"].map(catalogue) == name]
            frames.append(
                sample_candidates(
                    product_pools(members, outfits, pool),
                    configuration_probabilities(outfits[of_outfit == name]),
                    count,
                    seed + offset,
                )
            )
    candidates = pd.concat(frames, ignore_index=True)
    candidates["candidate_id"] = range(len(candidates))
    return candidates


def sample_candidates_for_product(product_id, catalog, outfits, pool, num_candidates, seed=42):
    """Candidates that all contain `product_id`, built from its catalogue."""
    product_id = str(product_id)
    product = catalog.loc[catalog["cod_modelo_color"] == product_id]
    if product.empty:
        raise ValueError(f"Unknown product {product_id}")
    slot = product["des_product_class"].iloc[0]
    catalogue = catalogues(catalog, outfits)
    of_outfit = outfit_catalogues(outfits, catalogue)
    members = catalog[catalog["cod_modelo_color"].map(catalogue) == catalogue[product_id]]
    pools = product_pools(members, outfits, pool)
    pools[slot] = np.array([product_id])
    same = outfits[of_outfit == catalogue[product_id]]
    configurations = configuration_probabilities(same if not same.empty else outfits)
    with_slot = configurations[configurations["pattern"].map(lambda p: p[PRODUCT_CLASSES.index(slot)])]
    if with_slot.empty:
        raise ValueError(f"No training configuration contains a {slot} product")
    return sample_candidates(pools, with_slot, num_candidates, seed)
