import json

import numpy as np
import pandas as pd
import pytest

from src.data_processing.schema import PRODUCT_CLASSES
from src.recommendation.candidates import sample_candidates
from src.recommendation.ranking import nearest_neighbors, recommend, write_index
from src.utils.setup_utilities import load_config


def fixture_catalog():
    data = load_config()["data"]
    rows, products = [], []
    for index, base in enumerate(
        [data["configurations_base"][0]] * 3 + [data["configurations_base"][1]]
    ):
        row = dict.fromkeys(PRODUCT_CLASSES, np.nan)
        row["cod_outfit"] = f"outfit-{index}"
        for slot in base:
            product = f"train-{slot}"
            row[slot] = product
            products.append((product, slot))
        rows.append(row)
    for slot in PRODUCT_CLASSES:
        products.extend([(f"new-0-{slot}", slot), (f"new-1-{slot}", slot)])
    return (
        pd.DataFrame(rows),
        pd.DataFrame(
            products, columns=["cod_modelo_color", "des_product_class"]
        ).drop_duplicates(),
        data,
    )


def test_frequency_sampling_and_unseen_products_are_reproducible():
    training, products, data = fixture_catalog()
    candidates, frequencies = sample_candidates(
        training, products, data, count=10000, seed=123
    )
    again, _ = sample_candidates(training, products, data, count=10000, seed=123)
    pd.testing.assert_frame_equal(candidates, again)
    assert frequencies.probability.sum() == 1
    assert sorted(frequencies.loc[frequencies.training_count > 0, "probability"]) == [
        0.25,
        0.75,
    ]
    assert frequencies.sampled_count.sum() == 10000
    observed = frequencies[frequencies.training_count > 0]
    np.testing.assert_allclose(
        observed.sampled_count / 10000, observed.probability, atol=0.02
    )
    for slot in PRODUCT_CLASSES:
        assert all(
            not product or product.startswith("new-") for product in candidates[slot]
        )
    assert candidates.candidate_id.is_unique
    assert len(frequencies) == 64
    assert candidates.signature.nunique() < len(
        candidates
    )  # Replacement is intentional.


def test_missing_pool_does_not_silently_change_distribution():
    training, products, data = fixture_catalog()
    products = products[products.cod_modelo_color != "new-0-Footwear"]
    products = products[products.cod_modelo_color != "new-1-Footwear"]
    with pytest.raises(ValueError, match="No unseen products"):
        sample_candidates(training, products, data)


def test_exact_blocked_distances_match_brute_force_and_ties():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(11, 5))
    reference[8] = reference[1]
    candidates = np.vstack([reference[1], rng.normal(size=(6, 5))])
    expected = np.linalg.norm(candidates[:, None, :] - reference[None, :, :], axis=2)
    distances, indices = nearest_neighbors(
        candidates, reference, candidate_block=2, reference_block=3
    )
    np.testing.assert_allclose(distances, expected.min(axis=1))
    np.testing.assert_array_equal(indices, expected.argmin(axis=1))
    assert distances[0] == 0 and indices[0] == 1
    with pytest.raises(ValueError, match="nonfinite"):
        nearest_neighbors(np.full((1, 5), np.nan), reference)
    with pytest.raises(ValueError, match="nonempty"):
        nearest_neighbors(candidates, reference[:0])


def test_index_filters_product_sorts_deduplicates_and_handles_no_results(tmp_path):
    training, products, data = fixture_catalog()
    candidates, _ = sample_candidates(training, products, data, count=4, seed=42)
    candidates.loc[1, list(PRODUCT_CLASSES)] = candidates.loc[
        0, list(PRODUCT_CLASSES)
    ].to_numpy()
    candidates.loc[1, "signature"] = candidates.loc[0, "signature"]
    path = tmp_path / "index.sqlite"
    write_index(path, candidates, [1, 1, 0.2, 0.4], ["001", "001", "003", "004"], {})
    product = candidates.iloc[0]["Footwear"]
    result = recommend(path, product, top_k=10)
    assert len({json.dumps(r["products"], sort_keys=True) for r in result}) == len(
        result
    )
    assert all(product in r["products"].values() for r in result)
    assert [r["distance"] for r in result] == sorted(r["distance"] for r in result)
    assert recommend(path, "' OR 1=1 --") == []
    assert recommend(path, "train-Footwear") == []
    with pytest.raises(ValueError, match="positive"):
        recommend(path, product, top_k=0)
