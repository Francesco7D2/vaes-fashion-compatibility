import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.data_processing.schema import PRODUCT_CLASSES
from src.generation.candidates import (
    configuration_probabilities,
    product_pools,
    sample_candidates,
)
from src.generation.embeddings import ProductImages, encode_outfits
from src.generation.retrieval import nearest_training_outfit, rank_for_product
from src.models.vae256 import Conv_VAE, load_trained_model


def wide(rows):
    frame = pd.DataFrame(rows, columns=PRODUCT_CLASSES)
    return frame.astype(object).where(frame.notna(), np.nan)


@pytest.fixture
def training_outfits():
    # Dress outfits (2x) and one top/bottom outfit.
    dress = [np.nan, np.nan, np.nan, "d1", "f1", np.nan, np.nan, "e1", np.nan, np.nan]
    dress_bag = [np.nan, np.nan, np.nan, "d2", "f2", "b1", np.nan, "e2", np.nan, np.nan]
    separates = ["t1", "b2", np.nan, np.nan, "f3", np.nan, np.nan, "e3", np.nan, np.nan]
    return wide([dress, dress_bag, separates])


@pytest.fixture
def catalog():
    classes = {
        "Dresses, jumpsuits and Complete set": ["d1", "d2", "d9"],
        "Footwear": ["f1", "f2", "f3", "f9"],
        "Earrings": ["e1", "e2", "e3", "e9"],
        "Tops": ["t1", "t9"],
        "Bottoms": ["b2", "b9"],
        "Bags": ["b1", "bag9"],
    }
    rows = [(p, c, f"{p}.jpg") for c, ps in classes.items() for p in ps]
    return pd.DataFrame(rows, columns=["cod_modelo_color", "des_product_class", "des_filename"])


def test_configuration_probabilities_follow_training_patterns(training_outfits):
    configs = configuration_probabilities(training_outfits)
    assert configs["count"].sum() == 3
    assert configs["probability"].sum() == pytest.approx(1)
    assert len(configs) == 3
    with pytest.raises(ValueError):
        configuration_probabilities(training_outfits.iloc[:0])


def test_pools_split_seen_and_unseen(catalog, training_outfits):
    unseen = product_pools(catalog, training_outfits, "unseen")
    assert sorted(unseen["Footwear"]) == ["f9"]
    assert sorted(unseen["Bags"]) == ["bag9"]
    seen = product_pools(catalog, training_outfits, "seen")
    assert sorted(seen["Footwear"]) == ["f1", "f2", "f3"]
    assert len(product_pools(catalog, training_outfits, "all")["Footwear"]) == 4
    with pytest.raises(ValueError):
        product_pools(catalog, training_outfits, "bogus")


def test_sampled_candidates_respect_patterns_and_pools(catalog, training_outfits):
    pools = product_pools(catalog, training_outfits, "unseen")
    candidates = sample_candidates(
        pools, configuration_probabilities(training_outfits), 200, seed=0
    )
    assert len(candidates) == 200
    assert candidates["Footwear"].eq("f9").all()
    assert candidates["Earrings"].eq("e9").all()
    assert candidates["Glasses"].isna().all()
    # Every candidate is a dress or a top/bottom outfit, never both or neither.
    dress = candidates["Dresses, jumpsuits and Complete set"].notna()
    tops = candidates["Tops"].notna()
    assert (dress ^ tops).all()
    assert (candidates["Bottoms"].notna() == tops).all()
    again = sample_candidates(pools, configuration_probabilities(training_outfits), 200, 0)
    pd.testing.assert_frame_equal(candidates, again)


def test_infeasible_configurations_are_dropped(catalog, training_outfits):
    pools = product_pools(catalog, training_outfits, "unseen")
    pools["Tops"] = pools["Tops"][:0]
    with pytest.warns(UserWarning, match="Dropped"):
        candidates = sample_candidates(
            pools, configuration_probabilities(training_outfits), 50, seed=1
        )
    assert candidates["Tops"].isna().all()
    pools["Dresses, jumpsuits and Complete set"] = pools["Tops"]
    with pytest.raises(ValueError, match="cannot fill"):
        sample_candidates(pools, configuration_probabilities(training_outfits), 5)


def test_nearest_training_outfit_matches_brute_force():
    rng = np.random.default_rng(0)
    training, candidates = rng.normal(size=(40, 8)), rng.normal(size=(25, 8))
    distances, indices = nearest_training_outfit(candidates, training, chunk_size=7)
    full = np.linalg.norm(candidates[:, None] - training[None], axis=2)
    np.testing.assert_allclose(distances, full.min(axis=1), atol=1e-6)
    np.testing.assert_array_equal(indices, full.argmin(axis=1))
    exact, _ = nearest_training_outfit(training[:3], training)
    np.testing.assert_allclose(exact, 0, atol=1e-6)
    with pytest.raises(ValueError):
        nearest_training_outfit(candidates[:, :4], training)


def test_rank_for_product_orders_by_distance(training_outfits):
    generated = training_outfits.copy()
    generated["distance"] = [0.5, 0.1, 0.9]
    best = rank_for_product(generated, "f1", k=3)
    assert best.index.tolist() == [0]
    generated.loc[2, "Footwear"] = "f1"
    assert rank_for_product(generated, "f1", k=2).index.tolist() == [0, 2]
    assert rank_for_product(generated, "f1", k=2, worst=True).index.tolist() == [2, 0]
    assert rank_for_product(generated, "missing").empty


def test_encoding_shapes_determinism_and_weight_loading(tmp_path, training_outfits):
    files = {}
    for index, product in enumerate(["d1", "f1", "e1"]):
        path = tmp_path / f"{product}.jpg"
        tf.io.write_file(
            str(path), tf.io.encode_jpeg(tf.fill((20, 20, 3), tf.constant(60 * index + 40, tf.uint8)))
        )
        files[product] = str(path)
    images = ProductImages(pd.Series(files), 96, 128)
    first = images("d1")
    assert first.shape == (1, 128, 96, 3) and first.dtype == tf.float32
    outfit = wide([[np.nan] * 3 + ["d1", "f1", np.nan, np.nan, "e1", np.nan, np.nan]] * 3)
    model = Conv_VAE()
    model(tf.zeros((1, 128, 960, 3)))
    embeddings = encode_outfits(model, outfit, images, batch_size=2, progress=False)
    assert embeddings.shape == (3, 256)
    np.testing.assert_allclose(embeddings[0], embeddings[2], atol=1e-5)
    raw = encode_outfits(
        model, outfit, images, batch_size=3, match_training_jpeg=False, progress=False
    )
    assert np.abs(raw - embeddings).max() < 1  # JPEG round trip is a small perturbation
    assert encode_outfits(model, outfit.iloc[:0], images, progress=False).shape == (0, 256)
    weights = tmp_path / "model.weights.h5"
    model.save_weights(weights)
    restored = load_trained_model(weights)
    np.testing.assert_allclose(
        encode_outfits(restored, outfit, images, progress=False), embeddings, atol=1e-5
    )


@pytest.mark.parametrize(
    "name", ["training", "outfit-generation"]
)
def test_notebooks_are_valid_and_output_free(name):
    import json
    from pathlib import Path

    notebook = json.loads(Path(f"notebooks/{name}.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), name, "exec")
            assert not cell["outputs"] and cell["execution_count"] is None

