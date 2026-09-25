from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import yaml

from scripts.data_processing.build_image_arrays import build_image_arrays
from scripts.data_processing.build_outfit_w_products_df import build_outfit_w_products
from scripts.data_processing.build_valid_outfits_df import build_valid_outfits_df
from scripts.run_pipeline import run_pipeline
from scripts.training.train_vae import train_outfit_embeddings
from src.data_processing.group_manipulation import (
    create_combinations,
    create_configurations,
    select_valid_outfits,
)
from src.data_processing.image_processing import parse_tfrecord_fn, serialize_image
from src.data_processing.schema import PRODUCT_CLASSES
from src.utils.pipeline_validation import validation_frame
from src.utils.setup_utilities import load_config
from src.utils.train_utilities import compute_loss, train_step


def test_all_optional_subsets():
    config = load_config()["data"]
    configurations = create_configurations(
        config["configurations_base"], config["optional_products"]
    )
    assert len(configurations) == 64
    assert len({frozenset(c) for c in configurations}) == 64
    assert config["configurations_base"][0] + ["Bags", "Glasses"] in configurations
    assert create_configurations([["Tops"]], []) == [["Tops"]]
    assert create_combinations([]) == []


def test_selection_handles_missing_sizes_nine_items_and_duplicates():
    config = load_config()["data"]
    configurations = create_configurations(
        config["configurations_base"], config["optional_products"]
    )
    classes = config["configurations_base"][1] + config["optional_products"]
    rows = [("valid", str(i), c, "image.jpg") for i, c in enumerate(classes)]
    rows += [
        ("invalid", str(i), c, "image.jpg") for i, c in enumerate(classes + ["Tops"])
    ]
    frame = pd.DataFrame(
        rows,
        columns=["cod_outfit", "cod_modelo_color", "des_product_class", "des_filename"],
    )
    selected, counts = select_valid_outfits(
        frame, "des_product_class", "cod_outfit", "outfit", configurations
    )
    assert selected.cod_outfit.unique().tolist() == ["valid"]
    assert len(selected) == 9
    assert sum(counts.values()) == 1
    empty, counts = select_valid_outfits(
        frame.iloc[:0], "des_product_class", "cod_outfit", "outfit", configurations
    )
    assert empty.empty and counts == {}


@pytest.mark.parametrize("brightness", [0.0, 0.5, 1.0])
def test_tfrecord_roundtrip_preserves_brightness(brightness):
    image = tf.fill((1, 16, 16, 3), brightness)
    decoded = parse_tfrecord_fn(serialize_image(image))
    assert decoded.dtype == tf.float32
    assert decoded.shape == (16, 16, 3)
    np.testing.assert_allclose(decoded.numpy(), brightness, atol=2 / 255)


def test_csv_to_records_pipeline(tmp_path):
    data = tmp_path / "data"
    (data / "tabular").mkdir(parents=True)
    (data / "images").mkdir()
    tf.io.write_file(
        str(data / "images" / "item.jpg"),
        tf.io.encode_jpeg(tf.zeros((16, 16, 3), tf.uint8)),
    )
    products = pd.DataFrame(
        [
            ["001", "Dresses, jumpsuits and Complete set", "Dresses", "Dress"],
            ["002", "Accesories, Swim and Intimate", "Footwear", "Boots"],
            ["003", "Accesories, Swim and Intimate", "Jewellery", "Earrings"],
        ],
        columns=[
            "cod_modelo_color",
            "des_product_category",
            "des_product_family",
            "des_product_type",
        ],
    )
    unseen = products.copy()
    unseen["cod_modelo_color"] = ["004", "005", "006"]
    products = pd.concat([products, unseen], ignore_index=True)
    products["des_filename"] = "datathon/images/item.jpg"
    products.to_csv(data / "tabular" / "product_data.csv", index=False)
    pd.DataFrame(
        {"cod_outfit": [1, 1, 1], "cod_modelo_color": ["001", "002", "003"]}
    ).to_csv(data / "tabular" / "outfit_data.csv", index=False)
    config = yaml.safe_load(Path("config.yml").read_text())
    config["training"]["train_sample_indices"] = [0]
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config))
    selected = build_valid_outfits_df(config_path)
    assert selected.cod_modelo_color.tolist() == ["001", "002", "003"]
    wide = build_outfit_w_products(config_path)
    assert list(wide.columns) == ["cod_outfit", *PRODUCT_CLASSES]
    build_image_arrays(config_path=config_path)
    paths = load_config(config_path)["training"]
    records = list(
        tf.data.TFRecordDataset(paths["train_dataset_path"]).as_numpy_iterator()
    )
    previews = list(
        tf.data.TFRecordDataset(paths["train_sample_path"]).as_numpy_iterator()
    )
    assert records == previews and len(records) == 1
    image = parse_tfrecord_fn(records[0]).numpy()
    assert image.shape == (128, 960, 3)
    assert image[:, :96].mean() > 0.98  # Missing Tops slot stays white.
    assert image[:, 3 * 96 : 4 * 96].mean() < 0.02  # Dress stays black.
    with pytest.raises(ValueError, match="Channel-stacked"):
        build_image_arrays(compact=True, config_path=config_path)

    config["training"].update(epochs=1, batch_size=2, progress_bar=False)
    config_path.write_text(yaml.safe_dump(config))
    model, history = train_outfit_embeddings(config_path)
    assert len(history) == 1 and np.isfinite(history[0])
    assert Path(paths["model_weights_path"]).is_file()
    assert Path(paths["metrics_path"]).is_file()
    assert Path(paths["model_manifest_path"]).is_file()
    assert (Path(paths["epoch_images_path"]) / "image_at_epoch_0001.png").is_file()
    import pickle

    with open(paths["model_data_path"], "rb") as stream:
        exported = pickle.load(stream)
    assert exported["latent_space"].shape == (1, 256)
    assert exported["embedding_type"] == "posterior_mean"
    assert model.decode(tf.zeros((1, model.latent_dim))).shape == (1, 128, 960, 3)

    from scripts.recommendation.build_index import build_index
    from scripts.recommendation.generate_candidates import generate_candidates
    from src.recommendation.ranking import recommend

    config["recommendation"].update(num_candidates=20, encoding_batch_size=3)
    config_path.write_text(yaml.safe_dump(config))
    candidates, _ = generate_candidates(config_path)
    metadata = build_index(config_path)
    rec = load_config(config_path)["recommendation"]
    result = recommend(rec["index_path"], "004")
    assert len(result) == 1  # Repeated random draws of the same outfit are collapsed.
    assert set(result[0]["products"].values()) == {"004", "005", "006"}
    assert result[0]["nearest_training_outfit"] == "1"
    assert result[0]["distance"] < 1e-3
    assert metadata["embedding_type"] == "posterior_mean"
    assert metadata["distinct_outfits"] == 1
    with np.load(rec["embeddings_path"], allow_pickle=False) as arrays:
        assert arrays["candidate_ids"].tolist() == candidates.candidate_id.tolist()
        assert arrays["reference_ids"].tolist() == ["1"]
    report = validation_frame(config_path)
    assert report.status.eq("PASS").all(), report[report.status != "PASS"]
    dry_run = run_pipeline(
        config_path,
        from_stage="images",
        to_stage="candidates",
        dry_run=True,
        device="cpu",
    )
    assert dry_run["status"] == "DRY_RUN"
    assert [stage["name"] for stage in dry_run["stages"]] == [
        "images",
        "train",
        "candidates",
    ]
    train_command = next(
        stage["command"] for stage in dry_run["stages"] if stage["name"] == "train"
    )
    assert train_command[-2:] == ["--device", "cpu"]
    # Checkpoint provenance is checked before attempting to load model weights.
    weights_path = Path(paths["model_weights_path"])
    with weights_path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="weights no longer matches"):
        build_index(config_path)
    with weights_path.open("r+b") as stream:
        stream.truncate(weights_path.stat().st_size - len(b"changed"))
    image_path = data / "images" / "item.jpg"
    hidden_path = image_path.with_suffix(".hidden")
    image_path.rename(hidden_path)
    with pytest.raises(FileNotFoundError, match="candidate product images missing"):
        build_index(config_path)
    hidden_path.rename(image_path)
    # A modified candidate file must never silently retain stale scores or lineage.
    with open(rec["candidates_path"], "a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed after generation"):
        build_index(config_path)


def test_training_step_updates_weights_and_returns_loss():
    model = tf.keras.Sequential(
        [tf.keras.layers.Input((2, 2, 3)), tf.keras.layers.Conv2D(3, 1)]
    )
    batch = tf.ones((2, 2, 2, 3))
    before = [v.numpy().copy() for v in model.trainable_variables]
    optimizer = tf.keras.optimizers.Adam(0.01)
    loss = train_step(model, batch, optimizer)
    assert np.isfinite(loss.numpy())
    assert any(
        not np.array_equal(old, new.numpy())
        for old, new in zip(before, model.trainable_variables)
    )
    assert np.isfinite(compute_loss(model, batch).numpy())
