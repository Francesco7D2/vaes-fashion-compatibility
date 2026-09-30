from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

from scripts.data_processing.build_image_arrays import build_image_arrays
from scripts.data_processing.build_outfit_w_products_df import build_outfit_w_products
from scripts.data_processing.build_valid_outfits_df import build_valid_outfits_df
from scripts.training.train_vae import train_outfit_embeddings
from src.data_processing.group_manipulation import (
    create_combinations,
    create_configurations,
    select_valid_outfits,
)
from src.data_processing.image_processing import (
    encode_jpeg,
    jpeg_round_trip,
    load_and_preprocess_image,
    load_strips,
    save_strips,
)
from src.data_processing.schema import PRODUCT_CLASSES
from src.utils.setup_utilities import load_config
from src.utils.train_utilities import compute_loss, make_optimizer, train_step


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


@pytest.mark.parametrize("brightness", [0, 128, 255])
def test_jpeg_roundtrip_preserves_brightness(brightness):
    decoded = jpeg_round_trip(np.full((16, 16, 3), brightness, np.uint8))
    assert decoded.dtype == np.uint8
    assert decoded.shape == (16, 16, 3)
    np.testing.assert_allclose(decoded, brightness, atol=2)


def test_strip_archive_roundtrip_keeps_order(tmp_path):
    strips = [np.full((8, 40, 3), value, np.uint8) for value in (0, 90, 255)]
    path = tmp_path / "strips.bin"  # any suffix: nothing is appended
    save_strips(path, [encode_jpeg(strip) for strip in strips])
    loaded = load_strips(path, workers=2)
    assert loaded.shape == (3, 8, 40, 3) and loaded.dtype == np.uint8
    np.testing.assert_allclose(loaded.mean(axis=(1, 2, 3)), [0, 90, 255], atol=2)
    save_strips(tmp_path / "empty.npz", [])
    assert len(load_strips(tmp_path / "empty.npz")) == 0


def test_resize_is_bilinear_without_antialiasing(tmp_path):
    # With half-pixel centres, halving the size averages each 2 x 2 block;
    # an antialiased filter would blur further.
    pixels = np.random.default_rng(0).integers(0, 256, (8, 6, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "image.png")
    resized = load_and_preprocess_image(tmp_path / "image.png", 3, 4)
    blocks = pixels.reshape(4, 2, 3, 2, 3).astype(float).mean(axis=(1, 3))
    np.testing.assert_allclose(resized, blocks, atol=0.51)  # uint8 rounding only


def test_csv_to_records_pipeline(tmp_path):
    data = tmp_path / "data"
    (data / "tabular").mkdir(parents=True)
    (data / "images").mkdir()
    Image.new("RGB", (16, 16)).save(data / "images" / "item.jpg")
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
    records = load_strips(paths["train_dataset_path"])
    previews = load_strips(paths["train_sample_path"])
    np.testing.assert_array_equal(records, previews)
    assert records.shape == (1, 128, 960, 3)
    image = records[0] / 255
    assert image[:, :96].mean() > 0.98  # Missing Tops slot stays white.
    assert image[:, 3 * 96 : 4 * 96].mean() < 0.02  # Dress stays black.
    with pytest.raises(ValueError, match="Channel-stacked"):
        build_image_arrays(compact=True, config_path=config_path)

    config["training"].update(epochs=1, batch_size=2)
    config_path.write_text(yaml.safe_dump(config))
    model, history = train_outfit_embeddings(config_path)
    assert len(history) == 1 and np.isfinite(history[0])
    assert Path(paths["model_weights_path"]).is_file()
    assert (Path(paths["epoch_images_path"]) / "image_at_epoch_0001.png").is_file()
    import pickle

    with open(paths["model_data_path"], "rb") as stream:
        exported = pickle.load(stream)
    assert exported["latent_space"].shape == (1, 256)
    assert exported["embedding_type"] == "posterior_mean"
    device = next(model.parameters()).device
    assert model.decode(torch.zeros(1, model.latent_dim, device=device)).shape == (1, 3, 128, 960)


def test_training_step_updates_weights_and_returns_loss():
    model = torch.nn.Conv2d(3, 3, 1)
    batch = torch.ones(2, 3, 2, 2)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = make_optimizer(model, 0.01)
    loss = train_step(model, batch, optimizer)
    assert loss.ndim == 0 and not loss.requires_grad and np.isfinite(loss.item())
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert np.isfinite(compute_loss(model, batch).item())
