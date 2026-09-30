"""Generate candidate outfits and score them by distance to the training latent space."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.product_processing import build_product_catalog
from src.data_processing.schema import PRODUCT_CLASSES
from src.generation.candidates import (
    configuration_probabilities,
    product_pools,
    sample_candidates,
)
from src.generation.embeddings import (
    ProductImages,
    encode_outfits,
    load_training_embeddings,
)
from src.generation.retrieval import nearest_training_outfit
from src.models.vae256 import load_trained_model
from src.utils.setup_utilities import ensure_parent, load_config


def load_catalog(config):
    """Product catalog (class + image path) and the training outfit table."""
    data = config["data"]
    products = pd.read_csv(data["products_path"], dtype={"cod_modelo_color": str})
    catalog = build_product_catalog(
        products, Path(data["outfits_path"]).parent.parent, require_images=True
    )
    outfits = pd.read_csv(
        data["outfits_w_products_path"], dtype={slot: str for slot in PRODUCT_CLASSES}
    )
    return catalog, outfits


def generate_outfits(config_path=None, overrides=None):
    """Sample, encode and score candidates; write and return the ranked table."""
    config = load_config(config_path, overrides)
    generation, training = config["generation"], config["training"]
    catalog, outfits = load_catalog(config)
    exported = load_training_embeddings(training["model_data_path"])
    if len(exported["latent_space"]) != len(outfits):
        raise ValueError("Training embeddings do not match the outfit table; retrain")
    candidates = sample_candidates(
        product_pools(catalog, outfits, generation["candidate_pool"]),
        configuration_probabilities(outfits),
        generation["num_candidates"],
        generation["seed"],
    )
    model = load_trained_model(
        training["model_weights_path"], generation.get("device", "auto")
    )
    images = ProductImages(
        catalog.set_index("cod_modelo_color")["des_filename"],
        config["data"]["image_width"],
        config["data"]["image_height"],
    )
    embeddings = encode_outfits(
        model,
        candidates,
        images,
        generation["encode_batch_size"],
        generation["match_training_jpeg"],
    )
    np.save(ensure_parent(generation["candidate_embeddings_path"]), embeddings)
    distances, nearest = nearest_training_outfit(embeddings, exported["latent_space"])
    candidates["distance"] = distances
    candidates["distance_percentile"] = candidates["distance"].rank(pct=True)
    candidates["nearest_training_row"] = nearest
    candidates.to_csv(ensure_parent(generation["generated_outfits_path"]), index=False)
    return candidates


if __name__ == "__main__":
    generate_outfits()
