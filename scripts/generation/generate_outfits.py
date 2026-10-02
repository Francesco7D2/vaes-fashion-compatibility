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
    sample_candidates_by_catalogue,
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


def load_resources(config):
    """Catalog, outfit table, training export, trained model and image loader."""
    catalog, outfits = load_catalog(config)
    exported = load_training_embeddings(config["training"]["model_data_path"])
    if len(exported["latent_space"]) != len(outfits):
        raise ValueError("Training embeddings do not match the outfit table; retrain")
    model = load_trained_model(
        config["training"]["model_weights_path"], config["generation"].get("device", "auto")
    )
    images = ProductImages(
        catalog.set_index("cod_modelo_color")["des_filename"],
        config["data"]["image_width"],
        config["data"]["image_height"],
    )
    return catalog, outfits, exported, model, images


def draw_candidates(catalog, outfits, generation):
    """Random candidates; with `same_catalogue`, each one stays within a catalogue."""
    if generation.get("same_catalogue", True):
        return sample_candidates_by_catalogue(
            catalog,
            outfits,
            generation["candidate_pool"],
            generation["num_candidates"],
            generation["seed"],
        )
    return sample_candidates(
        product_pools(catalog, outfits, generation["candidate_pool"]),
        configuration_probabilities(outfits),
        generation["num_candidates"],
        generation["seed"],
    )


def score_candidates(candidates, model, images, training_embeddings, generation, progress=True):
    """Add distance, percentile and nearest training row; return the embeddings."""
    embeddings = encode_outfits(
        model,
        candidates,
        images,
        generation["encode_batch_size"],
        generation["match_training_jpeg"],
        progress,
    )
    distances, nearest = nearest_training_outfit(embeddings, training_embeddings)
    candidates["distance"] = distances
    candidates["distance_percentile"] = candidates["distance"].rank(pct=True)
    candidates["nearest_training_row"] = nearest
    return embeddings


def generate_outfits(config_path=None, overrides=None):
    """Sample, encode and score candidates; write and return the ranked table."""
    config = load_config(config_path, overrides)
    generation = config["generation"]
    catalog, outfits, exported, model, images = load_resources(config)
    candidates = draw_candidates(catalog, outfits, generation)
    embeddings = score_candidates(
        candidates, model, images, exported["latent_space"], generation
    )
    np.save(ensure_parent(generation["candidate_embeddings_path"]), embeddings)
    candidates.to_csv(ensure_parent(generation["generated_outfits_path"]), index=False)
    return candidates


if __name__ == "__main__":
    generate_outfits()
