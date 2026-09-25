"""Encode both populations with one checkpoint and build an exact retrieval index."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from src.data_processing.schema import PRODUCT_CLASSES
from src.recommendation.candidates import (
    read_catalog,
    read_outfits,
    resolve_image_path,
)
from src.recommendation.ranking import nearest_neighbors, write_index
from src.utils.setup_utilities import ensure_parent, file_digest, load_config


def build_index(config_path=None, device=None):
    config = load_config(config_path)
    data, training, rec = config["data"], config["training"], config["recommendation"]
    with open(rec["generation_manifest_path"]) as stream:
        generation = json.load(stream)
    for key, path in {
        "training_outfits": data["outfits_w_products_path"],
        "products": data["products_path"],
        "candidates": rec["candidates_path"],
        "configurations": rec["configurations_path"],
    }.items():
        if file_digest(path) != generation["sha256"][key]:
            raise ValueError(f"{key} changed after generation; regenerate candidates")
    rules = {key: data[key] for key in ("configurations_base", "optional_products")}
    if rules != generation["rules"]:
        raise ValueError("Configuration rules changed; regenerate candidates")
    if (data["image_height"], data["image_width"]) != (128, 96):
        raise ValueError("The trained model requires 128 x 96 product images")
    if rec["encoding_batch_size"] < 1:
        raise ValueError("encoding_batch_size must be positive")
    candidates = pd.read_csv(rec["candidates_path"], dtype=str, keep_default_na=False)
    reference = read_outfits(data["outfits_w_products_path"])
    catalog = read_catalog(data["products_path"])
    data_root = Path(data["products_path"]).parent.parent
    paths = {
        row.cod_modelo_color: resolve_image_path(row.des_filename, data_root)
        for row in catalog.itertuples()
    }
    used = {
        product for slot in PRODUCT_CLASSES for product in candidates[slot] if product
    }
    missing = sorted(
        product
        for product in used
        if product not in paths or not Path(paths[product]).is_file()
    )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} candidate product images missing; first IDs: {missing[:10]}. Restore images before encoding."
        )
    manifest_path = training.get(
        "model_manifest_path", training["model_weights_path"] + ".json"
    )
    if not Path(manifest_path).is_file():
        raise ValueError(
            "Checkpoint provenance is missing; rerun training to create the model manifest"
        )
    with open(manifest_path) as stream:
        checkpoint = json.load(stream)
    for key, path in {
        "weights": training["model_weights_path"],
        "training_records": training["train_dataset_path"],
        "training_outfits": data["outfits_w_products_path"],
    }.items():
        if file_digest(path) != checkpoint["sha256"][key]:
            raise ValueError(
                f"{key} no longer matches the checkpoint; rebuild/retrain before ranking"
            )
    import tensorflow as tf

    from src.models.vae256 import Conv_VAE
    from src.recommendation.encoding import encode_candidates, encode_reference
    from src.utils.device_utilities import configure_tensorflow

    device_report = configure_tensorflow(device or training.get("device", "auto"))
    print(f"TensorFlow device: {json.dumps(device_report)}", flush=True)
    model = Conv_VAE()
    model(tf.zeros((1, 128, 960, 3)))
    model.load_weights(training["model_weights_path"])
    print("Encoding reference outfits with the saved checkpoint", flush=True)
    reference_embeddings = encode_reference(
        model, training["train_dataset_path"], rec["encoding_batch_size"]
    )
    if len(reference_embeddings) != len(reference):
        raise ValueError("Training record count does not match outfit IDs")
    candidate_embeddings = encode_candidates(
        model, candidates, paths, rec["encoding_batch_size"], rec["image_cache_size"]
    )
    print("Computing exact nearest-neighbor distances", flush=True)
    distances, nearest = nearest_neighbors(
        candidate_embeddings,
        reference_embeddings,
        rec["distance_candidate_block"],
        rec["distance_reference_block"],
    )
    nearest_ids = reference.cod_outfit.to_numpy()[nearest]
    np.savez_compressed(
        ensure_parent(rec["embeddings_path"]),
        candidate_ids=candidates.candidate_id.to_numpy(dtype=str),
        candidate_embeddings=candidate_embeddings,
        reference_ids=reference.cod_outfit.to_numpy(dtype=str),
        reference_embeddings=reference_embeddings,
        distances=distances,
        nearest_reference_indices=nearest,
    )
    metadata = {
        "generation": generation,
        "checkpoint": checkpoint,
        "metric": "minimum_euclidean_distance",
        "embedding_type": "posterior_mean",
        "candidate_preprocessing": "128x96_rgb_ten_slots_jpeg_roundtrip_float32_0_1",
        "reference_count": len(reference),
        "draws": len(candidates),
        "distinct_outfits": int(candidates.signature.nunique()),
        "covered_products": len(used),
        "duplicate_policy": "keep_first_identical_outfit_at_serving",
        "image_sha256": {
            product: file_digest(paths[product]) for product in sorted(used)
        },
    }
    write_index(rec["index_path"], candidates, distances, nearest_ids, metadata)
    with open(ensure_parent(rec["index_manifest_path"]), "w") as stream:
        json.dump(metadata, stream, indent=2)
    print(
        f"Indexed {metadata['distinct_outfits']} distinct outfits covering {len(used)} products"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"))
    args = parser.parse_args()
    build_index(args.config, args.device)
