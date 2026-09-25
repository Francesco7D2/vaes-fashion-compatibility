"""Reusable checks for raw data and artifacts produced by every pipeline stage."""

import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.schema import PRODUCT_CLASSES
from src.utils.setup_utilities import file_digest, load_config


def _result(stage, name, ok, details, path=None):
    return {
        "stage": stage,
        "check": name,
        "status": "PASS" if ok else "FAIL",
        "details": str(details),
        "path": str(path) if path else "",
    }


def _safe_check(results, stage, name, function, path=None):
    try:
        ok, details = function()
    except Exception as error:  # noqa: BLE001 - a report must include every failed check.
        ok, details = False, f"{type(error).__name__}: {error}"
    results.append(_result(stage, name, ok, details, path))


def _csv_rows(path, required_columns):
    frame = pd.read_csv(path, dtype=str)
    missing = sorted(set(required_columns) - set(frame.columns))
    return (
        not frame.empty and not missing,
        f"{len(frame):,} rows; missing columns: {missing}",
    )


def _tfrecord_count(path):
    import tensorflow as tf

    count = sum(1 for _ in tf.data.TFRecordDataset(path))
    return count


def validate_pipeline(config_path=None, include_record_counts=True):
    """Return one row per check without stopping at the first failure."""
    config = load_config(config_path)
    data, training, recommendation = (
        config["data"],
        config["training"],
        config["recommendation"],
    )
    results = []

    for name, path, columns in (
        ("outfit source CSV", data["outfits_path"], ["cod_outfit", "cod_modelo_color"]),
        (
            "product source CSV",
            data["products_path"],
            [
                "cod_modelo_color",
                "des_product_category",
                "des_product_family",
                "des_product_type",
                "des_filename",
            ],
        ),
    ):
        _safe_check(
            results,
            "inputs",
            name,
            lambda path=path, columns=columns: _csv_rows(path, columns),
            path,
        )

    def image_coverage():
        products = pd.read_csv(data["products_path"], dtype={"cod_modelo_color": str})
        root = Path(data["products_path"]).parent.parent

        def resolve(value):
            path = Path(value)
            if path.parts and path.parts[0] == "datathon":
                return root.joinpath(*path.parts[1:])
            if not path.is_absolute():
                return root.parent / path
            return path

        referenced = products.des_filename.map(resolve)
        missing = [str(path) for path in referenced if not path.is_file()]
        duplicate_ids = int(products.cod_modelo_color.duplicated().sum())
        ok = not missing and not duplicate_ids
        return ok, (
            f"{len(products):,} products; {len(missing):,} missing images; "
            f"{duplicate_ids:,} duplicate product IDs"
        )

    _safe_check(results, "inputs", "catalog image coverage", image_coverage)

    for name, path, columns in (
        (
            "selected products",
            data["valid_outfits_path"],
            ["cod_outfit", "cod_modelo_color", "des_product_class", "des_filename"],
        ),
        (
            "wide outfit table",
            data["outfits_w_products_path"],
            ["cod_outfit", *PRODUCT_CLASSES],
        ),
    ):
        _safe_check(
            results,
            "preprocessing",
            name,
            lambda path=path, columns=columns: _csv_rows(path, columns),
            path,
        )

    def record_check(path, expected=None):
        if not Path(path).is_file():
            return False, "file is missing"
        if not include_record_counts:
            return Path(path).stat().st_size > 0, f"{Path(path).stat().st_size:,} bytes"
        count = _tfrecord_count(path)
        ok = count > 0 and (expected is None or count == expected)
        suffix = "" if expected is None else f"; expected {expected:,}"
        return ok, f"{count:,} records{suffix}"

    expected_outfits = None
    if Path(data["outfits_w_products_path"]).is_file():
        expected_outfits = len(
            pd.read_csv(data["outfits_w_products_path"], usecols=["cod_outfit"])
        )
    _safe_check(
        results,
        "preprocessing",
        "training TFRecord",
        lambda: record_check(training["train_dataset_path"], expected_outfits),
        training["train_dataset_path"],
    )
    _safe_check(
        results,
        "preprocessing",
        "preview TFRecord",
        lambda: record_check(
            training["train_sample_path"], len(training["train_sample_indices"])
        ),
        training["train_sample_path"],
    )

    def training_outputs():
        required = [
            training["model_weights_path"],
            training["model_data_path"],
            training["metrics_path"],
            training["model_manifest_path"],
        ]
        missing = [path for path in required if not Path(path).is_file()]
        return (
            not missing,
            f"{len(required) - len(missing)}/{len(required)} files present; missing: {missing}",
        )

    _safe_check(results, "training", "model artifacts", training_outputs)

    def training_metrics():
        metrics = pd.read_csv(training["metrics_path"])
        required = {"epoch", "reconstruction_loss", "epoch_seconds"}
        numeric = sorted(required - {"epoch"})
        ok = (
            bool(len(metrics))
            and required <= set(metrics)
            and np.isfinite(metrics[numeric]).all().all()
        )
        return ok, f"{len(metrics):,} epochs; columns: {list(metrics.columns)}"

    _safe_check(
        results,
        "training",
        "training metrics",
        training_metrics,
        training["metrics_path"],
    )

    def model_data():
        with open(training["model_data_path"], "rb") as stream:
            payload = pickle.load(stream)
        embeddings = np.asarray(payload["latent_space"])
        ok = (
            embeddings.ndim == 2
            and embeddings.shape[1] == 256
            and np.isfinite(embeddings).all()
        )
        return (
            ok,
            f"embeddings shape {embeddings.shape}; type {payload.get('embedding_type')}",
        )

    _safe_check(
        results,
        "training",
        "exported embeddings",
        model_data,
        training["model_data_path"],
    )

    def checkpoint_lineage():
        with open(training["model_manifest_path"]) as stream:
            manifest = json.load(stream)
        paths = {
            "weights": training["model_weights_path"],
            "training_records": training["train_dataset_path"],
            "training_outfits": data["outfits_w_products_path"],
        }
        mismatches = [
            key
            for key, path in paths.items()
            if file_digest(path) != manifest["sha256"].get(key)
        ]
        return (
            not mismatches,
            f"hash mismatches: {mismatches}; epoch: {manifest.get('epoch')}",
        )

    _safe_check(
        results,
        "training",
        "checkpoint lineage",
        checkpoint_lineage,
        training["model_manifest_path"],
    )

    for name, path, columns in (
        (
            "candidate table",
            recommendation["candidates_path"],
            ["candidate_id", "configuration_id", *PRODUCT_CLASSES, "signature"],
        ),
        (
            "configuration frequencies",
            recommendation["configurations_path"],
            ["configuration_id", "training_count", "probability", "sampled_count"],
        ),
    ):
        _safe_check(
            results,
            "recommendation",
            name,
            lambda path=path, columns=columns: _csv_rows(path, columns),
            path,
        )

    def generation_lineage():
        with open(recommendation["generation_manifest_path"]) as stream:
            manifest = json.load(stream)
        paths = {
            "training_outfits": data["outfits_w_products_path"],
            "products": data["products_path"],
            "candidates": recommendation["candidates_path"],
            "configurations": recommendation["configurations_path"],
        }
        mismatches = [
            key
            for key, path in paths.items()
            if file_digest(path) != manifest["sha256"].get(key)
        ]
        return (
            not mismatches,
            f"hash mismatches: {mismatches}; draws: {manifest.get('draws'):,}",
        )

    _safe_check(
        results,
        "recommendation",
        "candidate lineage",
        generation_lineage,
        recommendation["generation_manifest_path"],
    )

    def recommendation_outputs():
        required = [
            recommendation["embeddings_path"],
            recommendation["index_path"],
            recommendation["index_manifest_path"],
        ]
        missing = [path for path in required if not Path(path).is_file()]
        if missing:
            return False, f"missing: {missing}"
        with sqlite3.connect(
            f"file:{Path(recommendation['index_path']).resolve()}?mode=ro", uri=True
        ) as connection:
            outfits = connection.execute("SELECT COUNT(*) FROM outfits").fetchone()[0]
            memberships = connection.execute(
                "SELECT COUNT(*) FROM membership"
            ).fetchone()[0]
        return (
            outfits > 0 and memberships > 0,
            f"{outfits:,} distinct outfits; {memberships:,} product memberships",
        )

    _safe_check(
        results,
        "recommendation",
        "recommendation index",
        recommendation_outputs,
        recommendation["index_path"],
    )
    return results


def validation_frame(config_path=None, include_record_counts=True):
    return pd.DataFrame(validate_pipeline(config_path, include_record_counts))
