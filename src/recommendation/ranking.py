"""Exact Euclidean nearest neighbors and a product-indexed offline result store."""

import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np

from src.data_processing.schema import PRODUCT_CLASSES


def nearest_neighbors(candidates, reference, candidate_block=64, reference_block=256):
    """Return minimum Euclidean distance and reference row, bounded by block sizes.

    Subtraction in float64 avoids cancellation in the squared-norm identity.
    Exact ties select the earliest reference row. No approximate search is used.
    """
    candidates, reference = np.asarray(candidates), np.asarray(reference)
    if (
        candidates.ndim != 2
        or reference.ndim != 2
        or not len(reference)
        or candidates.shape[1] != reference.shape[1]
    ):
        raise ValueError(
            "Embeddings must be 2D with equal dimensions and nonempty reference"
        )
    if candidate_block < 1 or reference_block < 1:
        raise ValueError("Distance block sizes must be positive")
    if not np.isfinite(candidates).all() or not np.isfinite(reference).all():
        raise ValueError("Embeddings contain nonfinite values")
    best = np.full(len(candidates), np.inf)
    indices = np.zeros(len(candidates), dtype=np.int64)
    for start in range(0, len(candidates), candidate_block):
        end = min(start + candidate_block, len(candidates))
        block = candidates[start:end].astype(np.float64)
        for offset in range(0, len(reference), reference_block):
            diff = block[:, None, :] - reference[
                None, offset : offset + reference_block, :
            ].astype(np.float64)
            squared = np.einsum("ijk,ijk->ij", diff, diff)
            local = squared.argmin(axis=1)
            values = squared[np.arange(len(block)), local]
            improve = values < best[start:end]
            best[start:end][improve] = values[improve]
            indices[start:end][improve] = offset + local[improve]
    return np.sqrt(best), indices


def write_index(path, candidates, distances, nearest_ids, metadata):
    """Atomically publish a self-contained index; identical outfits appear once."""
    if len(candidates) != len(distances) or len(candidates) != len(nearest_ids):
        raise ValueError("Scores and candidate rows are misaligned")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.executescript("""
                CREATE TABLE outfits(candidate_id TEXT PRIMARY KEY, signature TEXT UNIQUE,
                    configuration_id INTEGER, distance REAL, nearest_training_outfit TEXT, products TEXT);
                CREATE TABLE membership(product_id TEXT, candidate_id TEXT, distance REAL,
                    PRIMARY KEY(product_id, candidate_id));
                CREATE INDEX product_ranking ON membership(product_id, distance, candidate_id);
                CREATE TABLE metadata(document TEXT);
            """)
            connection.execute(
                "INSERT INTO metadata VALUES (?)", (json.dumps(metadata),)
            )
            for row, distance, nearest in zip(
                candidates.to_dict("records"), distances, nearest_ids
            ):
                products = {slot: row[slot] for slot in PRODUCT_CLASSES if row[slot]}
                inserted = connection.execute(
                    "INSERT OR IGNORE INTO outfits VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        row["candidate_id"],
                        row["signature"],
                        int(row["configuration_id"]),
                        float(distance),
                        str(nearest),
                        json.dumps(products),
                    ),
                )
                if inserted.rowcount:
                    connection.executemany(
                        "INSERT INTO membership VALUES (?, ?, ?)",
                        [
                            (product, row["candidate_id"], float(distance))
                            for product in products.values()
                        ],
                    )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def recommend(index_path, product_id, top_k=3):
    """Return the top distinct sampled outfits containing this exact product ID."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    uri = Path(index_path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        rows = connection.execute(
            """
            SELECT o.candidate_id, o.configuration_id, m.distance,
                   o.nearest_training_outfit, o.products
            FROM membership m JOIN outfits o USING(candidate_id)
            WHERE m.product_id = ? ORDER BY m.distance, m.candidate_id LIMIT ?
        """,
            (str(product_id), top_k),
        ).fetchall()
    return [
        {
            "candidate_id": r[0],
            "configuration_id": r[1],
            "distance": r[2],
            "nearest_training_outfit": r[3],
            "products": json.loads(r[4]),
        }
        for r in rows
    ]
