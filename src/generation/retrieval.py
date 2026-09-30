"""Rank candidate outfits by latent distance to the training outfits."""

import numpy as np

from src.data_processing.schema import PRODUCT_CLASSES


def nearest_training_outfit(candidates, training, chunk_size=2048):
    """Euclidean distance from each candidate to its closest training embedding.

    Returns `(distances, indices)`; `indices` are rows of `training`, which match
    the rows of the outfit table used for training.
    """
    training = np.asarray(training, dtype=np.float64)
    if training.ndim != 2 or not len(training):
        raise ValueError("training embeddings must be a nonempty 2-D array")
    candidates = np.asarray(candidates)
    if candidates.ndim != 2 or candidates.shape[1] != training.shape[1]:
        raise ValueError("candidate and training embeddings must share dimension")
    training_norm = (training**2).sum(axis=1)
    distances = np.empty(len(candidates))
    indices = np.empty(len(candidates), dtype=np.int64)
    for start in range(0, len(candidates), chunk_size):
        block = candidates[start : start + chunk_size].astype(np.float64)
        squared = (
            (block**2).sum(axis=1)[:, None] + training_norm[None] - 2 * block @ training.T
        )
        nearest = squared.argmin(axis=1)
        indices[start : start + chunk_size] = nearest
        distances[start : start + chunk_size] = np.sqrt(
            np.maximum(squared[np.arange(len(block)), nearest], 0)
        )
    return distances, indices


def rank_for_product(generated, product_id, k=3, worst=False):
    """The k candidates containing `product_id` with smallest (or largest) distance."""
    contains = (generated[list(PRODUCT_CLASSES)] == str(product_id)).any(axis=1)
    matches = generated[contains]
    return matches.nlargest(k, "distance") if worst else matches.nsmallest(k, "distance")
