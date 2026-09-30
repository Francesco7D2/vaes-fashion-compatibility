"""Plotting helpers for the notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data_processing.schema import PRODUCT_CLASSES
from src.utils.torch_utilities import to_images, to_model_input


def split_strip(strip, num_slots=len(PRODUCT_CLASSES)):
    """Cut an H x (num_slots * W) x 3 strip into its per-slot images."""
    return np.split(np.asarray(strip), num_slots, axis=1)


def show_strips(strips, titles=None, label_slots=True):
    """One row per 128 x 960 strip, slot names on the first row."""
    fig, axes = plt.subplots(len(strips), 1, figsize=(14, 1.6 * len(strips) + 0.6), squeeze=False)
    for index, (axis, strip) in enumerate(zip(axes[:, 0], strips)):
        strip = np.asarray(strip)
        axis.imshow(strip if strip.dtype == np.uint8 else np.clip(strip, 0, 1))
        axis.set_xticks([])
        axis.set_yticks([])
        if titles is not None:
            axis.set_ylabel(titles[index], rotation=0, ha="right", va="center", fontsize=9)
        if label_slots and index == 0:
            width = np.asarray(strip).shape[1] / len(PRODUCT_CLASSES)
            axis.set_xticks(width * (np.arange(len(PRODUCT_CLASSES)) + 0.5))
            axis.set_xticklabels([c.split(",")[0] for c in PRODUCT_CLASSES], fontsize=8)
            axis.xaxis.tick_top()
    fig.tight_layout()
    return fig


def show_outfits(outfits, images, titles=None, max_cols=9):
    """Draw occupied slots only, one outfit per row (`outfits`: slot-wide rows)."""
    rows = outfits.to_dict("records")
    fig, axes = plt.subplots(
        len(rows), max_cols, figsize=(1.3 * max_cols, 1.7 * len(rows)), squeeze=False
    )
    for axis in axes.ravel():
        axis.axis("off")
    for r, row in enumerate(rows):
        products = [(s, row[s]) for s in PRODUCT_CLASSES if not pd.isna(row[s])]
        for c, (slot, product) in enumerate(products[:max_cols]):
            axes[r, c].imshow(images(product))
            axes[r, c].set_title(slot.split(",")[0], fontsize=7)
        if titles is not None:
            axes[r, 0].text(-0.1, 0.5, titles[r], transform=axes[r, 0].transAxes,
                            ha="right", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_loss_curve(history):
    fig, axis = plt.subplots(figsize=(7, 3.5))
    axis.plot(range(1, len(history) + 1), history)
    axis.set(xlabel="epoch", ylabel="mean pixel cross entropy", title="Reconstruction loss")
    axis.grid(alpha=0.3)
    return fig


def plot_distance_histogram(distances, marks=(5, 50, 95)):
    fig, axis = plt.subplots(figsize=(7, 3.5))
    axis.hist(distances, bins=60, color="tab:blue", alpha=0.8)
    for percentile in marks:
        value = np.percentile(distances, percentile)
        axis.axvline(value, color="tab:red", ls="--", lw=1)
        axis.text(value, axis.get_ylim()[1] * 0.95, f"p{percentile}", color="tab:red", fontsize=8)
    axis.set(xlabel="distance to nearest training outfit", ylabel="candidates")
    return fig


@torch.inference_mode()
def reconstruct(model, strips):
    """Deterministic reconstructions (posterior mean) of a batch of H x W x 3 strips."""
    mean, _ = model.encode(to_model_input(strips, next(model.parameters()).device))
    return to_images(model.decode(mean, apply_sigmoid=True))


def pca_2d(embeddings):
    """Project embeddings on their first two principal axes (plain SVD)."""
    centered = embeddings - embeddings.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:2].T
