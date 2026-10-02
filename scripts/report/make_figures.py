"""Draw the figures of reports/report.tex from the trained model and generated outfits.

    python -m scripts.report.make_figures

Figures are drawn at their printed size: 3.3 in for one column, 7 in for the
full page width.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.generation.generate_outfits import load_catalog
from src.data_processing.image_processing import build_outfit_strip, load_strips
from src.data_processing.schema import PRODUCT_CLASSES
from src.generation.embeddings import ProductImages, load_training_embeddings
from src.generation.retrieval import rank_for_product
from src.models.vae256 import load_trained_model
from src.utils import visualization as viz
from src.utils.setup_utilities import PROJECT_ROOT, load_config

OUTPUT = PROJECT_ROOT / "reports" / "figures"
SEED = 7
COLUMN = (3.3, 2.1)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})


def save(fig, name):
    fig.savefig(OUTPUT / name, dpi=250, bbox_inches="tight")
    plt.close(fig)


def loss_curve(history):
    fig, axis = plt.subplots(figsize=COLUMN)
    viz.plot_loss_curve(history, axis)
    axis.set(yscale="log", title="")
    axis.grid(alpha=0.3, which="both")
    return fig


def reconstructions(samples, model, indices):
    """Originals beside their posterior-mean reconstructions."""
    rebuilt = viz.reconstruct(model, samples)
    fig, axes = plt.subplots(len(samples), 2, figsize=(7, 0.47 * len(samples) + 0.2), squeeze=False)
    for row, pair in enumerate(zip(samples, np.clip(rebuilt, 0, 1))):
        for column, image in enumerate(pair):
            axes[row, column].imshow(image)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(f"row {indices[row]}", rotation=0, ha="right", va="center")
    axes[0, 0].set_title("original")
    axes[0, 1].set_title("reconstruction")
    fig.subplots_adjust(wspace=0.04, hspace=0.08)
    return fig


def latent_projection(latent_space, outfits):
    projection = viz.pca_2d(latent_space)
    dress = outfits["Dresses, jumpsuits and Complete set"].notna().to_numpy()
    fig, axis = plt.subplots(figsize=(3.3, 2.5))
    for mask, label in ((~dress, "tops + bottoms"), (dress, "dress")):
        axis.scatter(projection[mask, 0], projection[mask, 1], s=2, alpha=0.6, label=label)
    axis.legend(markerscale=4)
    axis.set(xlabel="PC 1", ylabel="PC 2")
    return fig


def distance_panels(generated, images):
    """(a) distance histogram, (b) distance by size, (c) distance by coverage."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.1), width_ratios=(1, 1, 1.2))
    viz.plot_distance_histogram(generated["distance"], axis=axes[0])
    distance_by_size(axes[1], generated)
    distance_by_coverage(axes[2], generated, images)
    for axis, letter in zip(axes, "abc"):
        axis.set_title(f"({letter})", loc="left")
    fig.tight_layout(w_pad=1.5)
    return fig


def distance_by_size(axis, generated):
    sizes = generated[list(PRODUCT_CLASSES)].notna().sum(axis=1)
    groups = sorted(sizes.unique())
    axis.boxplot([generated.loc[sizes == size, "distance"] for size in groups], showfliers=False)
    axis.set_xticks(range(1, len(groups) + 1), groups)
    axis.set(xlabel="products in candidate", ylabel="distance")
    axis.grid(alpha=0.3, axis="y")


def distance_by_coverage(axis, generated, images, count=4000):
    """Distance against the non-white share of each candidate's strip."""
    sample = generated.sample(count, random_state=SEED)
    coverage = np.array(
        [
            1 - build_outfit_strip(row, images, images.width, images.height).mean() / 255
            for row in sample.to_dict("records")
        ]
    )
    sizes = sample[list(PRODUCT_CLASSES)].notna().sum(axis=1).to_numpy()
    distance = sample["distance"].to_numpy()
    points = axis.scatter(coverage, distance, c=sizes, cmap="viridis", s=1, alpha=0.6)
    axis.figure.colorbar(points, ax=axis, label="products in candidate")
    axis.set(xlabel="non-white share of the strip", ylabel="distance")
    within = [
        np.corrcoef(coverage[sizes == size], distance[sizes == size])[0, 1]
        for size in np.unique(sizes)
        if (sizes == size).sum() >= 30
    ]
    print(
        f"distance vs coverage over {count} candidates: r = {np.corrcoef(coverage, distance)[0, 1]:.2f}"
        f" (within sizes {min(within):.2f} to {max(within):.2f}); R^2 size only"
        f" {r_squared(sizes, distance):.2f}, size + coverage"
        f" {r_squared(np.column_stack([sizes, coverage]), distance):.2f}"
    )


def r_squared(features, target):
    design = np.column_stack([np.ones(len(target)), features])
    residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
    return 1 - residual.var() / target.var()


def recommendations(generated, images, products):
    """Best candidate for each input product, then the worst one for the first."""
    rows, titles = [], []
    for letter, product in zip("abcd", products):
        rows.append(rank_for_product(generated, product, k=1))
        titles.append(f"({letter}) top 1\nd = {rows[-1]['distance'].iloc[0]:.1f}")
    rows.append(rank_for_product(generated, products[0], k=1, worst=True))
    titles.append(f"(e) worst for (a)\nd = {rows[-1]['distance'].iloc[0]:.1f}")
    return viz.show_outfits(pd.concat(rows), images, titles, highlight=products, cell=0.75)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    training, generation = config["training"], config["generation"]
    catalog, outfits = load_catalog(config)
    exported = load_training_embeddings(training["model_data_path"])
    generated = pd.read_csv(
        generation["generated_outfits_path"], dtype={slot: str for slot in PRODUCT_CLASSES}
    )
    images = ProductImages(
        catalog.set_index("cod_modelo_color")["des_filename"],
        config["data"]["image_width"],
        config["data"]["image_height"],
    )

    save(loss_curve(exported["reconstruction_loss"]), "loss.png")
    save(latent_projection(exported["latent_space"], outfits), "latent_pca.png")
    save(distance_panels(generated, images), "distances.png")

    samples = load_strips(training["train_sample_path"])
    model = load_trained_model(training["model_weights_path"])
    indices = training["train_sample_indices"][: len(samples)]
    save(reconstructions(samples, model, indices), "reconstructions.png")

    # Random input products drawn from the candidates, and a random white skirt.
    rng = np.random.default_rng(SEED)
    candidates = pd.unique(generated[list(PRODUCT_CLASSES)].stack())
    products = list(rng.choice(candidates, size=4, replace=False))
    save(recommendations(generated, images, products), "recommendations.png")
    metadata = pd.read_csv(config["data"]["products_path"], dtype={"cod_modelo_color": str})
    white = metadata.loc[metadata["des_color_specification_esp"] == "BLANCO", "cod_modelo_color"]
    skirt = rng.choice(sorted(set(white) & set(generated["Bottoms"].dropna())))
    top3 = rank_for_product(generated, skirt, k=3)
    titles = [f"#{rank}\nd = {d:.1f}" for rank, d in enumerate(top3["distance"], start=1)]
    save(viz.show_outfits(top3, images, titles, highlight=skirt, cell=0.75), "top3.png")
    print(f"Figures written to {OUTPUT}; inputs {products}, white skirt {skirt}")


if __name__ == "__main__":
    main()
