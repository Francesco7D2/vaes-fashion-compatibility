"""Draw the figures and the evaluation table of reports/report.tex.

    python -m scripts.report.make_figures

Figures are drawn at their printed size: 3.3 in for one column, 7 in for the
full page width. Statistics quoted in the report are printed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

from scripts.generation.generate_outfits import load_resources
from src.data_processing.image_processing import load_strips
from src.data_processing.schema import PRODUCT_CLASSES
from src.generation.candidates import catalogues, outfit_catalogues
from src.generation.retrieval import rank_for_product
from src.utils import visualization as viz
from src.utils.setup_utilities import PROJECT_ROOT, load_config
from src.utils.torch_utilities import to_images, to_model_input

OUTPUT = PROJECT_ROOT / "reports" / "figures"
SEED = 7
COLUMN = (3.3, 2.1)
DRESS = "Dresses, jumpsuits and Complete set"

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


def latent_projection(latent_space, outfits, of_outfit):
    """First two principal components, by outfit base and catalogue."""
    projection = viz.pca_2d(latent_space)
    dress = outfits[DRESS].notna().to_numpy()
    fig, axis = plt.subplots(figsize=(3.3, 2.5))
    for name in sorted(set(of_outfit)):
        for is_dress, base in ((False, "tops + bottoms"), (True, "dress")):
            mask = (of_outfit == name) & (dress == is_dress)
            axis.scatter(projection[mask, 0], projection[mask, 1], s=2, alpha=0.6, label=f"{base}, {name}")
    axis.legend(markerscale=4, fontsize=6)
    axis.set(xlabel="PC 1", ylabel="PC 2")
    latest = sorted(set(of_outfit))[-1]
    for label, target in (("dress", dress), (f"catalogue {latest}", of_outfit == latest)):
        print(f"linear separability of {label} in the 2-D projection: {linear_accuracy(projection, target):.1%}")
    return fig


def linear_accuracy(features, target):
    """Training accuracy of a least-squares linear classifier (a separability measure)."""
    design = np.column_stack([np.ones(len(features)), features])
    signs = np.where(target, 1.0, -1.0)
    weights = np.linalg.lstsq(design, signs, rcond=None)[0]
    return np.mean((design @ weights > 0) == target)


def product_pixels(image):
    """(area, background darkness, product darkness) of one product image.

    Pixels that differ from the image border by more than 25 levels are product
    pixels; darkness is 1 - intensity, averaged over the whole image.
    """
    image = image.astype(float)
    edges = (image[:4], image[-4:], image[:, :4], image[:, -4:])
    background = np.concatenate([edge.reshape(-1, 3) for edge in edges]).mean(axis=0)
    mask = np.abs(image - background).max(axis=2) > 25
    darkness = (255 - image.mean(axis=2)) / 255
    return mask.mean(), 1 - background.mean() / 255, (darkness * mask).mean()


def r_squared(features, target):
    design = np.column_stack([np.ones(len(target)), features])
    residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
    return 1 - residual.var() / target.var()


def distance_panels(generated, images, count=4000):
    """(a) distance histogram, (b) by size, (c) by product darkness; prints the decomposition."""
    sample = generated.sample(count, random_state=SEED)
    slots = sample[list(PRODUCT_CLASSES)]
    stats = {p: product_pixels(images(p)) for p in pd.unique(slots.stack())}
    per_strip = np.array(
        [np.sum([stats[p] for p in row.dropna()], axis=0) / len(PRODUCT_CLASSES) for _, row in slots.iterrows()]
    )
    area, tint, darkness = per_strip.T
    sizes = slots.notna().sum(axis=1).to_numpy()
    distance = sample["distance"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.1), width_ratios=(1, 1, 1.2))
    viz.plot_distance_histogram(generated["distance"], axis=axes[0])
    all_sizes = generated[list(PRODUCT_CLASSES)].notna().sum(axis=1)
    groups = sorted(all_sizes.unique())
    axes[1].boxplot([generated.loc[all_sizes == size, "distance"] for size in groups], showfliers=False)
    axes[1].set_xticks(range(1, len(groups) + 1), groups)
    axes[1].set(xlabel="products in candidate", ylabel="distance")
    axes[1].grid(alpha=0.3, axis="y")
    points = axes[2].scatter(darkness, distance, c=sizes, cmap="viridis", s=1, alpha=0.6)
    fig.colorbar(points, ax=axes[2], label="products in candidate")
    axes[2].set(xlabel="product darkness of the strip", ylabel="distance")
    for axis, letter in zip(axes, "abc"):
        axis.set_title(f"({letter})", loc="left")
    fig.tight_layout(w_pad=1.5)

    within = [
        np.corrcoef(darkness[sizes == size], distance[sizes == size])[0, 1]
        for size in np.unique(sizes)
        if (sizes == size).sum() >= 30
    ]
    print(f"distance over {count} candidates: r with product darkness {np.corrcoef(darkness, distance)[0, 1]:.2f} "
          f"(within sizes {min(within):.2f} to {max(within):.2f})")
    for label, features in (
        ("size", sizes),
        ("product area", area),
        ("background darkness", tint),
        ("product darkness", darkness),
        ("all four", np.column_stack([sizes, area, tint, darkness])),
    ):
        print(f"  R^2 distance ~ {label}: {r_squared(features, distance):.2f}")
    return fig


@torch.inference_mode()
def slot_occupancy(model, strips, outfits, batch_size=64):
    """Share of slots whose reconstruction is white exactly when the slot is empty."""
    device = next(model.parameters()).device
    brightness = []
    for start in range(0, len(strips), batch_size):
        x = to_model_input(strips[start : start + batch_size], device)
        rebuilt = to_images(model.decode(model.encode(x)[0], apply_sigmoid=True))
        brightness.append(rebuilt.reshape(len(rebuilt), 128, 10, 96, 3).mean(axis=(1, 3, 4)))
    brightness = np.concatenate(brightness)
    empty = outfits[list(PRODUCT_CLASSES)].isna().to_numpy()
    threshold = max(np.linspace(0.85, 1, 301), key=lambda t: ((brightness > t) == empty).mean())
    correct = (brightness > threshold) == empty
    return correct.mean(), correct.all(axis=1).mean()


def statistics(model, strips, outfits, latent_space, generated, candidate_embeddings, colour):
    """Print the statistics of the generated outfits quoted in the report."""
    means = np.asarray(latent_space)
    active = int((means.var(axis=0) > 0.01).sum())
    print(f"active latent dimensions (variance of the posterior mean > 0.01): {active} of {means.shape[1]}")
    appearances = {
        slot.split(",")[0]: generated[slot].value_counts().median() for slot in PRODUCT_CLASSES
    }
    print("median candidates per product, by class:", appearances)
    slots, outfits_ok = slot_occupancy(model, strips, outfits)
    print(f"reconstructed slots correctly empty or occupied: {slots:.2%}; outfits with every slot right: {outfits_ok:.2%}")
    training = np.asarray(latent_space, dtype=np.float64)
    squared = (training**2).sum(1)[:, None] + (training**2).sum(1)[None] - 2 * training @ training.T
    np.fill_diagonal(squared, np.inf)
    spacing = np.sqrt(np.maximum(squared.min(axis=1), 0))
    median, p90 = np.median(spacing), np.percentile(spacing, 90)
    distance = generated["distance"]
    print(f"training spacing median {median:.1f}, p90 {p90:.1f}; candidates farther than the median "
          f"{(distance > median).mean():.1%}, within p90 {(distance <= p90).mean():.1%}")
    norm = np.linalg.norm(candidate_embeddings, axis=1)
    print(f"candidate distance vs embedding norm: r = {np.corrcoef(norm, distance)[0, 1]:.2f}")
    print(f"distance: mean {distance.mean():.1f}, sd {distance.std():.1f}, skewness {distance.skew():.2f}, "
          f"p5 {distance.quantile(0.05):.1f}, median {distance.median():.1f}, p95 {distance.quantile(0.95):.1f}; "
          f"nearest training outfits used: {generated['nearest_training_row'].nunique()}")
    sizes = generated[list(PRODUCT_CLASSES)].notna().sum(axis=1)
    print("mean distance by size:", generated.groupby(sizes)["distance"].mean().round(1).to_dict(),
          "| dress", round(distance[generated[DRESS].notna()].mean(), 1),
          "| tops + bottoms", round(distance[generated[DRESS].isna()].mean(), 1))

    def coherence(frame):
        groups = frame[list(PRODUCT_CLASSES)].apply(lambda column: column.map(colour))
        return groups.apply(lambda row: row.value_counts(normalize=True).iloc[0], axis=1)

    candidate_coherence = coherence(generated)
    deciles = generated.groupby(sizes)["distance"].transform(lambda d: pd.qcut(d, 10, labels=False))
    print(f"colour coherence: real outfits {coherence(outfits).mean():.3f}, candidates {candidate_coherence.mean():.3f}; "
          "by distance decile (within size):", candidate_coherence.groupby(deciles).mean().round(3).tolist())
    rng = np.random.default_rng(SEED)
    overlaps = []
    for product in rng.choice(pd.unique(generated[list(PRODUCT_CLASSES)].stack()), 300, replace=False):
        top = rank_for_product(generated, product, k=3)[list(PRODUCT_CLASSES)]
        sets = [set(row.dropna()) - {product} for _, row in top.iterrows()]
        overlaps += [len(a & b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    print(f"top-3 candidates of a product share {np.mean(overlaps):.2f} other products on average")


def recommendations(generated, images, products):
    """Best candidate for each input product, then the worst one for the first."""
    rows, titles = [], []
    for letter, product in zip("abcd", products):
        rows.append(rank_for_product(generated, product, k=1))
        titles.append(f"({letter}) top 1\nd = {rows[-1]['distance'].iloc[0]:.1f}")
    rows.append(rank_for_product(generated, products[0], k=1, worst=True))
    titles.append(f"(e) worst for (a)\nd = {rows[-1]['distance'].iloc[0]:.1f}")
    return viz.show_outfits(pd.concat(rows), images, titles, highlight=products, cell=0.75)


def evaluation_table(results_path, output):
    """LaTeX table of the fill-in-the-blank results: model scores per run, then baselines."""
    results = pd.read_csv(results_path)

    def cell(row):
        return f"{100 * row.accuracy:.1f} ({100 * row.ci_low:.1f}--{100 * row.ci_high:.1f})"

    def row(frame, score):
        pick = frame[frame["score"] == score].set_index("subset")
        return f"\\quad {score} & {cell(pick.loc['all'])} & {cell(pick.loc['novel pairs'])} \\\\"

    labels = {
        "reconstruction": "Reconstruction loss, ReLU head",
        "vae": "Complete VAE (final model)",
        "foreground": "Reconstruction loss, ReLU head, foreground weight 5",
    }
    model_scores = ["distance", "distance (same configuration)", "cosine distance", "reconstruction error"]
    lines = []
    for run, frame in results.groupby("run", sort=False):
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{labels.get(run, run)}}}}} \\\\")
        lines += [row(frame, score) for score in model_scores if score in set(frame["score"])]
    first = results[results["run"] == results["run"].iloc[0]]
    lines.append("\\multicolumn{3}{l}{\\emph{Baselines}} \\\\")
    lines += [row(first, score) for score in ["co-occurrence", "colour coherence", "lightest option", "random"]]
    counts = first.groupby("subset")["questions"].first()
    header = ["\\begin{tabular}{lll}", "\\toprule", "Score & All questions & Novel pairs \\\\", "\\midrule"]
    footer = ["\\bottomrule", "\\end{tabular}"]
    Path(output).write_text("\n".join(header + lines + footer) + "\n")
    print(f"evaluation table written; questions: {counts.to_dict()}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    training, generation = config["training"], config["generation"]
    catalog, outfits, exported, model, images = load_resources(config)
    generated = pd.read_csv(
        generation["generated_outfits_path"], dtype={slot: str for slot in PRODUCT_CLASSES}
    )
    of_outfit = outfit_catalogues(outfits, catalogues(catalog, outfits)).to_numpy()

    save(loss_curve(exported["loss"]), "loss.png")
    save(latent_projection(exported["latent_space"], outfits, of_outfit), "latent_pca.png")
    save(distance_panels(generated, images), "distances.png")

    samples = load_strips(training["train_sample_path"])
    indices = training["train_sample_indices"][: len(samples)]
    save(reconstructions(samples, model, indices), "reconstructions.png")
    metadata = pd.read_csv(config["data"]["products_path"], dtype={"cod_modelo_color": str})
    colour = metadata.drop_duplicates("cod_modelo_color").set_index("cod_modelo_color")["des_agrup_color_eng"]
    statistics(
        model,
        load_strips(training["train_dataset_path"]),
        outfits,
        exported["latent_space"],
        generated,
        np.load(generation["candidate_embeddings_path"]),
        colour,
    )

    # Random input products drawn from the candidates, and a random white skirt.
    rng = np.random.default_rng(SEED)
    candidates = pd.unique(generated[list(PRODUCT_CLASSES)].stack())
    products = list(rng.choice(candidates, size=4, replace=False))
    save(recommendations(generated, images, products), "recommendations.png")
    white = metadata.loc[metadata["des_color_specification_esp"] == "BLANCO", "cod_modelo_color"]
    skirt = rng.choice(sorted(set(white) & set(generated["Bottoms"].dropna())))
    top3 = rank_for_product(generated, skirt, k=3)
    titles = [f"#{rank}\nd = {d:.1f}" for rank, d in enumerate(top3["distance"], start=1)]
    save(viz.show_outfits(top3, images, titles, highlight=skirt, cell=0.75), "top3.png")
    print(f"inputs {products}, white skirt {skirt}")

    if Path(config["evaluation"]["results_path"]).exists():
        evaluation_table(config["evaluation"]["results_path"], OUTPUT / "evaluation.tex")
    print(f"Figures written to {OUTPUT}")


if __name__ == "__main__":
    main()
