r"""Draw the figures and the evaluation table of reports/report.tex, and compute the numbers it quotes.

    python -m scripts.report.make_figures

Figures are drawn at their printed size: 3.3 in for one column, 7 in for the
full page width. Numbers quoted in the text are written to
reports/figures/values.tex as \setval{key}{value} lines, which the report
prints with \val{key}.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

from scripts.generation.generate_outfits import load_resources
from src.data_processing.image_processing import load_strips
from src.data_processing.schema import PRODUCT_CLASSES
from src.evaluation.fitb import split_outfits
from src.generation.candidates import catalogues, outfit_catalogues
from src.generation.retrieval import rank_for_product
from src.utils import visualization as viz
from src.utils.setup_utilities import PROJECT_ROOT, load_config
from src.utils.torch_utilities import to_images, to_model_input

OUTPUT = PROJECT_ROOT / "reports" / "figures"
SEED = 7
COLUMN = (3.3, 2.1)
DRESS = "Dresses, jumpsuits and Complete set"
MODEL_SCORES = ("distance", "distance (same configuration)", "cosine distance", "reconstruction error")
BASELINES = ("co-occurrence", "colour coherence", "lightest option", "random")

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7})


def save(fig, name):
    fig.savefig(OUTPUT / name, dpi=250, bbox_inches="tight")
    plt.close(fig)


def percent(share, digits=1):
    return f"{100 * share:.{digits}f}"


def slug(text):
    """Key fragment: 'distance (same configuration)' -> 'distance-same-configuration'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def write_values(values, path):
    r"""One \setval{key}{value} line per number."""
    Path(path).write_text("".join(f"\\setval{{{key}}}{{{value}}}\n" for key, value in values.items()))


def loss_curve(history):
    fig, axis = plt.subplots(figsize=COLUMN)
    viz.plot_loss_curve(history, axis)
    axis.set(yscale="log", title="")
    axis.grid(alpha=0.3, which="both")
    return fig


def loss_values(history):
    loss = np.asarray(history)
    best = int(loss.argmin())
    return {
        "epochs": f"{len(loss)}",
        "loss-first": f"{loss[0]:.3f}",
        "loss-epoch-10": f"{loss[9]:.3f}",
        "loss-best": f"{loss[best]:.4f}",
        "loss-best-epoch": f"{best + 1}",
        "loss-gain-after-epoch-100": percent(1 - loss[best] / loss[99]),
    }


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
    """First two principal components, by outfit base and catalogue, and how linearly separable both are."""
    latent_space = np.asarray(latent_space)
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
    values = {}
    for key, target in (("base", dress), ("catalogue", of_outfit == latest)):
        values[f"separability-{key}-2d"] = percent(linear_accuracy(projection, target))
        values[f"separability-{key}"] = percent(linear_accuracy(latent_space, target))
    return fig, values


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
    """(a) distance histogram, (b) by size, (c) by product darkness, and how much darkness explains."""
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
    values = {
        "darkness-sample": f"{count:,}",
        "darkness-r": f"{np.corrcoef(darkness, distance)[0, 1]:.2f}",
        "darkness-r-within-min": f"{min(within):.2f}",
        "darkness-r-within-max": f"{max(within):.2f}",
    }
    for key, features in (
        ("size", sizes),
        ("area", area),
        ("background", tint),
        ("darkness", darkness),
        ("all", np.column_stack([sizes, area, tint, darkness])),
    ):
        values[f"r2-{key}"] = percent(r_squared(features, distance), 0)
    return fig, values


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
    return ((brightness > threshold) == empty).mean()


def statistics(model, strips, outfits, latent_space, generated, candidate_embeddings, colour):
    """Numbers on the latent space, the reconstructions and the generated outfits."""
    means = np.asarray(latent_space)
    distance = generated["distance"]
    values = {
        "latent-dims": f"{means.shape[1]}",
        "latent-active": f"{int((means.var(axis=0) > 0.01).sum())}",
        "candidates": f"{len(generated):,}",
        "slots": f"{outfits[list(PRODUCT_CLASSES)].size:,}",
        "slot-accuracy": percent(slot_occupancy(model, strips, outfits), 2),
    }
    for slot in PRODUCT_CLASSES:
        values[f"candidates-per-{slug(slot.split(',')[0])}"] = f"{generated[slot].value_counts().median():.0f}"

    training = means.astype(np.float64)
    squared = (training**2).sum(1)[:, None] + (training**2).sum(1)[None] - 2 * training @ training.T
    np.fill_diagonal(squared, np.inf)
    spacing = np.sqrt(np.maximum(squared.min(axis=1), 0))
    median, p90 = np.median(spacing), np.percentile(spacing, 90)
    norm = np.linalg.norm(candidate_embeddings, axis=1)
    values.update({
        "spacing-median": f"{median:.1f}",
        "spacing-p90": f"{p90:.1f}",
        "beyond-spacing-median": percent((distance > median).mean()),
        "within-spacing-p90": percent((distance <= p90).mean()),
        "norm-r": f"{np.corrcoef(norm, distance)[0, 1]:.2f}",
        "distance-mean": f"{distance.mean():.1f}",
        "distance-sd": f"{distance.std():.1f}",
        "distance-skewness": f"{distance.skew():.2f}",
        "distance-p5": f"{distance.quantile(0.05):.1f}",
        "distance-median": f"{distance.median():.1f}",
        "distance-p95": f"{distance.quantile(0.95):.1f}",
        "nearest-training-outfits": f"{generated['nearest_training_row'].nunique():,}",
        "distance-dress": f"{distance[generated[DRESS].notna()].mean():.1f}",
        "distance-tops-bottoms": f"{distance[generated[DRESS].isna()].mean():.1f}",
    })
    sizes = generated[list(PRODUCT_CLASSES)].notna().sum(axis=1)
    for size, mean in generated.groupby(sizes)["distance"].mean().items():
        values[f"distance-size-{size}"] = f"{mean:.1f}"

    def coherence(frame):
        groups = frame[list(PRODUCT_CLASSES)].apply(lambda column: column.map(colour))
        return groups.apply(lambda row: row.value_counts(normalize=True).iloc[0], axis=1)

    candidate_coherence = coherence(generated)
    deciles = generated.groupby(sizes)["distance"].transform(lambda d: pd.qcut(d, 10, labels=False))
    by_decile = candidate_coherence.groupby(deciles).mean()
    values.update({
        "coherence-outfits": percent(coherence(outfits).mean()),
        "coherence-candidates": percent(candidate_coherence.mean()),
        "coherence-closest-tenth": percent(by_decile.iloc[0]),
        "coherence-lowest-tenth": percent(by_decile.min()),
        "coherence-farthest-tenth": percent(by_decile.iloc[-1]),
    })

    rng = np.random.default_rng(SEED)
    overlaps = []
    for product in rng.choice(pd.unique(generated[list(PRODUCT_CLASSES)].stack()), 300, replace=False):
        top = rank_for_product(generated, product, k=3)[list(PRODUCT_CLASSES)]
        sets = [set(row.dropna()) - {product} for _, row in top.iterrows()]
        overlaps += [len(a & b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
    values["top3-shared-products"] = f"{np.mean(overlaps):.2f}"
    return values


def recommendations(generated, images, products):
    """Best candidate for each input product, then the worst one for the first."""
    rows, titles = [], []
    for letter, product in zip("abcd", products):
        rows.append(rank_for_product(generated, product, k=1))
        titles.append(f"({letter}) top 1\nd = {rows[-1]['distance'].iloc[0]:.1f}")
    worst = rank_for_product(generated, products[0], k=1, worst=True)
    rows.append(worst)
    titles.append(f"(e) worst for (a)\nd = {worst['distance'].iloc[0]:.1f}")
    values = {
        "worst-distance": f"{worst['distance'].iloc[0]:.1f}",
        "worst-size": f"{worst[list(PRODUCT_CLASSES)].notna().sum(axis=1).iloc[0]}",
    }
    return viz.show_outfits(pd.concat(rows), images, titles, highlight=products, cell=0.75), values


def evaluation_table(results, output):
    """LaTeX table of the fill-in-the-blank results: model scores per run, then baselines."""

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
    lines = []
    for run, frame in results.groupby("run", sort=False):
        lines.append(f"\\multicolumn{{3}}{{l}}{{\\emph{{{labels.get(run, run)}}}}} \\\\")
        lines += [row(frame, score) for score in MODEL_SCORES if score in set(frame["score"])]
    first = results[results["run"] == results["run"].iloc[0]]
    lines.append("\\multicolumn{3}{l}{\\emph{Baselines}} \\\\")
    lines += [row(first, score) for score in BASELINES]
    header = ["\\begin{tabular}{lll}", "\\toprule", "Score & All questions & Novel pairs \\\\", "\\midrule"]
    footer = ["\\bottomrule", "\\end{tabular}"]
    Path(output).write_text("\n".join(header + lines + footer) + "\n")


def evaluation_values(results, outfits, evaluation):
    """Split sizes and accuracies in %; model scores are keyed by run, baselines by subset and score only."""
    train_rows, test_rows = split_outfits(outfits, evaluation["held_out_share"], evaluation["seed"])
    first = results[results["run"] == results["run"].iloc[0]]
    values = {
        "fitb-held-out-share": percent(evaluation["held_out_share"], 0),
        "fitb-held-out": f"{len(test_rows):,}",
        "fitb-training": f"{len(train_rows):,}",
        "fitb-epochs": f"{first['epochs'].iloc[0]}",
        "fitb-chance": f"{100 / evaluation['options']:.0f}",
    }
    for subset, questions in first.groupby("subset", sort=False)["questions"].first().items():
        values[f"fitb-questions-{slug(subset)}"] = f"{questions:,}"
    for row in results[~results["score"].isin(BASELINES)].itertuples():
        values[f"fitb-{row.run}-{slug(row.subset)}-{slug(row.score)}"] = percent(row.accuracy)
    for row in first[first["score"].isin(BASELINES)].itertuples():
        values[f"fitb-{slug(row.subset)}-{slug(row.score)}"] = percent(row.accuracy)
    return values


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    training, generation, evaluation = config["training"], config["generation"], config["evaluation"]
    catalog, outfits, exported, model, images = load_resources(config)
    generated = pd.read_csv(
        generation["generated_outfits_path"], dtype={slot: str for slot in PRODUCT_CLASSES}
    )
    of_outfit = outfit_catalogues(outfits, catalogues(catalog, outfits)).to_numpy()
    samples = load_strips(training["train_sample_path"])
    indices = training["train_sample_indices"][: len(samples)]
    values = {
        "parameters": f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}",
        "batch-size": f"{training['batch_size']}",
        "learning-rate": f"{training['learning_rate']:g}",
        "training-seed": f"{training['seed']}",
        "sample-rows": ", ".join(map(str, indices[:-1])) + f" and {indices[-1]}",
        **loss_values(exported["loss"]),
    }

    save(loss_curve(exported["loss"]), "loss.png")
    figure, numbers = latent_projection(exported["latent_space"], outfits, of_outfit)
    save(figure, "latent_pca.png")
    values.update(numbers)
    figure, numbers = distance_panels(generated, images)
    save(figure, "distances.png")
    values.update(numbers)
    save(reconstructions(samples, model, indices), "reconstructions.png")

    metadata = pd.read_csv(config["data"]["products_path"], dtype={"cod_modelo_color": str})
    colour = metadata.drop_duplicates("cod_modelo_color").set_index("cod_modelo_color")["des_agrup_color_eng"]
    values.update(statistics(
        model,
        load_strips(training["train_dataset_path"]),
        outfits,
        exported["latent_space"],
        generated,
        np.load(generation["candidate_embeddings_path"]),
        colour,
    ))

    # Random input products drawn from the candidates, and a random white skirt.
    rng = np.random.default_rng(SEED)
    candidates = pd.unique(generated[list(PRODUCT_CLASSES)].stack())
    products = list(rng.choice(candidates, size=4, replace=False))
    figure, numbers = recommendations(generated, images, products)
    save(figure, "recommendations.png")
    values.update(numbers)
    white = metadata.loc[metadata["des_color_specification_esp"] == "BLANCO", "cod_modelo_color"]
    skirt = rng.choice(sorted(set(white) & set(generated["Bottoms"].dropna())))
    top3 = rank_for_product(generated, skirt, k=3)
    titles = [f"#{rank}\nd = {d:.1f}" for rank, d in enumerate(top3["distance"], start=1)]
    save(viz.show_outfits(top3, images, titles, highlight=skirt, cell=0.75), "top3.png")
    print(f"recommendation inputs {products}, white skirt {skirt}")

    if Path(evaluation["results_path"]).exists():
        results = pd.read_csv(evaluation["results_path"])
        evaluation_table(results, OUTPUT / "evaluation.tex")
        values.update(evaluation_values(results, outfits, evaluation))
    write_values(values, OUTPUT / "values.tex")
    print(f"Figures, tables and {len(values)} values written to {OUTPUT}")


if __name__ == "__main__":
    main()
