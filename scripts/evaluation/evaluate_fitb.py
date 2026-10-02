"""Train on most outfits and measure fill-in-the-blank accuracy on the others.

    python -m scripts.evaluation.evaluate_fitb --name vae
    python -m scripts.evaluation.evaluate_fitb --name reconstruction --no-linear-head --kl-weight 0

Results of every named run are kept in `evaluation.results_path`.
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.generation.generate_outfits import load_catalog
from scripts.training.train_vae import training_options
from src.data_processing.image_processing import build_outfit_strip, jpeg_round_trip, load_strips
from src.data_processing.schema import PRODUCT_CLASSES
from src.evaluation.fitb import (
    colour_scores,
    cooccurrence_scores,
    fitb_accuracy,
    fitb_questions,
    slot_patterns,
    split_outfits,
)
from src.generation.candidates import catalogues
from src.generation.embeddings import ProductImages, encode_strips
from src.generation.retrieval import nearest_training_outfit
from src.models.vae256 import Conv_VAE
from src.utils.setup_utilities import ensure_parent, load_config
from src.utils.torch_utilities import resolve_device, to_model_input
from src.utils.train_utilities import fit


@torch.inference_mode()
def model_scores(model, questions, images, train_embeddings, train_patterns, batch_size=64):
    """Model scores of every option (lower is better) and the darkness of its strip."""
    device = next(model.parameters()).device
    records = questions.to_dict("records")

    def strip(record):
        return jpeg_round_trip(build_outfit_strip(record, images, images.width, images.height))

    means, errors, darkness = [], [], []
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        for start in range(0, len(records), batch_size):
            batch = np.stack(list(pool.map(strip, records[start : start + batch_size])))
            x = to_model_input(batch, device)
            mean = model.encode(x)[0]
            pixels = F.binary_cross_entropy_with_logits(model.decode(mean), x, reduction="none")
            errors.append(pixels.mean(dim=(1, 2, 3)).cpu().numpy())
            means.append(mean.cpu().numpy())
            darkness.append(1 - batch.reshape(len(batch), -1).mean(axis=1) / 255)
    means = np.concatenate(means)
    distance = nearest_training_outfit(means, train_embeddings)[0]
    same = distance.copy()
    option_patterns = slot_patterns(questions)
    for pattern in np.unique(option_patterns):
        options, training = option_patterns == pattern, train_patterns == pattern
        if training.any():
            same[options] = nearest_training_outfit(means[options], train_embeddings[training])[0]
    def unit(vectors):
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    scores = {
        "distance": distance,
        "distance (same configuration)": same,
        "cosine distance": nearest_training_outfit(unit(means), unit(train_embeddings))[0],
        "reconstruction error": np.concatenate(errors),
    }
    return scores, np.concatenate(darkness)


def evaluate(name, config_path=None, overrides=None):
    config = load_config(config_path, overrides)
    data, training, evaluation = config["data"], config["training"], config["evaluation"]
    torch.manual_seed(training.get("seed", 42))
    device = resolve_device(training.get("device", "auto"))
    catalog, outfits = load_catalog(config)
    train_rows, test_rows = split_outfits(outfits, evaluation["held_out_share"], evaluation["seed"])
    print(f"{len(train_rows)} training and {len(test_rows)} held-out outfits")

    dataset = torch.from_numpy(load_strips(training["train_dataset_path"])[train_rows]).to(device)
    model = Conv_VAE(linear_head=training.get("linear_head", True)).to(device)
    history = fit(model, dataset, **{**training_options(training), "epochs": evaluation["epochs"]})
    model.eval()
    train_embeddings = encode_strips(model, dataset, training["batch_size"])
    del dataset

    catalogue = catalogues(catalog, outfits)
    questions = fitb_questions(
        outfits, train_rows, test_rows, catalogue, evaluation["options"], evaluation["seed"]
    )
    images = ProductImages(
        catalog.set_index("cod_modelo_color")["des_filename"],
        data["image_width"],
        data["image_height"],
    )
    scores, darkness = model_scores(
        model, questions, images, train_embeddings, slot_patterns(outfits.iloc[train_rows])
    )
    products = pd.read_csv(data["products_path"], dtype={"cod_modelo_color": str})
    colour = products.drop_duplicates("cod_modelo_color").set_index("cod_modelo_color")[
        "des_agrup_color_eng"
    ]
    cooccurrence = cooccurrence_scores(questions, outfits, train_rows)
    scorers = [
        *((score, values, False) for score, values in scores.items()),
        ("lightest option", darkness, False),
        ("colour coherence", colour_scores(questions, colour), True),
        ("co-occurrence", cooccurrence, True),
        ("random", np.random.default_rng(evaluation["seed"]).random(len(questions)), False),
    ]
    # Questions whose true product never appears with the outfit's other products
    # in a training outfit: memorized product pairs cannot answer them.
    novel = questions.loc[questions["is_true"] & (cooccurrence == 0), "question"]
    subsets = {"all": np.ones(len(questions), bool), "novel pairs": questions["question"].isin(novel).to_numpy()}
    rows = []
    for subset, mask in subsets.items():
        for score, values, higher_is_better in scorers:
            accuracy, low, high = fitb_accuracy(questions[mask], values[mask], higher_is_better)
            rows.append(
                {
                    "run": name,
                    "subset": subset,
                    "score": score,
                    "accuracy": accuracy,
                    "ci_low": low,
                    "ci_high": high,
                    "questions": questions.loc[mask, "question"].nunique(),
                    "epochs": evaluation["epochs"],
                    "final_loss": history[-1],
                    **{key: training.get(key) for key in ("linear_head", "kl_weight", "foreground_weight")},
                }
            )
    results = pd.DataFrame(rows)
    path = Path(evaluation["results_path"])
    if path.exists():
        previous = pd.read_csv(path)
        results = pd.concat([previous[previous["run"] != name], results], ignore_index=True)
    results.to_csv(ensure_parent(path), index=False)
    columns = ["subset", "score", "accuracy", "ci_low", "ci_high", "questions"]
    print(results.loc[results["run"] == name, columns].round(3).to_string(index=False))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m scripts.evaluation.evaluate_fitb")
    parser.add_argument("--name", required=True, help="label of this run in the results table")
    parser.add_argument("--config", help="YAML config file (default: config.yml)")
    parser.add_argument("--epochs", type=int, help="override evaluation.epochs")
    parser.add_argument(
        "--linear-head",
        action=argparse.BooleanOptionalAction,
        help="override training.linear_head (no ReLU on means and log-variances)",
    )
    parser.add_argument("--kl-weight", type=float, help="override training.kl_weight")
    parser.add_argument("--foreground-weight", type=float, help="override training.foreground_weight")
    args = parser.parse_args(argv)
    training = {
        key: value
        for key, value in (
            ("linear_head", args.linear_head),
            ("kl_weight", args.kl_weight),
            ("foreground_weight", args.foreground_weight),
        )
        if value is not None
    }
    overrides = {"training": training, "evaluation": {"epochs": args.epochs} if args.epochs else {}}
    evaluate(args.name, args.config, overrides)


if __name__ == "__main__":
    main()
