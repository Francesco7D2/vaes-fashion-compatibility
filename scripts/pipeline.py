"""Run the pipeline from raw CSVs to a trained model (and optionally generated outfits).

    python -m scripts.pipeline                      # preprocess + train
    python -m scripts.pipeline preprocess           # data preparation only
    python -m scripts.pipeline train generate       # reuse prepared data
    python -m scripts.pipeline all --resume         # skip steps that are up to date
    python -m scripts.pipeline --dry-run            # show the plan and exit
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.utils.setup_utilities import PROJECT_ROOT, ensure_parent, load_config

LOGGER = logging.getLogger("pipeline")
LOG_FILE = str(PROJECT_ROOT / "reports/logs/pipeline.log")


@dataclass(frozen=True)
class Step:
    """One pipeline step; inputs and outputs are (config section, key) path entries."""

    name: str
    group: str
    run: Callable
    inputs: tuple
    outputs: tuple


def _valid_outfits(config_path, overrides):
    from scripts.data_processing.build_valid_outfits_df import build_valid_outfits_df

    build_valid_outfits_df(config_path)


def _outfit_table(config_path, overrides):
    from scripts.data_processing.build_outfit_w_products_df import (
        build_outfit_w_products,
    )

    build_outfit_w_products(config_path)


def _image_arrays(config_path, overrides):
    from scripts.data_processing.build_image_arrays import build_image_arrays

    build_image_arrays(config_path=config_path)


def _train(config_path, overrides):
    from scripts.training.train_vae import train_outfit_embeddings

    train_outfit_embeddings(config_path, overrides)


def _generate(config_path, overrides):
    from scripts.generation.generate_outfits import generate_outfits

    generate_outfits(config_path, overrides)


STEPS = (
    Step(
        "valid-outfits",
        "preprocess",
        _valid_outfits,
        inputs=(("data", "outfits_path"), ("data", "products_path")),
        outputs=(("data", "valid_outfits_path"),),
    ),
    Step(
        "outfit-table",
        "preprocess",
        _outfit_table,
        inputs=(("data", "valid_outfits_path"),),
        outputs=(("data", "outfits_w_products_path"),),
    ),
    Step(
        "image-arrays",
        "preprocess",
        _image_arrays,
        inputs=(("data", "valid_outfits_path"), ("data", "outfits_w_products_path")),
        outputs=(
            ("training", "train_dataset_path"),
            ("training", "train_sample_path"),
        ),
    ),
    Step(
        "train",
        "train",
        _train,
        inputs=(
            ("training", "train_dataset_path"),
            ("training", "train_sample_path"),
        ),
        outputs=(
            ("training", "model_weights_path"),
            ("training", "model_data_path"),
        ),
    ),
    Step(
        "generate",
        "generate",
        _generate,
        inputs=(
            ("data", "products_path"),
            ("data", "outfits_w_products_path"),
            ("training", "model_weights_path"),
            ("training", "model_data_path"),
        ),
        outputs=(("generation", "generated_outfits_path"),),
    ),
)
GROUPS = ("preprocess", "train", "generate")
DEFAULT_GROUPS = ("preprocess", "train")


def select_steps(groups):
    """Steps belonging to the requested groups, in pipeline order."""
    requested = set(GROUPS if "all" in groups else groups)
    return [step for step in STEPS if step.group in requested]


def _paths(config, entries):
    return [Path(config[section][key]) for section, key in entries]


def is_up_to_date(step, config):
    """True when every output exists and is newer than every input."""
    outputs = _paths(config, step.outputs)
    if not all(path.exists() for path in outputs):
        return False
    inputs = [path for path in _paths(config, step.inputs) if path.exists()]
    newest_input = max((path.stat().st_mtime for path in inputs), default=0)
    return min(path.stat().st_mtime for path in outputs) >= newest_input


def missing_inputs(steps, config):
    """Input files absent on disk that no earlier selected step will produce."""
    produced, missing = set(), []
    for step in steps:
        for path in _paths(config, step.inputs):
            if not path.exists() and path not in produced and path not in missing:
                missing.append(path)
        produced.update(_paths(config, step.outputs))
    return missing


def run_pipeline(
    config_path=None, groups=DEFAULT_GROUPS, resume=False, dry_run=False, overrides=None
):
    """Run the selected steps; returns a list of (step name, status, seconds)."""
    config = load_config(config_path, overrides)
    steps = select_steps(groups)
    plan = [
        (step, "skip" if resume and is_up_to_date(step, config) else "run")
        for step in steps
    ]
    to_run = [step for step, action in plan if action == "run"]
    missing = missing_inputs(to_run, config)
    if missing:
        raise FileNotFoundError(
            "Missing input files:\n  " + "\n  ".join(map(str, missing))
        )
    results = []
    for step, action in plan:
        if dry_run or action == "skip":
            status = "planned" if dry_run and action == "run" else "up to date"
            LOGGER.info("%-14s %s", step.name, status)
            results.append((step.name, status, 0.0))
            continue
        LOGGER.info("%-14s started", step.name)
        start = time.monotonic()
        step.run(config_path, overrides)
        elapsed = time.monotonic() - start
        LOGGER.info("%-14s finished in %.1fs", step.name, elapsed)
        results.append((step.name, "done", elapsed))
    return results


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m scripts.pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "groups",
        nargs="*",
        choices=(*GROUPS, "all"),
        metavar="STAGE",
        help="preprocess, train, generate or all (default: preprocess train)",
    )
    parser.add_argument("--config", help="YAML config file (default: config.yml)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip steps whose outputs are newer than their inputs",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan without running it"
    )
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument("--batch-size", type=int, help="override training.batch_size")
    parser.add_argument(
        "--num-candidates", type=int, help="override generation.num_candidates"
    )
    return parser


def _overrides(args):
    values = {
        "training": {"epochs": args.epochs, "batch_size": args.batch_size},
        "generation": {"num_candidates": args.num_candidates},
    }
    values = {
        section: {key: value for key, value in items.items() if value is not None}
        for section, items in values.items()
    }
    return {section: items for section, items in values.items() if items}


def _configure_logging():
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(ensure_parent(LOG_FILE)),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    _configure_logging()
    try:
        results = run_pipeline(
            args.config, args.groups or DEFAULT_GROUPS, args.resume, args.dry_run, _overrides(args)
        )
    except (FileNotFoundError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130
    LOGGER.info(
        "Finished: %s",
        ", ".join(f"{name} ({status})" for name, status, _ in results) or "nothing",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
