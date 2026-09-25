"""Run the complete reproducible outfit pipeline or a selected range of stages."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.utils.setup_utilities import ensure_parent, load_config

STAGES = (
    ("select", "scripts.data_processing.build_valid_outfits_df"),
    ("reshape", "scripts.data_processing.build_outfit_w_products_df"),
    ("images", "scripts.data_processing.build_image_arrays"),
    ("train", "scripts.training.train_vae"),
    ("candidates", "scripts.recommendation.generate_candidates"),
    ("index", "scripts.recommendation.build_index"),
)


def run_pipeline(
    config_path=None,
    from_stage="select",
    to_stage="index",
    dry_run=False,
    device=None,
):
    """Run inclusive stage bounds; each stage gets a fresh process."""
    names = [name for name, _ in STAGES]
    start, stop = names.index(from_stage), names.index(to_stage)
    if start > stop:
        raise ValueError("from-stage must come before or equal to to-stage")
    config = load_config(config_path)
    manifest_path = config["pipeline"]["run_manifest_path"]
    selected = STAGES[start : stop + 1]
    run = {
        "config": str(Path(config_path).resolve())
        if config_path
        else str(Path("config.yml").resolve()),
        "from_stage": from_stage,
        "to_stage": to_stage,
        "status": "DRY_RUN" if dry_run else "RUNNING",
        "device_override": device,
        "stages": [],
    }
    for name, module in selected:
        command = [sys.executable, "-m", module]
        if config_path:
            command.extend(["--config", str(Path(config_path).resolve())])
        if device and name in {"train", "index"}:
            command.extend(["--device", device])
        print(f"\n[{name}] {' '.join(command)}", flush=True)
        if dry_run:
            run["stages"].append(
                {"name": name, "command": command, "status": "DRY_RUN"}
            )
            continue
        started = time.monotonic()
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            run["stages"].append(
                {
                    "name": name,
                    "command": command,
                    "status": "FAILED",
                    "return_code": error.returncode,
                    "seconds": time.monotonic() - started,
                }
            )
            run["status"] = "FAILED"
            with open(ensure_parent(manifest_path), "w") as stream:
                json.dump(run, stream, indent=2)
            raise
        run["stages"].append(
            {
                "name": name,
                "command": command,
                "status": "COMPLETE",
                "seconds": time.monotonic() - started,
            }
        )
    if not dry_run:
        run["status"] = "COMPLETE"
    with open(ensure_parent(manifest_path), "w") as stream:
        json.dump(run, stream, indent=2)
    print(f"\nPipeline {run['status'].lower()}; manifest: {manifest_path}")
    return run


def main():
    names = [name for name, _ in STAGES]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--from-stage", choices=names, default=names[0])
    parser.add_argument("--to-stage", choices=names, default=names[-1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"))
    args = parser.parse_args()
    run_pipeline(args.config, args.from_stage, args.to_stage, args.dry_run, args.device)


if __name__ == "__main__":
    main()
