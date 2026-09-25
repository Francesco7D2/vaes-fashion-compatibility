"""Configuration and output-directory helpers."""

import hashlib
import logging
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path=None):
    """Resolve paths relative to the YAML file, not the shell cwd."""
    path = (Path(config_path) if config_path else PROJECT_ROOT / "config.yml").resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    for section in ("data", "training", "recommendation", "pipeline"):
        for key, value in config.get(section, {}).items():
            if key.endswith("_path"):
                config[section][key] = str((path.parent / value).resolve())
    return config


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_file="reports/logs/main.log", log_level=logging.INFO):
    logging.basicConfig(
        filename=ensure_parent(log_file),
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
