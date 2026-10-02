"""Configuration and output-directory helpers."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICES = ("auto", "cpu", "mps", "cuda")


def load_config(config_path=None, overrides=None):
    """Resolve paths relative to the YAML file, not the shell cwd.

    `overrides` is a nested dict (e.g. {"training": {"epochs": 3}}) merged over
    the YAML values before paths are resolved.
    """
    path = (Path(config_path) if config_path else PROJECT_ROOT / "config.yml").resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    for section, values in (overrides or {}).items():
        config.setdefault(section, {}).update(values)
    for section in ("data", "training", "generation"):
        for key, value in config.get(section, {}).items():
            if key.endswith("_path"):
                config[section][key] = str((path.parent / value).resolve())
    return config


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path
