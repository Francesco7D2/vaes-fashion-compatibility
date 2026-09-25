"""Generate candidates without needing images or TensorFlow."""

import argparse
import json

from src.recommendation.candidates import (
    read_catalog,
    read_outfits,
    sample_candidates,
)
from src.utils.setup_utilities import ensure_parent, file_digest, load_config


def generate_candidates(config_path=None):
    config = load_config(config_path)
    data, rec = config["data"], config["recommendation"]
    candidates, frequencies = sample_candidates(
        read_outfits(data["outfits_w_products_path"]),
        read_catalog(data["products_path"]),
        data,
        rec["num_candidates"],
        rec["seed"],
    )
    candidates.to_csv(ensure_parent(rec["candidates_path"]), index=False)
    frequencies.to_csv(ensure_parent(rec["configurations_path"]), index=False)
    manifest = {
        "method": "empirical_configuration_uniform_unseen_products_with_replacement",
        "seed": rec["seed"],
        "draws": len(candidates),
        "unique_outfits": int(candidates.signature.nunique()),
        "allowed_configurations": len(frequencies),
        "observed_configurations": int((frequencies.training_count > 0).sum()),
        "rules": {
            key: data[key] for key in ("configurations_base", "optional_products")
        },
        "sha256": {
            key: file_digest(path)
            for key, path in {
                "training_outfits": data["outfits_w_products_path"],
                "products": data["products_path"],
                "candidates": rec["candidates_path"],
                "configurations": rec["configurations_path"],
            }.items()
        },
    }
    with open(ensure_parent(rec["generation_manifest_path"]), "w") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2))
    return candidates, frequencies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    generate_candidates(parser.parse_args().config)
