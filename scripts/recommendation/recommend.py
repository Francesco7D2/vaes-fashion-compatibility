"""Look up precomputed recommendations; no TensorFlow or images are needed."""

import argparse
import json

from src.recommendation.ranking import recommend
from src.utils.setup_utilities import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    path = load_config(args.config)["recommendation"]["index_path"]
    results = recommend(path, args.product_id, args.top_k)
    print(
        json.dumps(
            {
                "product_id": args.product_id,
                "results": results,
                "message": "OK"
                if results
                else "No sampled candidate contains this product; it may be a training product, unsupported, or unsampled.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
