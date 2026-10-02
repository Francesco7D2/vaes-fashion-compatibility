"""Recommend outfits for one product from candidates generated around it.

    python -m scripts.generation.recommend PRODUCT_ID --k 3 --figure outfits.png
"""

import argparse

from scripts.generation.generate_outfits import load_resources, score_candidates
from src.data_processing.schema import PRODUCT_CLASSES
from src.generation.candidates import sample_candidates_for_product
from src.utils import visualization as viz
from src.utils.setup_utilities import load_config


def recommend(product_id, k=3, num_candidates=1000, config_path=None):
    """The `k` lowest-distance candidates among `num_candidates` containing the product."""
    config = load_config(config_path)
    generation = config["generation"]
    catalog, outfits, exported, model, images = load_resources(config)
    candidates = sample_candidates_for_product(
        product_id, catalog, outfits, generation["candidate_pool"], num_candidates, generation["seed"]
    ).drop_duplicates(list(PRODUCT_CLASSES))
    score_candidates(candidates, model, images, exported["latent_space"], generation, progress=False)
    return candidates.nsmallest(k, "distance"), images


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m scripts.generation.recommend")
    parser.add_argument("product", help="product ID (cod_modelo_color)")
    parser.add_argument("--k", type=int, default=3, help="outfits to return")
    parser.add_argument("--candidates", type=int, default=1000, help="candidates to generate")
    parser.add_argument("--config", help="YAML config file (default: config.yml)")
    parser.add_argument("--figure", help="save the outfits as an image")
    args = parser.parse_args(argv)
    best, images = recommend(args.product, args.k, args.candidates, args.config)
    columns = [slot for slot in PRODUCT_CLASSES if best[slot].notna().any()]
    print(best[[*columns, "distance"]].to_string(index=False))
    if args.figure:
        titles = [f"#{rank}\nd = {d:.1f}" for rank, d in enumerate(best["distance"], start=1)]
        fig = viz.show_outfits(best, images, titles, highlight=args.product, cell=0.9)
        fig.savefig(args.figure, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
