# Fashion compatibility with variational autoencoders

Does a variational autoencoder (VAE) trained on images of whole outfits learn
which products go together? A PyTorch study on the MANGO data of the
[UPC Datathon 2023](https://github.com/Francesco7D2/datathon-2023-fashion-compatibility).

![Best-ranked outfits for four products, and the worst-ranked one for the first](reports/figures/recommendations.png)

*Best-ranked outfit for four products (red frame), and the worst-ranked one
for the first.*

## Results

Fill-in-the-blank test on held-out outfits: pick the true product among 4
options (chance 25%).

| Score | All questions | Unseen product pairs |
| --- | --- | --- |
| Product co-occurrence | 57.3% | 17.7% |
| VAE latent distance | 44.1% | 32.6% |
| Colour coherence | 33.4% | 32.2% |
| Random | 24.2% | 22.8% |

- The VAE beats random, but loses to co-occurrence on products already seen
  together.
- On unseen product pairs it does no better than a colour heuristic.
- The embeddings mostly capture photo darkness (48% of the ranking's variance)
  and catalogue style, not compatibility.

Confidence intervals and model variants: [evaluation.csv](reports/evaluation.csv)
and the [report](reports/report.pdf).

## How it works

Each outfit becomes one image with ten product slots, a convolutional VAE
learns a 256-dimensional embedding per outfit, and candidate outfits are
ranked by distance to the nearest training outfit. Details in
[docs/project-walkthrough.md](docs/project-walkthrough.md).

## Next step

Replace whole-outfit pixels with per-product embeddings from a pretrained
image encoder, and train a ranker that combines them with co-occurrence and
colour.

## Run it

With Python 3.12:

```bash
python -m pip install -r requirements.txt
python -m pytest -q              # no data needed
python -m scripts.pipeline all   # needs the MANGO data
```

The data is not included; see [docs/usage.md](docs/usage.md).

## History

Started after the UPC Datathon 2023. First version in TensorFlow (2024);
rewritten in PyTorch with Claude Code (2026). Code under the MIT licence.

## Data availability

The MANGO dataset is not included. Redistribution permissions have not been verified, so I cannot currently provide a copy. Running the experiments requires access to the original data; see docs/usage.md for the expected files.