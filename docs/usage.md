# Usage

How to set up, run and evaluate the project. For the design and the code map
see [project-walkthrough.md](project-walkthrough.md); for the results see the
[README](../README.md) and the [report](../reports/report.pdf).

## Setup

Use Python 3.12 (tested with PyTorch 2.14). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Tests generate a tiny synthetic dataset and perform one training update with the actual model. No MANGO data or GPU is required; with a GPU, one more test checks that its gradients match the CPU's. The large dense layers mean even this smoke test needs substantial memory.

On Linux without an NVIDIA GPU, install the CPU wheel first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) to avoid downloading several GB of CUDA libraries.

## Required data

The data, images and trained weights are not committed. Supply:

- `data/tabular/outfit_data.csv`: `cod_outfit`, `cod_modelo_color`.
- `data/tabular/product_data.csv`: `cod_modelo_color`, `des_product_category`, `des_product_family`, `des_product_type`, `des_filename`.
- Product image files referenced by `des_filename`.

Product metadata must contain one row per product ID. IDs are strings, to keep leading zeros. The source spelling `Accesories, Swim and Intimate` is intentional. Image paths starting with `datathon/` map to `data/`; other relative image paths resolve from the parent of the data directory. Absolute image paths are accepted.

## Run

Edit `config.yml`, then run from the repository root:

```bash
python -m scripts.pipeline                # preprocessing + training
python -m scripts.pipeline all            # ... plus outfit generation
python -m scripts.pipeline train generate # reuse already prepared data
python -m scripts.pipeline all --resume   # skip steps whose outputs are up to date
python -m scripts.pipeline --dry-run      # print the plan without running it
python -m scripts.pipeline train --device cpu  # force a device
```

Stages are `preprocess`, `train`, `generate` and `all`. Options: `--config PATH`, `--epochs N`, `--batch-size N`, `--num-candidates N`, `--device {auto,cpu,mps,cuda}`. Missing input files are reported before any step starts, and progress is logged to the console and `reports/logs/pipeline.log`.

Each step is also available on its own (use module invocation; running the script files directly is not supported):

```bash
python -m scripts.data_processing.build_valid_outfits_df
python -m scripts.data_processing.build_outfit_w_products_df
python -m scripts.data_processing.build_image_arrays
python -m scripts.training.train_vae
python -m scripts.generation.generate_outfits
```

Each step function also accepts `config_path` for a custom YAML file. Configured paths resolve relative to that file, and output directories are created automatically.

1. Select outfits matching dress + footwear + earrings, or top + bottom + footwear + earrings, plus any subset of five optional classes (64 configurations).
2. Create a table with one row per outfit and ten product slots.
3. Resize images to 128 × 96 and concatenate them horizontally; missing products become white slots. Write full and preview archives (`.npz` of JPEG bytes).
4. Train the 256-dimensional model and export weights, embeddings, loss history and previews.
5. Generate outfits (see below).

Slot order: Tops, Bottoms, Outerwear, Dresses/jumpsuits/complete sets, Footwear, Bags, Glasses, Earrings, Ring, Necklace. Each outfit becomes a 128 × 960 RGB strip, which the model reads as a 3 × 128 × 960 tensor.

`train_sample_indices` are zero-based positions in the outfit table; update them for your data. They select training previews, not a held-out evaluation. Reduce `batch_size` if memory is limited. The default duration is 400 epochs.

Training visits the outfits in a new random order every epoch (`shuffle`) and exports the weights of its lowest-loss epoch (`keep_best`). `linear_head`, `kl_weight` and `foreground_weight` select model and loss variants; see `config.yml` and the evaluation below.

## Devices

Training and generation run on the fastest available device: an NVIDIA GPU (CUDA), then an Apple Silicon GPU (MPS), then the CPU. Set `device` in `config.yml` or pass `--device`.

On an Apple M5 Pro, one epoch over 3,164 outfits (batch 32) takes about 13.5 seconds on the GPU and about 200 seconds on the CPU, so the default 400 epochs take about 1.5 hours on the GPU.

## Outfit generation

`scripts/generation/generate_outfits.py` (settings under `generation:` in `config.yml`):

1. Draw `num_candidates` random outfits. The slot pattern follows the frequency of patterns among the training outfits; products come from `candidate_pool` (`unseen`: never in a training outfit). With `same_catalogue`, every candidate takes its products from one catalogue: image years that appear together in training outfits.
2. Encode each candidate to its posterior mean, with the same strip construction as training (including the JPEG round trip of the archives).
3. Score it by the Euclidean distance to the nearest training embedding.
4. Write `generated_outfits.csv` (slots, `distance`, `distance_percentile`, `nearest_training_row`) and `candidate_embeddings.npy`.

To recommend for a product, `src.generation.retrieval.rank_for_product(generated, product_id, k)` returns the closest candidates containing it. Random candidates contain some products much more often than others; `recommend` instead generates candidates around one product and ranks them in a few seconds:

```bash
python -m scripts.generation.recommend PRODUCT_ID --k 3 --figure outfits.png
```

Distance to a known outfit is a hypothesis for compatibility, not a validated score; it also grows with the share of the image covered by products (see the report). Generating and scoring 100,000 candidates takes under a minute on an Apple M5 Pro GPU.

## Evaluation

`scripts/evaluation/evaluate_fitb.py` measures how well a score recognises real outfits with a fill-in-the-blank test (settings under `evaluation:` in `config.yml`):

1. Hold out a share of the outfits, stratified by configuration, and train a model on the others.
2. For every occupied slot of a held-out outfit, compare the true product with decoys of the same class and catalogue that appear in training outfits.
3. Report the share of questions each score answers correctly, with a 95 % bootstrap interval over outfits, for all questions and for *novel pairs*: questions whose true product never appears in a training outfit with the outfit's other products.

Scores: distance to the nearest training outfit (of any or of the same configuration), reconstruction error, and baselines (lightest option, colour coherence, product co-occurrence in training outfits, random). Each named run adds its rows to `reports/evaluation.csv`:

```bash
python -m scripts.evaluation.evaluate_fitb --name vae
python -m scripts.evaluation.evaluate_fitb --name reconstruction --no-linear-head --kl-weight 0
python -m scripts.evaluation.evaluate_fitb --name foreground --no-linear-head --kl-weight 0 --foreground-weight 5
```

## Notebooks

The notebooks only call code from `src/` and `scripts/`. Set `FASHION_CONFIG` to use a different config file.

- `notebooks/exploratory-data-analysis.ipynb`: outfit sizes, product descriptions, outfit rules, selection, catalogues and the train/test overlap analysis.
- `notebooks/training.ipynb`: outfit strips, model, optional short demo training (writes to `models/demo/`), loss curve, reconstructions, latent-space projection.
- `notebooks/outfit-generation.ipynb`: candidates, encoding, distances, closest and farthest candidates beside their nearest training outfit, and recommendations for a product.

The training and generation notebooks are committed without outputs.

## Model and outputs

The encoder produces 256 means and 256 log-variances from a linear layer, samples a latent vector and decodes image logits. The default model is a complete VAE: the loss is the mean pixel binary cross entropy plus the KL divergence to a standard normal prior, divided by the number of pixels (`kl_weight: 1.0`). `kl_weight: 0.0` trains with the reconstruction loss only, `linear_head: false` adds a ReLU on the means and log-variances, and `foreground_weight` gives product pixels more weight than the background. The evaluation compares these variants.

Every 20 epochs and at the end, training saves (the final export holds the lowest-loss epoch when `keep_best` is set):

- `models/vae_256.pt`: architecture option and PyTorch `state_dict` (written by `src.models.vae256.save_model`, read by `load_trained_model`).
- `models/vae_data_256.pkl`: `latent_space` (N × 256 NumPy array of posterior means) and `loss` (epoch means). Rows match `outfit_w_products.csv`. Only load trusted pickle files.
- `figures/epoch_images_256/`: original and reconstructed training previews.

## Report

The report source is `reports/report.tex`. After training, generation and evaluation, redraw its figures and compile it:

```bash
python -m scripts.report.make_figures
cd reports && latexmk -pdf report.tex
```

`make_figures` also writes the evaluation table (`reports/figures/evaluation.tex`) and every number quoted in the text (`reports/figures/values.tex`, one `\setval{key}{value}` line each). The report prints a number with `\val{key}`, so a new run updates the text without copying numbers by hand; a key missing from the file appears in bold as `??key` and LaTeX warns about it. The descriptions of the example outfits and the interpretation of the results remain hand-written and need a review after each run.

## All next steps

The report details the evidence behind each step.

1. **Combine the signals:** co-occurrence, colour coherence and visual distance capture different information; train a ranker on fill-in-the-blank questions from training outfits.
2. **Better visual features:** per-product features from an image encoder pretrained on fashion images, or normalized photo backgrounds, so that embeddings depend less on catalogue photo style and product darkness.
3. **A ranking independent of size and darkness:** extend the evaluation to candidates of different configurations and normalize the distance accordingly.
4. **Tune the model:** combine the complete VAE with foreground weighting, vary the KL weight and the number of epochs.
5. **Human evaluation:** stylist ratings of top-ranked and random candidates.
6. **More data and metadata:** relax the outfit rules, and give the model colour, fabric and category metadata.
7. **Serving:** an API over `recommend`, with filters for configuration, colour and material.
