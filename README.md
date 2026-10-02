# Fashion compatibility with outfit embeddings

A research prototype that recommends complete outfits from a single product, using the MANGO data of the [UPC Datathon fashion-compatibility challenge](https://github.com/Francesco7D2/datathon-2023-fashion-compatibility). Each outfit becomes one image with ten fixed product slots, a convolutional autoencoder with a stochastic latent layer (PyTorch) learns a 256-dimensional embedding per outfit, and random candidate outfits made of unseen products are ranked by their distance to the training outfits.

- [Report](reports/report.pdf): methodology, results and next steps.
- [Project overview](docs/project-walkthrough.md): pipeline, code map, model and limitations.

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

## Devices

Training and generation run on the fastest available device: an NVIDIA GPU (CUDA), then an Apple Silicon GPU (MPS), then the CPU. Set `device` in `config.yml` or pass `--device`.

On an Apple M5 Pro, one epoch over 3,164 outfits (batch 32) takes about 13 seconds on the GPU and about 200 seconds on the CPU, so the default 400 epochs take about 1.5 hours on the GPU.

## Outfit generation

`scripts/generation/generate_outfits.py` (settings under `generation:` in `config.yml`):

1. Draw `num_candidates` random outfits. The slot pattern follows the frequency of patterns among the training outfits; products come from `candidate_pool` (`unseen`: never in a training outfit).
2. Encode each candidate to its posterior mean, with the same strip construction as training (including the JPEG round trip of the archives).
3. Score it by the Euclidean distance to the nearest training embedding.
4. Write `generated_outfits.csv` (slots, `distance`, `distance_percentile`, `nearest_training_row`) and `candidate_embeddings.npy`.

To recommend for a product, `src.generation.retrieval.rank_for_product(generated, product_id, k)` returns the closest candidates containing it. Distance to a known outfit is a hypothesis for compatibility, not a validated score; it also grows with the share of the image covered by products (see the report). Generating and scoring 100,000 candidates takes under a minute on an Apple M5 Pro GPU.

## Notebooks

The notebooks only call code from `src/` and `scripts/`. Set `FASHION_CONFIG` to use a different config file.

- `notebooks/exploratory-data-analysis.ipynb`: outfit sizes, product descriptions, outfit rules, selection and the train/test overlap analysis.
- `notebooks/training.ipynb`: outfit strips, model, optional short demo training (writes to `models/demo/`), loss curve, reconstructions, latent-space projection.
- `notebooks/outfit-generation.ipynb`: candidates, encoding, distances, closest and farthest candidates beside their nearest training outfit, and recommendations for a product.

The training and generation notebooks are committed without outputs.

## Model and outputs

The encoder produces 256 means and 256 log-variances, samples a latent vector and decodes image logits. **The loss is the mean pixel binary cross entropy only, not the ELBO: there is no KL term**, so random prior samples are not expected to decode into useful outfits. The final encoder ReLU also applies to the means and log-variances.

Every 20 epochs and at the final epoch, training saves:

- `models/vae_256.pt`: PyTorch `state_dict` (load with `src.models.vae256.load_trained_model`).
- `models/vae_data_256.pkl`: `latent_space` (N × 256 NumPy array of posterior means) and `reconstruction_loss` (epoch means). Rows match `outfit_w_products.csv`. Only load trusted pickle files.
- `figures/epoch_images_256/`: original and reconstructed training previews.

## Report

The report source is `reports/report.tex`. After training and generation, redraw its figures and compile it:

```bash
python -m scripts.report.make_figures
cd reports && latexmk -pdf report.tex
```

## Next steps

1. **Quantitative evaluation:** a protocol independent of visual inspection (for example fill-in-the-blank on outfits with unseen products, or human ratings) and simple baselines (random candidates, colour and category rules, nearest neighbours on pretrained image features).
2. **A score independent of coverage:** compare candidates only with training outfits of the same configuration, or normalize the distance by outfit size or covered share.
3. **A complete VAE:** add a weighted KL term and a linear mean/log-variance layer instead of the final ReLU.
4. **A better image objective:** weight product pixels against the white background, use a perceptual loss, or use pretrained per-product features.
5. **Training:** shuffle outfits every epoch, keep the lowest-loss checkpoint and add a learning-rate schedule.
6. **Candidate generation:** generate candidates around the input product, so that every product gets enough candidates.
7. **Serving:** an API over the ranked table, with filters for configuration, colour and material.
