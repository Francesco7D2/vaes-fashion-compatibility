# Fashion compatibility with outfit embeddings

A research prototype following the [UPC Datathon project](https://github.com/Francesco7D2/datathon-2023-fashion-compatibility). It learns to reconstruct complete outfits from a compressed representation of product images.

**Current scope:** preprocessing, outfit-image TFRecords, a convolutional stochastic autoencoder, embedding export, and the report's outfit generation: random candidate outfits are encoded and ranked by latent distance to the training outfits. The [original report](reports/report.pdf) figures are historical and not reproduced by this code.

Read [the project walkthrough](docs/project-walkthrough.md) for architecture, report/code discrepancies, and research next steps.

## Setup

Use Python 3.12 (tested with TensorFlow 2.16.2). From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Tests generate a tiny synthetic dataset and perform one training update using the actual model. No MANGO data or GPU is required. Large dense layers mean even this smoke test needs substantial memory.

## Required data

The data, images, and trained weights are not committed. Supply:

- `data/tabular/outfit_data.csv`: `cod_outfit`, `cod_modelo_color`.
- `data/tabular/product_data.csv`: `cod_modelo_color`, `des_product_category`, `des_product_family`, `des_product_type`, `des_filename`.
- Product image files referenced by `des_filename`.

Product metadata must contain one row per product ID. IDs are strings to retain leading zeros. The source spelling `Accesories, Swim and Intimate` is intentional. Image paths starting with `datathon/` map to `data/`; other relative image paths resolve from the parent of the data directory. Absolute image paths are accepted.

## Run

Edit `config.yml`, then run from the repository root:

```bash
python -m scripts.pipeline                # preprocessing + training
python -m scripts.pipeline all            # ... plus outfit generation
python -m scripts.pipeline train generate # reuse already prepared data
python -m scripts.pipeline all --resume   # skip steps whose outputs are up to date
python -m scripts.pipeline --dry-run      # print the plan without running it
```

Stages are `preprocess`, `train`, `generate` and `all`. Options: `--config PATH`, `--epochs N`, `--batch-size N`, `--num-candidates N`. Missing input files are reported before any step starts, and progress is logged to the console and `reports/logs/pipeline.log`.

Each step is also available on its own (use module invocation; direct script-file invocation is not supported):

```bash
python -m scripts.data_processing.build_valid_outfits_df
python -m scripts.data_processing.build_outfit_w_products_df
python -m scripts.data_processing.build_image_arrays
python -m scripts.training.train_vae
python -m scripts.generation.generate_outfits
```

Each step function also accepts `config_path` for a custom YAML file. Configured paths resolve relative to that file. Output directories are created automatically.

1. Select outfits matching dress + footwear + earrings, or top + bottom + footwear + earrings, plus any subset of five optional classes.
2. Create a table with one row per outfit and ten product slots.
3. Resize images to 128 × 96 and concatenate horizontally. Missing products become white slots. Write full and preview TFRecords.
4. Train the 256-dimensional model and export weights, embeddings, loss history, and previews.
5. Generate outfits (see below).

Slot order: Tops, Bottoms, Outerwear, Dresses/jumpsuits/complete sets, Footwear, Bags, Glasses, Earrings, Ring, Necklace. Model input is 128 × 960 × 3. Channel-stacked images are unsupported.

`train_sample_indices` are zero-based positions in the generated outfit table; update them for your data. These are training previews, not held-out evaluation. `batch_size` now controls actual batching; reduce it if memory is limited. Default duration is 400 epochs.

## Outfit generation

`scripts/generation/generate_outfits.py` (settings under `generation:` in `config.yml`):

1. Draw `num_candidates` random outfits. The slot pattern follows the frequency of patterns among training outfits; products come from `candidate_pool` (`unseen` = never in a training outfit, as in the report).
2. Encode each candidate to its posterior mean, using the same strip construction as training (including the JPEG round trip of the TFRecords).
3. Score it by the Euclidean distance to the nearest training embedding.
4. Write `generated_outfits.csv` (slots, `distance`, `distance_percentile`, `nearest_training_row`) and `candidate_embeddings.npy`.

To recommend for a product, `src.generation.retrieval.rank_for_product(generated, product_id, k)` returns the closest candidates containing it. Distance to a known outfit is a hypothesis for compatibility, not a validated score. Encoding 100,000 candidates took about 30 minutes on a Colab T4 (per the report).

## Notebooks

Both notebooks only call code from `src/` and `scripts/` and are committed without outputs. Set `FASHION_CONFIG` to use a different config file.

- `notebooks/training.ipynb`: outfit strips, model, optional short demo training (writes to `models/demo/`), loss curve, reconstructions, latent-space projection.
- `notebooks/outfit-generation.ipynb`: candidates, encoding, distances, closest/farthest candidates beside their nearest training outfit, and recommendations for a product.

## Objective and outputs

The encoder produces 256 means and 256 log-variances, samples a latent vector, and decodes image logits. **The preserved loss is mean pixel binary cross entropy only, not ELBO: there is no KL term.** Random prior samples have no demonstrated guarantee of useful outfits. The final encoder ReLU is preserved, including for means/log-variances, to avoid silently redesigning the model.

Every 20 epochs and at the final epoch, training saves:

- `models/vae_256.weights.h5`: Keras weights.
- `models/vae_data_256.pkl`: `latent_space` (N × 256 NumPy array), `reconstruction_loss` (epoch means), `embedding_type` (`posterior_mean`). Rows match `outfit_w_products.csv`. Only load trusted pickle files.
- `figures/epoch_images_256/`: original and reconstructed training previews.

The historical `ELBO_list` is replaced by accurately named positive reconstruction losses. Embeddings now use deterministic posterior means instead of sampled tensors. Update consumers of the old pickle schema.

## Changes affecting historical results

- Every optional subset is included: **64 configurations**, replacing the old incomplete set of 42. Valid nine-item outfits are no longer dropped by a fixed size loop.
- Normalized images are no longer divided by 255 again before encoding. TFRecord decoding returns floats in [0, 1].
- Missing slots use `pd.isna`, replacing identity comparisons with `np.nan`.
- Training honors batch size instead of silently using one image per update.
- Sampling has a defined latent dimension, model/training code is shared, and plotting imports and output directories are fixed.
- Generated bytecode and notebook checkpoints are removed from version control.

**Rebuild both TFRecords and retrain.** Old records may have lost information through quantization; normalizing them at read time cannot repair it. Report figures and notebook outputs are historical, not validated results of this refactor. No full-data accuracy or compatibility improvement is claimed.
