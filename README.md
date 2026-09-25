# Fashion compatibility with outfit embeddings

A research prototype following the [UPC Datathon project](https://github.com/Francesco7D2/datathon-2023-fashion-compatibility). It learns to reconstruct complete outfits from a compressed representation of product images.

**Current scope:** preprocessing, outfit-image TFRecords, a convolutional stochastic autoencoder, embedding export, frequency-weighted candidate generation, exact nearest-neighbor scoring, and product-conditioned top-k recommendations. The missing stage was recovered from [the second repository's prediction notebook](https://github.com/Francesco7D2/MANGO-fashion-compatibility/blob/c61f8692ca31203ab067fde33949f38abd60661a/notebooks/outfit_predictions.ipynb). See [recommendation instructions](docs/recommendation.md) and [report updates with measured counts](docs/report-update.md).

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

The full data/images and trained weights are not committed in this repository. The additional repository supplies the CSVs but only 13 referenced images; recovered files are in the local ignored data folder. For a complete run, supply:

- `data/tabular/outfit_data.csv`: `cod_outfit`, `cod_modelo_color`.
- `data/tabular/product_data.csv`: `cod_modelo_color`, `des_product_category`, `des_product_family`, `des_product_type`, `des_filename`.
- Product image files referenced by `des_filename`.

Product metadata must contain one row per product ID. IDs are strings to retain leading zeros. The source spelling `Accesories, Swim and Intimate` is intentional. Image paths starting with `datathon/` map to `data/`; other relative image paths resolve from the parent of the data directory. Absolute image paths are accepted.

## Run

Edit `config.yml`, then run from the repository root:

```bash
python -m scripts.data_processing.build_valid_outfits_df
python -m scripts.data_processing.build_outfit_w_products_df
python -m scripts.data_processing.build_image_arrays
python -m scripts.training.train_vae
python -m scripts.recommendation.generate_candidates
python -m scripts.recommendation.build_index
python -m scripts.recommendation.recommend --product-id YOUR_PRODUCT_ID --top-k 3
```

Use module invocation (`python -m`); direct script-file invocation is no longer supported. Each pipeline function also accepts `config_path` for a custom YAML file. Configured paths resolve relative to that file. Output directories are created automatically.

1. Select outfits matching dress + footwear + earrings, or top + bottom + footwear + earrings, plus any subset of five optional classes.
2. Create a table with one row per outfit and ten product slots.
3. Resize images to 128 × 96 and concatenate horizontally. Missing products become white slots. Write full and preview TFRecords.
4. Train the 256-dimensional model and export weights, embeddings, loss history, previews, and checkpoint provenance.
5. Sample unseen-product candidates using observed configuration frequencies, encode and score them, then query the saved product index.

Slot order: Tops, Bottoms, Outerwear, Dresses/jumpsuits/complete sets, Footwear, Bags, Glasses, Earrings, Ring, Necklace. Model input is 128 × 960 × 3. Channel-stacked images are unsupported.

`train_sample_indices` are zero-based positions in the generated outfit table; update them for your data. These are training previews, not held-out evaluation. `batch_size` now controls actual batching; reduce it if memory is limited. Default duration is 400 epochs.

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

**Rebuild both TFRecords and retrain.** Old records may have lost information through quantization; normalizing them at read time cannot repair it. Report figures and notebook outputs are historical, not validated results of this refactor. The recovered CSVs confirm 3,164 corrected selected outfits versus 2,303 under the old rules. No full-data accuracy or compatibility improvement is claimed.

The second repository includes ELBO-based notebook variants. The absence of KL described above applies to this executable baseline; it does not establish that your historical report used the same loss. See the report-update document before changing historical claims.
