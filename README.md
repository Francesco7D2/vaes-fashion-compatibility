# Fashion compatibility with outfit embeddings

A research prototype following the [UPC Datathon project](https://github.com/Francesco7D2/datathon-2023-fashion-compatibility). It learns to reconstruct complete outfits from a compressed representation of product images.

This version of the code is being refactored with assistance from OpenAI Codex, using GPT-5.6 Sol with medium reasoning effort.

## Setup

Use Python 3.11 on Apple silicon so Apple's Metal plugin is available. Python 3.12 remains supported for CPU execution on all platforms. From the repository root on macOS:

```bash
uv python install 3.11
uv venv --python 3.11 .venv-metal
source .venv-metal/bin/activate
uv pip install -r requirements.txt
python -m pytest -q
```

On Linux, native Windows, or a CPU-only macOS environment, create and activate a normal `.venv` using your supported Python 3.11 or 3.12 interpreter. Windows activation is `.venv\\Scripts\\activate` in Command Prompt or `.venv\\Scripts\\Activate.ps1` in PowerShell.

Check the TensorFlow device before a long run:

```bash
python -m scripts.check_device
```

If `auto` reports CPU when you expect a GPU, run `python -m scripts.check_device --device gpu`. It exits with a clear error rather than silently starting a long CPU training run.

You can enforce GPU use for the long stages as well:

```bash
python -m scripts.run_pipeline --from-stage train --to-stage index --device gpu
```

`training.device` in `config.yml` accepts `auto`, `cpu`, or `gpu`. `auto` selects the first TensorFlow GPU and otherwise uses the CPU. Training and index building print the selected device before doing expensive work.

Training displays the batch progress, speed, and estimated completion time for each epoch. The first batch compiles TensorFlow's training graph, so its initial estimate is not representative; use the completed epoch time for CPU/GPU comparisons. Checkpoint embedding export has its own progress display. Set `training.progress_bar: false` in `config.yml` or pass `--no-progress` to `python -m scripts.training.train_vae` to disable the displays.

On Apple silicon, `requirements.txt` installs Apple's `tensorflow-metal` plugin. On Linux with an NVIDIA GPU—or Windows using WSL2 with NVIDIA support—install the CUDA extras instead:

```bash
python -m pip install -r requirements-accelerated.txt
```

Native Windows uses TensorFlow CPU with this TensorFlow version. GPU training on Windows should use WSL2. The code itself uses `pathlib`, subprocess argument lists, atomic file replacement, and SQLite APIs that work across macOS, Linux, and Windows.

Platform setup references: [Apple tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/) and [TensorFlow pip installation](https://www.tensorflow.org/install/pip).

To open the inspection notebooks locally, install Jupyter once and start it:

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyter lab
```

Tests generate a tiny synthetic dataset and perform one training update using the actual model. No MANGO data or GPU is required. Large dense layers mean even this smoke test needs substantial memory.

## Required data

The local ignored `data/` directory now contains the source CSVs and all 9,222 referenced product images. They are intentionally excluded from Git because of their size. A fresh clone still needs:

- `data/tabular/outfit_data.csv`: `cod_outfit`, `cod_modelo_color`.
- `data/tabular/product_data.csv`: `cod_modelo_color`, `des_product_category`, `des_product_family`, `des_product_type`, `des_filename`.
- Product image files referenced by `des_filename`.

Product metadata must contain one row per product ID. IDs are strings to retain leading zeros. The source spelling `Accesories, Swim and Intimate` is intentional. Image paths starting with `datathon/` map to `data/`; other relative image paths resolve from the parent of the data directory. Absolute image paths are accepted.

## Run

Edit `config.yml`, then run the complete pipeline from the repository root:

```bash
python -m scripts.run_pipeline
python -m scripts.validate_pipeline
```

The full command includes 400 training epochs and recommendation indexing, so it can take hours. To rebuild only the image preprocessing artifacts:

```bash
python -m scripts.run_pipeline --from-stage select --to-stage images
python -m scripts.validate_pipeline --allow-incomplete
```

Resume from a later stage without rerunning earlier work:

```bash
python -m scripts.run_pipeline --from-stage train --to-stage index
```

Use `--dry-run` to print commands without executing them. Every heavy stage runs in a fresh Python process so TensorFlow and image memory are released between stages. Stage status and duration are saved in `reports/pipeline_run.json`.

The equivalent individual commands are:

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

The second repository includes ELBO-based notebook variants. The absence of KL described above applies to this executable baseline; it does not establish that your historical report used the same loss.

## Inspection notebooks

Open these notebooks after activating the project environment:

- `notebooks/01_pipeline_file_check.ipynb`: validate inputs and every expected artifact, then inspect a training image strip.
- `notebooks/02_training_metrics.ipynb`: plot reconstruction loss, epoch duration, embedding diagnostics, and the latest reconstruction preview.
- `notebooks/03_outfit_generation_check.ipynb`: compare configuration frequencies, inspect generated outfits, and view top recommendations for one product.

The metrics notebook intentionally labels the executable baseline's value as training reconstruction loss. It is not a held-out accuracy metric or ELBO.
