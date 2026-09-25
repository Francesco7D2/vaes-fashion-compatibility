# Candidate generation and recommendations

The implementation follows [the recovered prediction notebook](https://github.com/Francesco7D2/MANGO-fashion-compatibility/blob/c61f8692ca31203ab067fde33949f38abd60661a/notebooks/outfit_predictions.ipynb), with the explicit changes listed in [report-update.md](report-update.md).

## Method

1. Count configurations in the actual training outfit table. An unobserved configuration has probability zero.
2. Exclude every product appearing in that table from the candidate catalog.
3. Draw 100,000 configurations with replacement using their empirical frequencies. For each required class, choose a product uniformly from its unseen pool. The expected proportions match training; finite-sample counts need not match exactly.
4. Encode candidates and training outfits with the **same saved model checkpoint**, using the posterior mean. Candidate images use the ten-slot layout, white missing slots, and the same JPEG round trip and normalization as training records.
5. Score each candidate as the minimum Euclidean distance to any training outfit embedding. Lower is better. Compute exact distances in bounded blocks; no approximate search, latent normalization, or outfit-size adjustment is applied.
6. Store the scores and product membership in SQLite. At query time, select the top distinct candidate outfits containing the requested product. No image processing or model inference happens during lookup.

This is the report's compatibility heuristic, not a probability or validated compatibility classifier. Retrieving a low-distance outfit does not prove it is fashionable.

## Run locally

From the repository root, activate the environment:

```bash
source .venv/bin/activate
```

The complete workflow, after restoring the original product images, is:

```bash
python -m scripts.data_processing.build_valid_outfits_df
python -m scripts.data_processing.build_outfit_w_products_df
python -m scripts.data_processing.build_image_arrays
python -m scripts.training.train_vae
python -m scripts.recommendation.generate_candidates
python -m scripts.recommendation.build_index
python -m scripts.recommendation.recommend --product-id YOUR_PRODUCT_ID --top-k 3
```

Replace `YOUR_PRODUCT_ID` with an exact ID from the generated candidate CSV; preserve leading zeros. Every recommendation command accepts `--config /absolute/path/config.yml`. Candidate generation only needs the CSVs and may run before training; encoding/indexing requires images, rebuilt training TFRecords, and a checkpoint from the updated training script.

For a smaller first run, reduce `recommendation.num_candidates` in `config.yml`. Lower `encoding_batch_size` if memory is limited. Restore 100000 candidates for the report-method run. Changing generation settings requires rerunning generation and indexing.

The default policy intentionally recommends only products unseen in the selected training outfits. A training product, unsupported category, or product missed by finite random sampling returns an empty result with an explanation. It does not fall back to arbitrary outfits. Fewer than k matches returns the available matches.

## Output files

| File under `data/processed/` | Meaning |
| --- | --- |
| `candidates.csv` | One row per random draw, candidate ID, configuration ID, ten product IDs, exact-outfit signature |
| `configuration_frequencies.csv` | All configurations, class lists, training counts, probabilities, sampled counts |
| `candidate_generation.json` | Seed, sampling method, counts, configuration rules, input/output SHA-256 hashes |
| `recommendation_embeddings.npz` | Candidate/reference IDs, posterior means, distances, nearest reference row indices; load with `allow_pickle=False` |
| `recommendations.sqlite` | Self-contained serving index, product memberships, nearest training outfit IDs, provenance |
| `recommendation_index.json` | Checkpoint/data hashes, image hashes, preprocessing, metric, and coverage |

Training also writes `models/vae_256.weights.h5.json`, binding weights to the training records and outfit-ID table. The index builder rejects changes to candidates, catalog, rules, training records, outfit table, or checkpoint instead of mixing stale artifacts. It checks candidate image availability before loading the model. Missing provenance requires a new training run; old sampled-embedding pickle files are not imported as deterministic reference embeddings.

Identical candidate draws are retained in the generation table to preserve sampling statistics. At serving time they collapse to the first identical outfit, so duplicate suggestions do not fill the top k. Equal distances sort by candidate ID; nearest-reference ties select the earliest reference row.

Image memory is bounded by the encoding batch and LRU product-image cache. Distance scratch memory is bounded by both distance block settings. Embeddings and candidate tables remain in memory (roughly 100 MB for 100,000 float32 vectors of length 256, plus tables and model); this is not a disk-streaming billion-item index. Keep the serving SQLite file; serving does not require the NPZ, training model, or images. The saved index remains an intentional snapshot until rebuilt.

## Data recovered in this session

The additional repository contains both original CSVs and only 13 of 9,222 referenced product images. The remaining JPGs and trained weights were not found there. The local ignored `data/` folder includes the recovered CSVs and available sample images. The candidate CSV and frequencies have been generated from the recovered real catalog. Complete image encoding and real-data ranking still require the missing images and a new checkpoint.

The automated tests cover frequency sampling, unseen-product exclusion, reproducibility, empty pools, brute-force distance equivalence and ties, duplicate suppression, product lookup, leading-zero IDs, stale-file rejection, missing images, and a synthetic CSV → training → saved checkpoint → candidate encoding → nearest-neighbor → recommendation run.
