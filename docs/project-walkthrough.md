# Project overview

## Approach

The goal is to recommend a complete outfit given one product. Instead of learning a compatibility score between pairs of products, the project represents an entire outfit as one image and learns a compressed representation of it.

```text
Outfit membership + product metadata + images
    -> rule-based outfit selection
    -> ten fixed product slots, missing slots white
    -> concatenated 128 × 960 RGB image
    -> encoder -> 256-dimensional representation -> decoder
    -> reconstruction training and exported embeddings
    -> random candidate outfits ranked by distance to the training embeddings
```

The selection rules cover a narrower set than all possible outfits. Footwear and earrings are mandatory; dresses and top/bottom outfits are the two alternative bases. Five optional classes (outerwear, bags, glasses, ring, necklace) can be added, each at most once, which gives 2 × 2^5 = 64 configurations. There are ten slots but at most nine occupied ones: the top/bottom base has four products plus five optional classes, and a dress-based outfit has at most eight.

## Code map

| File | Responsibility |
| --- | --- |
| `config.yml` | Paths, rules, image sizes, training and generation settings |
| `src/data_processing/schema.py` | Slot order |
| `src/data_processing/product_processing.py` | Product class from category, family and type; image paths |
| `src/data_processing/group_manipulation.py` | Exploratory counts, configurations, outfit selection |
| `src/data_processing/image_processing.py` | Image loading and resizing, outfit strips, JPEG strip archives |
| `src/models/vae256.py` | Encoder, sampling, decoder (PyTorch) |
| `src/utils/torch_utilities.py` | Device choice (CUDA, Apple GPU, CPU), image/tensor layout |
| `src/utils/train_utilities.py` | Loss, optimizer, training step, previews |
| `src/utils/visualization.py` | Plots for the notebooks and the report |
| `src/generation/` | Candidate sampling, candidate encoding, nearest-training-embedding distance and ranking |
| `scripts/data_processing/` | The three preprocessing steps |
| `scripts/training/train_vae.py` | Batching, epochs, checkpoints, embeddings |
| `scripts/generation/generate_outfits.py` | Offline candidate generation and ranking |
| `scripts/pipeline.py` | Command-line pipeline over all steps |
| `scripts/report/make_figures.py` | Figures of the report |
| `notebooks/` | Exploratory analysis, training and generation walkthroughs |
| `reports/report.tex` | Report source; `reports/report.pdf` is the compiled report |
| `tests/` | Synthetic regressions, actual-model smoke test, GPU/CPU gradient parity |

## How the model works

Each product image is 128 × 96 RGB. Concatenating ten slots produces a 128 × 960 RGB strip, a 3 × 128 × 960 tensor for the model. Fixed positions allow class-specific learning; absent classes use white images.

Two convolution/pooling blocks reduce it to 8 channels of 32 × 240. Flattening produces 61,440 values. The next dense layer maps those to 512, requiring over 31 million weights. The decoder maps 256 values back to 61,440, adding over 15 million weights. These two layers hold most of the 47.5 million parameters and most of the memory cost.

The final 512 encoder outputs split into `mean` and `logvar`, each of length 256. Sampling is `z = mean + exp(0.5 * logvar) * noise`. Upsampling and convolutions reconstruct image logits, which a sigmoid converts into image intensities for previews. Exported embeddings and previews use the posterior mean; training samples `z`.

The loss rewards pixel reconstruction only; there is no KL term, so the model is a stochastic autoencoder rather than a full VAE. The final encoder ReLU applies to the means and log-variances, which keeps the log-variance non-negative. White backgrounds and empty slots make up most of each image, so a low reconstruction error alone is weak evidence of useful recommendations.

## Outfit generation

Candidate outfits are generated offline: a configuration is drawn with the frequencies of the training outfits, then one product per class is drawn from the products that never appear in a training outfit. Each candidate is built and encoded exactly like a training outfit and scored by the Euclidean distance to its nearest training embedding. A recommendation for a product is the list of its lowest-distance candidates, a lookup in the ranked table.

## Evaluation and limitations

Products are shared by many outfits, so a held-out split that avoids product overlap leaves almost no test outfits (see the exploratory notebook). The model is therefore trained on all selected outfits, and the training previews measure fit rather than generalization. Recommendations are evaluated qualitatively.

The distance to the nearest training outfit is a hypothesis for compatibility, not a validated score. It grows with the number of products in a candidate and, more strongly, with the share of the image that products cover, so it favours small and light outfits. The report quantifies this effect and lists the next steps.
