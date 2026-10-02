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
| `src/generation/` | Catalogues, candidate sampling, candidate encoding, nearest-training-embedding distance and ranking |
| `src/evaluation/fitb.py` | Held-out split, fill-in-the-blank questions, accuracy and baseline scores |
| `scripts/data_processing/` | The three preprocessing steps |
| `scripts/training/train_vae.py` | Batching, epochs, checkpoints, embeddings |
| `scripts/generation/generate_outfits.py` | Offline candidate generation and ranking |
| `scripts/generation/recommend.py` | Candidates generated and ranked around one product |
| `scripts/evaluation/evaluate_fitb.py` | Fill-in-the-blank evaluation of a model and the baselines |
| `scripts/pipeline.py` | Command-line pipeline over all steps |
| `scripts/report/make_figures.py` | Figures of the report |
| `notebooks/` | Exploratory analysis, training and generation walkthroughs |
| `reports/report.tex` | Report source; `reports/report.pdf` is the compiled report |
| `tests/` | Synthetic regressions, actual-model smoke test, GPU/CPU gradient parity, evaluation |

## How the model works

Each product image is 128 × 96 RGB. Concatenating ten slots produces a 128 × 960 RGB strip, a 3 × 128 × 960 tensor for the model. Fixed positions allow class-specific learning; absent classes use white images.

Two convolution/pooling blocks reduce it to 8 channels of 32 × 240. Flattening produces 61,440 values. The next dense layer maps those to 512, requiring over 31 million weights. The decoder maps 256 values back to 61,440, adding over 15 million weights. These two layers hold most of the 47.5 million parameters and most of the memory cost.

The final 512 encoder outputs, from a linear layer, split into `mean` and `logvar`, each of length 256. Sampling is `z = mean + exp(0.5 * logvar) * noise`. Upsampling and convolutions reconstruct image logits, which a sigmoid converts into image intensities for previews. Exported embeddings and previews use the posterior mean; training samples `z`.

The loss is the per-pixel negative ELBO: pixel reconstruction plus the KL divergence to a standard normal prior divided by the number of pixels. Reconstruction-only training and a ReLU on the means and log-variances are available as options and compared in the evaluation. White backgrounds and empty slots make up most of each image, so a low reconstruction error alone is weak evidence of useful recommendations.

## Outfit generation

Product images come from two catalogues (2019–2020 and 2021–2024), and no outfit mixes them. Candidate outfits are generated offline within one catalogue: a configuration is drawn with the frequencies of that catalogue's training outfits, then one product per class is drawn from its products that never appear in a training outfit. Each candidate is built and encoded exactly like a training outfit and scored by the Euclidean distance to its nearest training embedding. A recommendation for a product is the list of its lowest-distance candidates, a lookup in the ranked table; `recommend` instead generates and ranks candidates around one product in a few seconds.

## Evaluation and limitations

Products are shared by many outfits, so a held-out split that avoids product overlap leaves almost no test outfits (see the exploratory notebook). The final model is therefore trained on all selected outfits, and its training previews measure fit rather than generalization.

The fill-in-the-blank evaluation holds out outfits instead: for each slot of a held-out outfit, the true product competes with decoys of the same class and catalogue. The distance recognizes the true product clearly more often than chance, but co-occurrence of products in training outfits is much stronger when product pairs repeat, and colour coherence is as accurate on novel pairs.

Across candidates of different sizes, the distance also grows with the number of products and, more strongly, with the darkness of the products, so it favours small and light outfits. The embeddings separate the two catalogues, a difference of photo style rather than compatibility. The report quantifies these effects and lists the next steps.
