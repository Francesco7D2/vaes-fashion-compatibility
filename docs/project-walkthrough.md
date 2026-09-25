# Understanding this project

## What you built

The goal was to recommend a complete outfit given one product. You represented an entire outfit as one image and learned a compressed representation, rather than directly learning a compatibility score between each pair of products.

```text
Outfit membership + product metadata + images
    -> rule-based outfit selection
    -> ten fixed product slots, missing slots white
    -> concatenated 128 × 960 RGB image
    -> encoder -> 256-dimensional representation -> decoder
    -> reconstruction training and exported embeddings
    -> frequency-weighted unseen-product candidates
    -> exact nearest-training-embedding scores -> product lookup
```

The rules are narrower than all possible fashion outfits. Footwear and earrings are mandatory; dresses and top/bottom outfits are alternatives. This reflects your original dataset analysis, not a general definition of compatibility.

There are ten available slots but at most nine occupied slots: the top/bottom base has four products plus five optional classes. A dress-based outfit has at most eight. Five independently optional products and two bases give 2 × 2^5 = 64 configurations. The historical count of 42 comes from omitting subsets without Outerwear except singletons.

## Code map

| File | Responsibility |
| --- | --- |
| `notebooks/exploratory-data-analysis.ipynb` | Original category, size, and selection analysis |
| `config.yml` | Paths, rules, image sizes, training settings |
| `src/data_processing/product_processing.py` | Derive product class from category/family/type |
| `src/data_processing/group_manipulation.py` | EDA helpers, subsets, outfit selection |
| `src/data_processing/schema.py` | Canonical slot order |
| `src/data_processing/image_processing.py` | Image loading and TFRecord serialization |
| `scripts/data_processing/` | Three ordered preprocessing entry points |
| `src/models/vae256.py` | Encoder, sampling, decoder |
| `src/utils/train_utilities.py` | Shared loss, gradient update, previews |
| `scripts/training/train_vae.py` | Batching, epochs, checkpoints, embeddings |
| `src/recommendation/candidates.py` | Configuration frequencies and unseen-product sampling |
| `src/recommendation/encoding.py` | Batched checkpoint encoding with an image cache |
| `src/recommendation/ranking.py` | Exact distances and SQLite product lookup |
| `scripts/recommendation/` | Candidate generation, index building, and top-k CLI commands |
| `tests/test_pipeline.py`, `tests/test_recommendation.py` | Synthetic end-to-end test and sampling/retrieval regressions |

The notebook and report remain historical artifacts; saved outputs have not been regenerated. The notebook assumes it starts inside the notebooks directory when changing its working directory.

## How the model works

Each product image is 128 × 96 RGB. Concatenating ten slots produces 128 × 960 × 3. Fixed positions allow category-specific learning; absent categories use white images.

Two convolution/pooling blocks reduce the shape to 32 × 240 × 8. Flattening produces 61,440 values. The next dense layer maps those to 512, requiring over 31 million weights. The decoder maps 256 values back to 61,440, adding over 15 million weights. This explains much of the memory cost.

The final 512 encoder outputs split into `mean` and `logvar`, each length 256. Sampling is `z = mean + exp(0.5 * logvar) * noise`. Upsampling and convolution reconstruct image logits, which sigmoid converts into image intensities for previews.

The loss rewards pixel reconstruction, not human judgments of compatibility. White backgrounds and empty slots can dominate a pixel-average loss, so low reconstruction error alone is weak evidence of useful recommendations.

## Report versus available code

| Report describes | Available implementation |
| --- | --- |
| 128- versus 256-dimensional comparison | Only the 256-dimensional model |
| Maximizing ELBO | Reconstruction cross entropy; no KL/prior regularization |
| 42 configurations, 2,303 selected outfits | Recomputed: 64 allowed configurations, 45 observed, 3,164 selected outfits |
| 100,000 generated candidates | Restored from the second repository; real metadata generation validated |
| Ranking by nearest training embedding | Restored; synthetic end-to-end test passes, full images/checkpoint still needed |
| Qualitative recommendations | Report figures, without executable evaluation |

Your proposed retrieval method moves expensive work offline: generate candidates, encode them, find their distances to training embeddings, then serve highly ranked candidates containing the requested product. This is an experiment worth testing, but distance to a known outfit is a hypothesis for compatibility, not a validated score. The restored pipeline implements this process, but the original figures have not been reproduced with complete images and weights.

The report describes product overlap preventing a useful held-out split, then training on all selected outfits. Preview reconstruction therefore measures training examples. New-product generalization needs a separate evaluation design. A roughly bell-shaped distance histogram also does not establish the central limit theorem as its cause.

## Refactoring choices

Layer dimensions, activations, and the reconstruction-only objective remain unchanged. Adding KL loss, removing the final encoder ReLU, shrinking dense layers, or using pretrained product features are separate experiments. In particular, ReLU restricts log-variance to nonnegative values; this deserves investigation in a proper VAE redesign.

Image scaling, subset generation, NaN handling, and batching fixes intentionally change behavior. Evaluate them with a fresh run. Exported embeddings and previews use posterior means for stable inference; training remains stochastic.

## Next research steps

1. Restore the missing product images. The original CSVs and retrieval notebook have now been recovered. Rebuild records, inspect strips, and run the corrected baseline.
2. Define evaluation for new combinations of known products versus entirely unseen products. Compare simple category/color or feature-neighbor baselines, controlling for outfit size and empty slots.
3. Run a separate VAE experiment with a linear mean/log-variance head and explicitly scaled KL loss. Track reconstruction and KL independently.
4. Run the restored candidate generation and retrieval stages with real images and a new checkpoint. Measure ranking quality and serving latency.

Refactor base: commit `e598727`, dated 2024-03-09. Full-data training and report reproduction require external data and were not performed.

## Additional source recovered

The MANGO-fashion-compatibility repository contains the original prediction workflow and genuine ELBO loss definitions in notebook variants. The no-KL finding above applies to this repository's executable baseline, not every historical experiment. See [the detailed report corrections](report-update.md) for measured dataset changes and [the recommendation guide](recommendation.md) for commands. The corrected metadata pipeline selects 3,164 outfits; the old rules reproduce 2,303. A seed-42 run generated 100,000 distinct candidates covering 3,378 unseen products. Full-image model results remain unverified.
