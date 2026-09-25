# Report updates for the restored recommendation pipeline

## Sources and scope

Original refactor base: `vaes-fashion-compatibility` commit `e598727`. Recovered source: [MANGO-fashion-compatibility at c61f869](https://github.com/Francesco7D2/MANGO-fashion-compatibility/tree/c61f8692ca31203ab067fde33949f38abd60661a), especially [outfit_predictions.ipynb](https://github.com/Francesco7D2/MANGO-fashion-compatibility/blob/c61f8692ca31203ab067fde33949f38abd60661a/notebooks/outfit_predictions.ipynb), the embedding notebooks, and the two tabular CSVs.

The second repository resolves the previously missing recommendation implementation. It also contains a **genuine ELBO loss definition**, including reconstruction, log prior, and log posterior terms. The earlier observation that KL is absent applies to the executable training script in the first repository, not to every version of your original work. The available files do not establish which exact checkpoint produced each report figure.

The current executable training baseline still uses reconstruction-only loss; this task restores candidate generation and recommendation without changing that objective. Do not label newly generated loss curves “ELBO” unless you first change and rerun the training objective. Historical report figures should be described according to the actual original experiment; do not retroactively relabel them based solely on the refactored script.

## Measured changes using the recovered CSVs

| Quantity | Old selection logic | Corrected selection / new generation |
| --- | ---: | ---: |
| Input outfits | 7,842 | 7,842 |
| Catalog products | 9,222 | 9,222 |
| Allowed configurations | 42 | 64 |
| Selected training outfits | 2,303 | 3,164 |
| Observed configurations after corrected selection | — | 45 |
| Products in corrected training outfits | — | 4,909 |
| Eligible unseen products in supported classes | — | 3,378 |
| Candidate draws, seed 42 | Report target: 100,000 | 100,000 |
| Distinct candidates in this generation run | Not measured here | 100,000 |
| Eligible products covered by this generation run | Not measured here | 3,378 |

The 2,303 count was reproduced with the old incomplete combinations and size-3-through-8 filter. The 3,164 count uses every allowed subset and no hard-coded size cutoff. There are 861 additional selected outfits. These are measured preprocessing results, not model quality results.

Each of two base configurations permits every subset of five optional classes: `2 × 2^5 = 64`. The old generator omitted multi-item optional subsets without Outerwear. There are ten image slots, but at most **nine occupied products** for the top/bottom base and **eight** for the dress base. The previous ten-product maximum is not supported by these rules. Forty-five configurations actually occur in the corrected selected data; the other nineteen have zero sampling probability.

Updated unseen-product counts (replace the old report's Table 5 for a new run):

| Class | Count |
| --- | ---: |
| Tops | 1,242 |
| Bottoms | 662 |
| Dresses, jumpsuits and Complete set | 605 |
| Outerwear | 360 |
| Bags | 151 |
| Footwear | 139 |
| Earrings | 95 |
| Glasses | 58 |
| Necklace | 54 |
| Ring | 12 |
| Total | 3,378 |

“Unseen” means absent from the **selected training outfits**, not absent from the raw catalog or all original outfits.

## What is preserved and what changed

| Part | Current behavior and report implication |
| --- | --- |
| Candidate strategy | Preserved: 100,000 frequency-weighted random configuration draws; uniform product choice within each unseen class pool |
| Frequency matching | Matches the training distribution in expectation, not exactly in a finite sample; probabilities are recomputed from 3,164 selected outfits |
| Distance | Preserved: minimum Euclidean distance to a training outfit; lower ranks first; no new compatibility labels or metric normalization |
| Serving | Preserved: condition on membership of the input product and return top k; now uses a SQLite product index |
| Embeddings | Changed: deterministic posterior means replace sampled latent vectors for both references and candidates; distances/rankings will change |
| Checkpoint consistency | Both populations are encoded with the same checkpoint; hashes prevent mixing model/data versions |
| Images | Corrected normalization; candidates also use the training JPEG round trip, whereas the old prediction notebook used in-memory image arrays |
| Combinations | 42 becomes 64 allowed; nine-item valid outfits are considered; observed frequencies and unseen pools change |
| Randomness | Explicit seed 42 and sorted product pools; reproducible draws for fixed input/software, not the old unseeded sample sequence |
| Duplicates | Sampling remains with replacement. Identical outfits collapse only during serving; no duplicate draws happened in this 100,000-draw audit |
| Missing class pool | Fail clearly if any observed configuration needs a class with no unseen products; probabilities are never silently redistributed |
| Missing images | Fail before candidate encoding with missing product IDs; no silent removal that changes the evaluated population |
| Efficiency | Batch encoding, bounded image cache, exact blockwise distances, indexed product lookup; new runtime claims require measurement |
| Training baseline | Previous refactor's corrected scaling and actual batching remain; loss is still reconstruction-only, unlike genuine-ELBO notebook variants |

## Suggested replacement methodology text

> After applying the corrected category rules, 3,164 of the 7,842 original outfits were selected. The two base configurations and five optional product classes permit 64 configurations, of which 45 occurred in the selected data. Products occurring in selected training outfits were excluded from the candidate pool, leaving 3,378 products across the supported classes. We generated 100,000 candidate outfits by sampling configurations according to their empirical training frequencies and selecting products uniformly within each required class. Sampling used replacement and seed 42. This generation run contained 100,000 distinct outfits and covered all 3,378 eligible products.

For the implemented ranking procedure, describe it without claiming a full-data run has already occurred:

> The recommendation pipeline encodes reference and candidate outfits with the same checkpoint and uses the encoder posterior mean as the embedding. Each candidate is scored by its minimum Euclidean distance to a reference outfit. At query time, candidates containing the requested product are ordered by ascending distance, and up to k distinct outfits are returned. Distances are exact and computed in bounded blocks; results are stored in a product-indexed database for lookup without model inference.

## Results that still need rerunning

Only 13 referenced product images are available in the recovered repository. The 100,000 real candidate outfits have been generated from metadata, but they have **not** been encoded or ranked with a newly trained full-data model. End-to-end encoding and retrieval were tested with synthetic images and the actual model architecture/checkpoint path.

Do not reuse the old reconstruction curves, distance distribution, qualitative examples, 30-minute/2-hour encoding times, or serving latency as measured results of this changed pipeline. Rebuild TFRecords, retrain, rebuild the index, then replace the figures and timings. Training previews are not held-out tests; reconstruction and nearest-neighbor proximity do not by themselves establish fashion compatibility or unseen-product generalization.
