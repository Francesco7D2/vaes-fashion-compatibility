# Hypothesis

We test whether frozen FashionCLIP product embeddings improve fill-in-the-blank accuracy over colour coherence, particularly on unseen product pairs.

The ultimate goal of this repo is predicting or generating an unseen outfit that matches the style of the database. We try several methods towards that goal and test them with the fill-in-the-blank evaluation. We previously trained outfit embeddings with a VAE, which did not beat product co-occurrence on fill-in-the-blank. For this reason, it seems appropriate to try the embeddings of a model pretrained on fashion images.

# Method

We use the checkpoint `patrickjohncyh/fashion-clip`, frozen and in evaluation mode. As in the VAE procedure, we use the product images, converted to RGB, and obtain the embedding of each product from the loaded checkpoint (with `use_fast=False`). Each of the 2,976 distinct products is encoded once, in batches of 32. We normalise the embeddings to unit length. For each question, we score each candidate by its mean cosine similarity to the remaining outfit items, and rank the four candidates by this score, with higher values meaning greater average similarity. We kept this scoring rule fixed throughout the experiment.

# Evaluation protocol

To be more concrete about how we evaluated the embeddings, we start from the same split used for the other methods: 2,850 training outfits and 314 held-out outfits (held-out share 0.1, stratified by occupied-slot pattern, seed 42).

1,616 questions, each with one correct product and three sampled alternatives.
Alternatives follow the existing same-class, same-catalogue rules.
Ties use the existing `np.isclose` rule and shared credit.

711 unseen-pair questions across 285 outfits: the correct product never co-occurred with any remaining item in a training outfit.

This subset can include individually unseen products.

The questions are the same as in the VAE evaluation: both scripts build them with the same functions, configuration and seed, and colour coherence gets identical accuracy in both (33.37% on all questions, 32.16% on unseen pairs).

Difference intervals use 1,000 paired bootstrap resamples of whole outfits, seed 42.

Both reported subsets contain questions from held-out outfits; unseen pairs are a subset of those questions.

“Unseen” refers only to co-occurrence in this repository’s training split. Overlap with FashionCLIP’s pretraining data was not assessed. These results describe one fixed split and candidate set; the bootstrap measures uncertainty from resampling held-out outfits, not from changing the split, candidates, or pretrained checkpoint.

# Results

| Subset | FashionCLIP | Colour | Difference | Paired 95% interval |
|---|---:|---:|---:|---|
| All | 39.05% | 33.37% | +5.68 pp | [+2.71, +8.63] pp |
| Unseen pairs | 34.18% | 32.16% | +2.02 pp | [−1.70, +6.27] pp |

Colour coherence scores each completed outfit by the fraction of its products belonging to its most common metadata colour group.

Comparison with the other scores on the same questions:

| Score | All (1,616) | Unseen pairs (711) |
|---|---:|---:|
| Product co-occurrence | 57.32% | 17.65% |
| VAE latent distance | 44.06% | 32.63% |
| FashionCLIP | 39.05% | 34.18% |
| Colour coherence | 33.37% | 32.16% |
| Random | 24.20% | 22.78% |

The other rows come from `reports/evaluation.csv` (run `vae`). Only the colour comparison has a paired interval: per-question credits for the other scores are not saved, so the differences from the VAE and co-occurrence are not tested here.

Results are saved in `fashion_clip_questions.csv` (per-question credits) and `fashion_clip_summary.csv` (aggregate results and paired confidence intervals). Differences are FashionCLIP minus colour.

# Interpretation and limits

The overall result supports an improvement over colour under this protocol. The unseen-pair result is inconclusive because its interval includes zero; this does not establish equivalence.

FashionCLIP is below the VAE latent distance overall (39.05% against 44.06%) and well below co-occurrence (57.32%). On unseen pairs, FashionCLIP, the VAE and colour are within a few points of each other.

A possible explanation, to test next: mean cosine similarity rewards products that look alike, while an outfit combines products that go together, so similarity is not the same as compatibility. A natural next step is a small model on FashionCLIP features, trained to score which products appear together in training outfits.

This tests choosing among four candidates, not full-catalogue retrieval or generating good outfits from one garment. Visual similarity can reflect colour or photographic style as well as compatibility.

# Reproduction and saved outputs

The dataset is not included, and redistribution permissions have not been verified. Reproducing this experiment requires access to the original MANGO data and the prepared outfit table. The commands below assume those files and a configured virtual environment are available.

To reproduce the experiment, run from the repository folder:

```
.venv/bin/python -m pip install -r requirements.txt "transformers==4.57.6"
.venv/bin/python -m scripts.evaluation.evaluate_fashion_clip
```
