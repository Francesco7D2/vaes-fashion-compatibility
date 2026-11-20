# Hypothesis

We test whether frozen FashionCLIP product embeddings improve fill-in-the-blank accuracy over colour coherence, particularly on unseen product pairs.
With the ultimate goal of this repo is predicting / generating and 
unseen outfit that matches the style of the database. Then, we try 
test several methodologies to allow reach the goal, and we test 
those methodologies using fill in the black. We previsouly trained
some embeddings of the outfits from a VAE, and didn't reach significant improvement compared to co ocurrence fill in the black. For this reason, should be appropiate trying with the embeddings of a pre trained model for generatig embeddings in a context of fashion. 

# Method

For that, we will use the following model Checkpoint: patrickjohncyh/fashion-clip frozen and in evaluation mode. Then as in VAE procedure, we use the product images, converted to RGB and obtain the embedding of each product from the loaded checkpoint with (use_fast = False) and Each of the 2,976 distinct products encoded once, in batches of 32. Then we normalize those embeddings to unit length, we proceed by, for each question, we score each candidate by its mean cosine similarity to the remaining outfit items. We rank the four candidates by this score, with higher values indicating greater average similarity. We rank the candidates and evaluate on held-out questions. We kept this scoring rule fixed throughout the experiment.


# Evaluation protocol 



To be more concrete about on how we evaluated the embeddings, we start with the same split used for the other methodolgies. Starting with the existing split with 2,850 training outfits and 314 held-out outfits and held out share 0.1 approximately 10% held out, stratified by occupied-slot pattern, using seed 42.


1,616 questions, each with one correct product and three sampled alternatives.
Alternatives follow the existing same-class, same-catalogue rules.
Ties use the existing np.isclose rule and shared credit.

711 unseen-pair questions across 285 outfits: the correct product never co-occurred with any remaining item in a training outfit. 

This subset can include individually unseen products.

Difference intervals use 1,000 paired bootstrap resamples of whole outfits, seed 42.

Both reported subsets contain questions from held-out outfits; unseen pairs are a subset of those questions.


“Unseen” refers only to co-occurrence in this repository’s training split. Overlap with FashionCLIP’s pretraining data was not assessed. These results describe one fixed split and candidate set; the bootstrap measures uncertainty from resampling held-out outfits, not from changing the split, candidates, or pretrained checkpoint.

# Results

| Subset | FashionCLIP | Colour | Difference | Paired 95% interval |
|---|---:|---:|---:|---|
| All | 39.05% | 33.37% | +5.68 pp | [+2.71, +8.63] pp |
| Unseen pairs | 34.18% | 32.16% | +2.02 pp | [−1.70, +6.27] pp |


Colour coherence scores each completed outfit by the fraction of its products belonging to its most common metadata colour group.

Results are saved in fashion_clip_questions.csv (per-question credits) and fashion_clip_summary.csv (aggregate results and paired confidence intervals). Differences are FashionCLIP minus colour.

# Interpretation and limits

The overall result supports an improvement over colour under this protocol. The unseen-pair result is inconclusive because its interval includes zero; this does not establish equivalence.
This tests choosing among four candidates, not full-catalogue retrieval or generating good outfits from one garment. Visual similarity can reflect colour or photographic style as well as compatibility. 

# Reproduction and saved outputs

The dataset is not included, and redistribution permissions have not been verified. Reproducing this experiment requires access to the original MANGO data and the prepared outfit table. The commands below assume those files and a configured virtual environment are available.

To reproduce the experiment, from the repository folder, run: 

```
.venv/bin/python -m pip install -r requirements.txt "transformers==4.57.6"
.venv/bin/python -m scripts.evaluation.fashion_clip_example
```
