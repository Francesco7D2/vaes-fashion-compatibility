"""Fill-in-the-blank (FITB) evaluation of outfit scores on held-out outfits.

For every occupied slot of a held-out outfit, the true product competes with
decoys: products of the same class and catalogue that appear in training
outfits. A score answers a question correctly when the true product gets the
best value.
"""

import numpy as np
import pandas as pd

from src.data_processing.schema import PRODUCT_CLASSES


def slot_patterns(outfits):
    """Occupied-slot pattern of each outfit as an integer bit mask."""
    return outfits[list(PRODUCT_CLASSES)].notna().to_numpy() @ (1 << np.arange(len(PRODUCT_CLASSES)))


def split_outfits(outfits, held_out_share=0.1, seed=42):
    """Row positions (training, held out), stratified by occupied-slot pattern."""
    rng = np.random.default_rng(seed)
    patterns = slot_patterns(outfits)
    held_out = []
    for pattern in np.unique(patterns):
        rows = rng.permutation(np.flatnonzero(patterns == pattern))
        count = rng.binomial(1, held_out_share) if len(rows) == 1 else round(held_out_share * len(rows))
        held_out.extend(rows[:count])
    held_out = np.sort(np.array(held_out, dtype=int))
    return np.setdiff1d(np.arange(len(outfits)), held_out), held_out


def fitb_questions(outfits, train_rows, test_rows, catalogue, options=4, seed=42):
    """One question per occupied slot of each held-out outfit, as a long table.

    Columns: `question`, `outfit_row`, `slot`, `option`, `is_true`, then one
    column per slot holding the option outfit. Slots without enough decoys are
    skipped.
    """
    rng = np.random.default_rng(seed)
    slots = list(PRODUCT_CLASSES)
    seen = {}
    for slot in slots:
        products = outfits.iloc[train_rows][slot].dropna().unique()
        for product in products:
            seen.setdefault((slot, catalogue.get(product, "unknown")), []).append(product)
    rows = []
    for row in test_rows:
        outfit = outfits.iloc[row][slots]
        for slot in slots:
            true = outfit[slot]
            if pd.isna(true):
                continue
            pool = [p for p in seen.get((slot, catalogue.get(true, "unknown")), []) if p != true]
            if len(pool) < options - 1:
                continue
            decoys = rng.choice(pool, size=options - 1, replace=False)
            question = len(rows) // options
            for option, product in enumerate([true, *decoys]):
                rows.append(
                    {
                        "question": question,
                        "outfit_row": row,
                        "slot": slot,
                        "option": option,
                        "is_true": option == 0,
                        **outfit.to_dict(),
                        slot: product,
                    }
                )
    return pd.DataFrame(rows)


def fitb_accuracy(questions, scores, higher_is_better=False, resamples=1000, seed=42):
    """Share of questions answered correctly, with a 95 % bootstrap interval.

    Ties share the credit evenly. The interval resamples held-out outfits, since
    questions from one outfit are not independent.
    """
    frame = questions[["question", "outfit_row", "is_true"]].assign(
        score=np.asarray(scores, dtype=float) * (-1 if higher_is_better else 1)
    )
    best = frame.groupby("question")["score"].transform("min")
    winners = frame[np.isclose(frame["score"], best)]
    credit = winners.groupby("question").agg(
        credit=("is_true", "mean"), outfit_row=("outfit_row", "first")
    )
    per_outfit = credit.groupby("outfit_row")["credit"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_outfit), size=(resamples, len(per_outfit)))
    sums, counts = per_outfit["sum"].to_numpy(), per_outfit["count"].to_numpy()
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return credit["credit"].mean(), *np.percentile(boot, [2.5, 97.5])


def cooccurrence_scores(questions, outfits, train_rows):
    """Training outfits that contain the option product together with each other product."""
    slots = list(PRODUCT_CLASSES)
    outfits_of = {}
    for row in train_rows:
        for product in outfits.iloc[row][slots].dropna():
            outfits_of.setdefault(product, set()).add(row)
    scores = []
    for record in questions.to_dict("records"):
        mine = outfits_of.get(record[record["slot"]], set())
        others = [record[s] for s in slots if s != record["slot"] and isinstance(record[s], str)]
        scores.append(sum(len(mine & outfits_of.get(other, set())) for other in others))
    return np.array(scores)


def colour_scores(questions, colour):
    """Share of the outfit's products in its most common colour group."""
    slots = list(PRODUCT_CLASSES)
    groups = questions[slots].apply(lambda column: column.map(colour))
    return groups.apply(lambda row: row.value_counts(normalize=True).iloc[0], axis=1).to_numpy()
