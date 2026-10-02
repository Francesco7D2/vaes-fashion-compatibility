import numpy as np
import pandas as pd
import pytest

from src.data_processing.schema import PRODUCT_CLASSES
from src.evaluation.fitb import (
    colour_scores,
    cooccurrence_scores,
    fitb_accuracy,
    fitb_questions,
    split_outfits,
)
from src.generation.candidates import (
    catalogues,
    sample_candidates_by_catalogue,
    sample_candidates_for_product,
)

DRESS = "Dresses, jumpsuits and Complete set"


def wide(rows):
    frame = pd.DataFrame([{slot: row.get(slot) for slot in PRODUCT_CLASSES} for row in rows])
    return frame.astype(object).where(frame.notna(), np.nan)


@pytest.fixture
def catalog():
    # Two catalogues: images from 2019/2020 and from 2023; one product without a year.
    rows = []
    for year, prefix in ((2019, "a"), (2020, "b"), (2023, "c")):
        for slot in (DRESS, "Footwear", "Earrings"):
            for index in range(3):
                rows.append((f"{prefix}{slot[0]}{index}", slot, f"{year}_{prefix}{index}.jpg"))
    rows.append(("plain", "Footwear", "plain.jpg"))
    return pd.DataFrame(rows, columns=["cod_modelo_color", "des_product_class", "des_filename"])


@pytest.fixture
def outfits():
    # The first outfit mixes 2019 and 2020 products, which joins those years.
    return wide(
        [
            {DRESS: "aD0", "Footwear": "bF0", "Earrings": "aE0"},
            {DRESS: "aD1", "Footwear": "aF1", "Earrings": "bE1"},
            {DRESS: "cD0", "Footwear": "cF0", "Earrings": "cE0"},
        ]
    )


def test_catalogues_join_years_that_share_an_outfit(catalog, outfits):
    catalogue = catalogues(catalog, outfits)
    assert catalogue["aD0"] == catalogue["bF2"] == "2019-2020"
    assert catalogue["cE1"] == "2023-2023"
    assert catalogue["plain"] == "unknown"


def test_candidates_stay_within_one_catalogue(catalog, outfits):
    catalogue = catalogues(catalog, outfits)
    candidates = sample_candidates_by_catalogue(catalog, outfits, "all", 300, seed=0)
    assert len(candidates) == 300 and candidates["candidate_id"].is_unique
    per_candidate = candidates[list(PRODUCT_CLASSES)].apply(
        lambda row: row.dropna().map(catalogue).nunique(), axis=1
    )
    assert (per_candidate == 1).all()
    # Two thirds of the training outfits are in 2019-2020.
    share = (candidates[DRESS].map(catalogue) == "2019-2020").mean()
    assert 0.55 < share < 0.78


def test_candidates_for_a_product(catalog, outfits):
    candidates = sample_candidates_for_product("cF2", catalog, outfits, "all", 50, seed=0)
    assert (candidates["Footwear"] == "cF2").all()
    assert set(candidates[DRESS]) <= {"cD0", "cD1", "cD2"}
    with pytest.raises(ValueError, match="Unknown"):
        sample_candidates_for_product("missing", catalog, outfits, "all", 5)


def test_split_is_disjoint_and_stratified():
    outfits = wide([{DRESS: f"d{i}", "Footwear": f"f{i}"} for i in range(20)])
    outfits = pd.concat([outfits, wide([{"Tops": f"t{i}", "Bottoms": f"b{i}"} for i in range(10)])])
    train, held_out = split_outfits(outfits.reset_index(drop=True), 0.2, seed=1)
    assert set(train) | set(held_out) == set(range(30)) and not set(train) & set(held_out)
    assert sum(held_out < 20) == 4 and sum(held_out >= 20) == 2


def test_fitb_questions_use_seen_decoys_of_the_same_class_and_catalogue():
    outfits = wide(
        [{DRESS: f"d{i}", "Footwear": f"f{i}", "Earrings": "e0"} for i in range(6)]
        + [{DRESS: "x9", "Footwear": "f0", "Earrings": "e1"}]
    )
    catalogue = pd.Series({p: "old" for p in ["d0", "d1", "d2", "d3", "d4", "f0", "f1", "f2", "f3", "f4", "e0", "e1"]})
    catalogue["d5"] = catalogue["f5"] = "new"
    questions = fitb_questions(outfits, np.arange(5), np.array([6]), catalogue, options=3, seed=0)
    assert questions.groupby("question")["is_true"].sum().eq(1).all()
    assert questions.groupby("question").size().eq(3).all()
    dress = questions[questions["slot"] == DRESS]
    decoys = dress.loc[~dress["is_true"], DRESS]
    assert set(decoys) <= {"d0", "d1", "d2", "d3", "d4"}  # seen, same catalogue, not d5
    assert (dress[["Footwear", "Earrings"]] == ["f0", "e1"]).all().all()
    # Earrings: e0 is the only other seen earring, too few decoys for three options.
    assert "Earrings" not in set(questions["slot"])


def test_fitb_accuracy_handles_ties():
    questions = pd.DataFrame(
        {
            "question": np.repeat([0, 1], 4),
            "outfit_row": np.repeat([10, 11], 4),
            "is_true": [True, False, False, False] * 2,
        }
    )
    perfect = np.tile([0.0, 1.0, 2.0, 3.0], 2)
    assert fitb_accuracy(questions, perfect)[0] == 1.0
    assert fitb_accuracy(questions, perfect, higher_is_better=True)[0] == 0.0
    assert fitb_accuracy(questions, np.zeros(8))[0] == pytest.approx(0.25)


def test_baseline_scores():
    outfits = wide(
        [
            {"Tops": "t1", "Bottoms": "b1", "Footwear": "f1"},
            {"Tops": "t1", "Bottoms": "b2", "Footwear": "f1"},
            {"Tops": "t2", "Bottoms": "b2", "Footwear": "f2"},
        ]
    )
    questions = wide([{"Tops": "t1", "Bottoms": "b9", "Footwear": "f1"}, {"Tops": "t1", "Bottoms": "b9", "Footwear": "f2"}])
    questions["slot"] = "Footwear"
    # f1 appears with t1 in two training outfits; f2 never does.
    assert cooccurrence_scores(questions, outfits, np.arange(3)).tolist() == [2, 0]
    colour = pd.Series({"t1": "BLACK", "b9": "BLACK", "f1": "BLACK", "f2": "RED"})
    assert colour_scores(questions, colour).tolist() == pytest.approx([1.0, 2 / 3])
