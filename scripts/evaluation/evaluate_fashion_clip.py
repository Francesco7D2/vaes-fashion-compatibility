from src.utils.setup_utilities import load_config
from scripts.generation.generate_outfits import load_catalog
from src.generation.candidates import catalogues
from src.evaluation.fitb import split_outfits, fitb_questions, fitb_accuracy
from src.evaluation.fitb import cooccurrence_scores, colour_scores 
from src.data_processing.schema import PRODUCT_CLASSES
from PIL import Image
from transformers import CLIPModel, AutoProcessor
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path



def score_question(question, embedding_by_id):
    """Return one mean-cosine score per candidate, in question-row order."""

    # candidates IDs from slot 0 (first row)
    first_row = question.iloc[0]
    missing_slot = first_row["slot"]
    candidate_ids = question[missing_slot].tolist()


    remaining_items = first_row[list(PRODUCT_CLASSES)]
    remaining_ids = remaining_items.drop(missing_slot).dropna().tolist()

    candidate_embeddings = torch.stack(
        [embedding_by_id[product_id] for product_id in candidate_ids]
    )
    
    remaining_embeddings = torch.stack(
        [embedding_by_id[product_id] for product_id in remaining_ids]
    )

    similarities = candidate_embeddings @ remaining_embeddings.T

    scores = similarities.mean(dim=1).cpu().numpy()

    return scores


def encode_products(product_ids, image_paths, processor, model, batch_size=32):
    """Return a dictionary mapping product IDs to normalized embeddings."""

    embedding_by_id = dict()    

    for start in range(0, len(product_ids), batch_size):
        
        batch_ids = product_ids[start : start + batch_size]
        batch_paths = image_paths.loc[batch_ids]

        batch_images = []
        
        for path in batch_paths:
            with Image.open(path) as image:
                batch_images.append(image.convert("RGB"))


        inputs = processor(images=batch_images, return_tensors="pt")

        with torch.inference_mode():
            embeddings = model.get_image_features(**inputs)


        unit_embeddings = F.normalize(embeddings, p=2, dim=1)
        embedding_by_id.update(zip(batch_ids, unit_embeddings.cpu()))

    return embedding_by_id

def question_credit(questions, scores):
    """Return credit and outfit_row per question; higher scores are better."""

    frame = questions[["question", "outfit_row", "is_true"]].assign(
        score=np.asarray(scores, dtype=float)
    )
   
    best = frame.groupby("question")["score"].transform("max")

    winners = frame[np.isclose(frame["score"], best)]

    return winners.groupby("question").agg(
        credit=("is_true", "mean"),
        outfit_row=("outfit_row", "first"),
    )

def main():

    config = load_config()

    catalog, outfits = load_catalog(config)

    print("Catalog shape:", catalog.shape)

    print("Outfits shape:", outfits.shape)


    rows_train, rows_test = split_outfits(  
        outfits=outfits,
        held_out_share=config["evaluation"]["held_out_share"], 
        seed=config["evaluation"]["seed"]
    )

    product_to_catalogue = catalogues( 
        catalog=catalog, 
        outfits=outfits
    )


    questions_df = fitb_questions(
        outfits=outfits,
        train_rows=rows_train, 
        test_rows=rows_test, 
        catalogue=product_to_catalogue,
        options=config["evaluation"]["options"], 
        seed = config["evaluation"]["seed"]
    )

    all_scores = pd.Series(index=questions_df.index, dtype=float)

    print("Questions:", questions_df["question"].nunique())
    print("Option scores:", len(all_scores))

    products = pd.read_csv(
        config["data"]["products_path"],
        dtype={"cod_modelo_color": str},
    )

    colour = (
        products.drop_duplicates("cod_modelo_color")
        .set_index("cod_modelo_color")["des_agrup_color_eng"]
    )


    product_cells = questions_df[list(PRODUCT_CLASSES)]
    product_values = product_cells.stack()
    evaluation_product_ids = product_values.unique().tolist()

    #print(product_values.head(10))
    print("Distinct products:", len(evaluation_product_ids))


    #question =  questions_df[questions_df["question"] == 0].copy()
    # id == 0
    # question: rows
    
    image_paths = catalog.set_index("cod_modelo_color")["des_filename"]


    model_id = "patrickjohncyh/fashion-clip"

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = CLIPModel.from_pretrained(model_id)

    model.eval()
    model.requires_grad_(False)


    embedding_by_id = encode_products(
        product_ids=evaluation_product_ids,
        image_paths=image_paths, 
        processor=processor, 
        model=model, 
        batch_size=32
    )

    print("Number of embeddings:", len(embedding_by_id))


    for question_id, question_rows in questions_df.groupby(
                                                    "question", sort=False
                                                ):
        

        scores = score_question(question_rows, embedding_by_id)

        all_scores.loc[question_rows.index] = scores


    print("Missing scores:", all_scores.isna().sum())


    accuracy, ci_low, ci_high = fitb_accuracy(
        questions_df,
        all_scores.to_numpy(),
        higher_is_better=True,
    )

    print(f"FashionCLIP accuracy: {accuracy:.2%}")
    print(f"95% confidence interval: [{ci_low:.2%}, {ci_high:.2%}]")


    colour_values = colour_scores(questions_df, colour)

    accuracy, ci_low, ci_high = fitb_accuracy(
        questions_df,
        colour_values,
        higher_is_better=True,
    )

    print(f"Colour baseline accuracy: {accuracy:.2%}")
    print(f"95% confidence interval: [{ci_low:.2%}, {ci_high:.2%}]")



    clip_credit = question_credit(questions_df, all_scores)
    colour_credit = question_credit(questions_df, colour_values)
    
    comparison = clip_credit.rename(columns={"credit": "clip_credit"})
    comparison["colour_credit"] = colour_credit["credit"]
    comparison["difference"] = (
        comparison["clip_credit"] - comparison["colour_credit"]
    )

    print(comparison.head())
    print(f"Mean advantage: {100 * comparison['difference'].mean():.2f} percentage points")

    per_outfit = comparison.groupby("outfit_row")["difference"].agg(
        ["sum", "count"]
    )

    print(per_outfit.head())
    print("Questions represented:", per_outfit["count"].sum())

    reconstructed = per_outfit["sum"].sum() / per_outfit["count"].sum()
    print(f"Reconstructed advantage: {100 * reconstructed:.2f} percentage points")


    sums = per_outfit["sum"].to_numpy()
    counts = per_outfit["count"].to_numpy()
    n_outfits = len(per_outfit)

    rng = np.random.default_rng(42)
    bootstrap_differences = []

    for i in range(1000):
        sampled = rng.integers(0, n_outfits, size=n_outfits)

        difference = sums[sampled].sum() / counts[sampled].sum()

        bootstrap_differences.append(difference)

    low, high = np.percentile(bootstrap_differences, [2.5, 97.5])

    print("Bootstrap repetitions:", len(bootstrap_differences))
    print(
        f"95% paired bootstrap interval: "
        f"[{100 * low:.2f}, {100 * high:.2f}] percentage points"
    )



    cooccurrence = cooccurrence_scores(
        questions_df, outfits, rows_train
    )


    novel_question_ids = questions_df.loc[
        questions_df["is_true"] & (cooccurrence == 0),
        "question",
    ]

    novel_mask = questions_df["question"].isin(novel_question_ids)

    print("Unseen-pair questions:", novel_question_ids.nunique())
    print("Unseen-pair option rows:", novel_mask.sum())

    novel_accuracy, novel_low, novel_high = fitb_accuracy(
        questions_df.loc[novel_mask],
        all_scores.loc[novel_mask].to_numpy(),
        higher_is_better=True,
    )

    print(f"FashionCLIP unseen-pair accuracy: {novel_accuracy:.2%}")
    print(f"95% confidence interval: [{novel_low:.2%}, {novel_high:.2%}]")


    novel_accuracy, novel_low, novel_high = fitb_accuracy(
        questions_df.loc[novel_mask],
        colour_values[novel_mask.to_numpy()],
        higher_is_better=True,
    )

    print(f"Colour baseline unseen-pair accuracy: {novel_accuracy:.2%}")
    print(f"95% confidence interval (colour scores): [{novel_low:.2%}, {novel_high:.2%}]")

    novel_comparison = comparison.loc[
        comparison.index.isin(novel_question_ids)
    ].copy()

    novel_per_outfit = novel_comparison.groupby("outfit_row")["difference"].agg(
        ["sum", "count"]
    )

    print("Unseen-pair questions:", len(novel_comparison))
    print("Questions represented:", novel_per_outfit["count"].sum())

    novel_advantage = (
        novel_per_outfit["sum"].sum() / novel_per_outfit["count"].sum()
    )
    print(f"Unseen-pair advantage: {100 * novel_advantage:.2f} percentage points")

    novel_sums = novel_per_outfit["sum"].to_numpy()
    novel_counts = novel_per_outfit["count"].to_numpy()
    n_novel_outfits = len(novel_per_outfit)

    novel_rng = np.random.default_rng(42)
    novel_bootstrap_differences = []

    for i in range(1000):
        sampled = novel_rng.integers(0, n_novel_outfits, size=n_novel_outfits)

        difference = novel_sums[sampled].sum() / novel_counts[sampled].sum()

        novel_bootstrap_differences.append(difference)

    novel_diff_low, novel_diff_high = np.percentile(
        novel_bootstrap_differences, [2.5, 97.5]
    )

    print("Unseen-pair bootstrap repetitions:", len(novel_bootstrap_differences))
    print(
        f"Unseen-pair 95% paired bootstrap interval: "
        f"[{100 * novel_diff_low:.2f}, {100 * novel_diff_high:.2f}] "
        f"percentage points"
    )





    summary_rows = [
        {
            "subset": "all",
            "questions": len(comparison),
            "outfits": comparison["outfit_row"].nunique(),
            "fashion_clip_accuracy": comparison["clip_credit"].mean(),
            "colour_accuracy": comparison["colour_credit"].mean(),
            "difference": comparison["difference"].mean(),
            "difference_ci_low": low,
            "difference_ci_high": high,
        },
        {
            "subset": "unseen_pairs",
            "questions": len(novel_comparison),
            "outfits": novel_comparison["outfit_row"].nunique(),
            "fashion_clip_accuracy": novel_comparison["clip_credit"].mean(),
            "colour_accuracy": novel_comparison["colour_credit"].mean(),
            "difference": novel_comparison["difference"].mean(),
            "difference_ci_low": novel_diff_low,
            "difference_ci_high": novel_diff_high,
        },
    ]


    
    question_results = comparison.reset_index()


    question_results["is_unseen_pair"] = question_results["question"].isin(
        novel_question_ids
    )

    output_path = (
        Path(config["evaluation"]["results_path"]).parent
        / "fashion_clip_questions.csv"
    )

    question_results.to_csv(output_path, index = False)



    summary = pd.DataFrame(summary_rows)
    summary_path = output_path.with_name("fashion_clip_summary.csv")

    summary.to_csv(summary_path, index=False)

    """
    # candidates IDs from slot 0 (first row)
    first_row = question.iloc[0]
    missing_slot = first_row["slot"]
    #print(missing_slot)
    candidate_ids = question[missing_slot].tolist()

    print("Candidates:", candidate_ids)

    remaining_items = first_row[list(PRODUCT_CLASSES)]
    remaining_ids= remaining_items.drop(missing_slot).dropna().tolist()
    print("Remaining:",remaining_ids)
    all_ids = candidate_ids + remaining_ids
    """


    """
    candidate_paths = image_paths.loc[candidate_ids].tolist()
    remaining_paths = image_paths.loc[remaining_ids].tolist()

    #for product_id, path in zip(candidate_ids, candidate_paths):
    #    print(product_id, "->", path)

    candidate_images = []

    for path in candidate_paths:
        with Image.open(path) as image:
            candidate_images.append(image.convert("RGB"))

    remaining_images = []

    for path in remaining_paths:
        with Image.open(path) as image:
            remaining_images.append(image.convert("RGB"))

    # loading checks
    print("Image counts:", len(candidate_images), len(remaining_images))
    print("First candidate:", candidate_images[0].mode, candidate_images[0].size)

    model_id = "patrickjohncyh/fashion-clip"

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = CLIPModel.from_pretrained(model_id)

    model.eval()
    model.requires_grad_(False)

    print("Training mode:", model.training)
    print("Any trainable parameters:",
        any(parameter.requires_grad for parameter in model.parameters()))

    all_images = candidate_images + remaining_images

    inputs = processor(images=all_images, return_tensors="pt")
    print("Input shape:", inputs["pixel_values"].shape)

    with torch.inference_mode():
        embeddings = model.get_image_features(**inputs)

    print("Embedding shape:", embeddings.shape)
    print("All finite:", torch.isfinite(embeddings).all().item())
    print("Requires gradients:", embeddings.requires_grad)

    # p=2 selects Euclidean length.
    #dim=1 normalizes across the 512 features within each row.
    unit_embeddings = F.normalize(embeddings, p=2, dim=1)

    print("Embedding lengths:", unit_embeddings.norm(dim=1))

    all_ids = candidate_ids + remaining_ids

    embedding_by_id = dict(zip(all_ids, unit_embeddings))

    scores = score_question(question, embedding_by_id)

    n_candidates = len(candidate_ids)
    candidate_embeddings = unit_embeddings[:n_candidates]
    remaining_embeddings = unit_embeddings[n_candidates:]

    similarities = candidate_embeddings @ remaining_embeddings.T

    print("Similarity shape:", similarities.shape)
    print(similarities)
    print("All finite:", torch.isfinite(similarities).all().item())

    print(similarities.mean(dim=1))
    scores = similarities.mean(dim=1).cpu().numpy()

    results = question[["option", "is_true"]].copy()
    results["product_id"] = candidate_ids
    results["score"] = scores

    ranked = results.sort_values("score", ascending=False)
    print(ranked.to_string(index=False))

    top = ranked.iloc[0]
    print("Top product:", top["product_id"])
    print("Is the recorded dress:", bool(top["is_true"]))

    credit, _, _ = fitb_accuracy(
        question,
        scores,
        higher_is_better=True,
    )
    print("Question credit:", credit)
    """
    
if __name__ == "__main__":
    main()