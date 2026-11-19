from src.utils.setup_utilities import load_config
from scripts.generation.generate_outfits import load_catalog
from src.generation.candidates import catalogues
from src.evaluation.fitb import split_outfits, fitb_questions, fitb_accuracy
from src.data_processing.schema import PRODUCT_CLASSES
from PIL import Image
from transformers import CLIPModel, AutoProcessor
import torch
import torch.nn.functional as F


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

    question =  questions_df[questions_df["question"] == 0].copy()


    # candidates IDs from slot 0 (first row)
    first_row = question.iloc[0]
    missing_slot = first_row["slot"]
    #print(missing_slot)
    candidate_ids = question[missing_slot].tolist()

    print("Candidates:", candidate_ids)

    remaining_items = first_row[list(PRODUCT_CLASSES)]
    remaining_ids= remaining_items.drop(missing_slot).dropna().tolist()
    print("Remaining:",remaining_ids)

    image_paths = catalog.set_index("cod_modelo_color")["des_filename"]

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
    
if __name__ == "__main__":
    main()