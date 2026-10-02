from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from scripts import pipeline
from src.utils.setup_utilities import load_config


@pytest.fixture
def project(tmp_path):
    """Minimal dataset: one training outfit plus unseen products for generation."""
    data = tmp_path / "data"
    (data / "tabular").mkdir(parents=True)
    (data / "images").mkdir()
    image = Image.new("RGB", (16, 16))
    classes = [
        ("Dresses, jumpsuits and Complete set", "Dresses", "Dress"),
        ("Accesories, Swim and Intimate", "Footwear", "Boots"),
        ("Accesories, Swim and Intimate", "Jewellery", "Earrings"),
    ]
    rows = []
    for index, (category, family, kind) in enumerate(classes * 2):
        code = f"{index:03d}"
        image.save(data / "images" / f"{code}.jpg")
        rows.append((code, category, family, kind, f"datathon/images/{code}.jpg"))
    pd.DataFrame(
        rows,
        columns=[
            "cod_modelo_color",
            "des_product_category",
            "des_product_family",
            "des_product_type",
            "des_filename",
        ],
    ).to_csv(data / "tabular" / "product_data.csv", index=False)
    pd.DataFrame(
        {"cod_outfit": [1, 1, 1], "cod_modelo_color": ["000", "001", "002"]}
    ).to_csv(data / "tabular" / "outfit_data.csv", index=False)
    config = yaml.safe_load(Path("config.yml").read_text())
    config["training"].update(
        train_sample_indices=[0], epochs=1, batch_size=1, checkpoint_every=1
    )
    config["generation"].update(num_candidates=3, encode_batch_size=2)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_default_stages_and_groups():
    names = [step.name for step in pipeline.select_steps(pipeline.DEFAULT_GROUPS)]
    assert names == ["valid-outfits", "outfit-table", "image-arrays", "train"]
    assert [s.name for s in pipeline.select_steps(["generate"])] == ["generate"]
    assert len(pipeline.select_steps(["all"])) == len(pipeline.STEPS)


def test_missing_inputs_are_reported_before_anything_runs(project, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "LOG_FILE", str(tmp_path / "pipeline.log"))
    config = load_config(project)
    Path(config["data"]["outfits_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="outfit_data.csv"):
        pipeline.run_pipeline(project)
    assert not Path(config["data"]["valid_outfits_path"]).exists()
    assert pipeline.main(["--config", str(project)]) == 1
    assert pipeline.main(["--config", str(tmp_path / "absent.yml")]) == 1


def test_later_stage_inputs_may_come_from_earlier_stages(project):
    config = load_config(project)
    assert pipeline.missing_inputs(pipeline.select_steps(["all"]), config) == []
    assert pipeline.missing_inputs(pipeline.select_steps(["train"]), config) == [
        Path(config["training"]["train_dataset_path"]),
        Path(config["training"]["train_sample_path"]),
    ]


def test_dry_run_changes_nothing(project):
    results = pipeline.run_pipeline(project, ["all"], dry_run=True)
    assert {status for _, status, _ in results} == {"planned"}
    assert not Path(load_config(project)["data"]["valid_outfits_path"]).exists()


def test_full_run_then_resume_skips_up_to_date_steps(project):
    results = pipeline.run_pipeline(project, ["all"])
    assert [status for _, status, _ in results] == ["done"] * 5
    config = load_config(project)
    generated = pd.read_csv(config["generation"]["generated_outfits_path"])
    assert len(generated) == 3 and (generated["distance"] >= 0).all()
    resumed = pipeline.run_pipeline(project, ["all"], resume=True)
    assert {status for _, status, _ in resumed} == {"up to date"}
    Path(config["data"]["outfits_path"]).touch()
    plan = pipeline.run_pipeline(project, ["preprocess"], resume=True, dry_run=True)
    assert plan[0][:2] == ("valid-outfits", "planned")


def test_cli_overrides_reach_the_config():
    args = pipeline.build_parser().parse_args(["train", "--epochs", "2"])
    assert pipeline._overrides(args) == {"training": {"epochs": 2}}
    args = pipeline.build_parser().parse_args(["--device", "cpu"])
    assert pipeline._overrides(args) == {
        "training": {"device": "cpu"},
        "generation": {"device": "cpu"},
    }
    with pytest.raises(SystemExit):
        pipeline.build_parser().parse_args(["--device", "tpu"])
    assert pipeline._overrides(pipeline.build_parser().parse_args([])) == {}


def test_no_arguments_selects_default_groups():
    assert pipeline.build_parser().parse_args([]).groups == []
    assert pipeline.build_parser().parse_args(["train", "generate"]).groups == [
        "train",
        "generate",
    ]
    with pytest.raises(SystemExit):
        pipeline.build_parser().parse_args(["bogus"])
