import argparse
from pathlib import Path

from scripts.run_video import apply_cli_overrides
from src.config import load_config


def test_cli_overrides_model_confidence_and_display():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "default.yaml")

    args = argparse.Namespace(
        model="custom-model.pt",
        conf=0.5,
        no_display=True,
    )

    overridden = apply_cli_overrides(config, args)

    assert overridden.detection.model_path == "custom-model.pt"
    assert overridden.detection.confidence == 0.5
    assert not overridden.output.display