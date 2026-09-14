"""Step tests for unet-buildings.

Each test runs the real @step entrypoint against toy OAM chips + GeoJSON labels.
Telemetry sinks (zenml/mlflow) are no-ops via models/conftest.py.
"""

from pathlib import Path
from typing import Any

import pytest

RESNET34_IMAGENET_URL = "https://download.pytorch.org/models/resnet34-b627a593.pth"


@pytest.fixture(scope="session")
def pretrained_weights() -> str:
    """The real checkpoint asset href declared in stac-item.json -- train_model
    downloads it itself via `_download_checkpoint`, same as production."""
    return RESNET34_IMAGENET_URL


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.unet_buildings.pipeline import split_dataset

    hp = {**base_hyperparameters, "val_ratio": 0.34, "block_size": 1}
    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hp,
    )
    assert info["strategy"] == "spatial"
    assert info["train_count"] > 0
    assert info["val_count"] > 0
    assert len(info["train_chip_names"]) == info["train_count"]
    assert len(info["val_chip_names"]) == info["val_count"]
    assert set(info["train_chip_names"]).isdisjoint(info["val_chip_names"])


def test_train_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.unet_buildings.pipeline import split_dataset, train_model

    hp = {**base_hyperparameters, "val_ratio": 0.34, "block_size": 1, "epochs": 1, "batch_size": 2}
    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hp,
    )
    model = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hp,
        split_info=info,
        num_classes=2,
    )
    assert model is not None
    assert hasattr(model, "parameters")
    assert next(model.parameters()).device.type == "cpu"


def test_evaluate_model(
    toy_chips: Path,
    toy_labels: Path,
    base_hyperparameters: dict[str, Any],
    pretrained_weights: str,
) -> None:
    from models.unet_buildings.pipeline import evaluate_model, split_dataset, train_model

    # A handful of epochs on the (easily separable) toy chips exercises real learning,
    # not just that the step runs -- IoU should land well above a random baseline.
    hp = {**base_hyperparameters, "val_ratio": 0.34, "block_size": 1, "epochs": 8, "batch_size": 4}
    info = split_dataset.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hp,
    )
    model = train_model.entrypoint(
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        base_model_weights=pretrained_weights,
        hyperparameters=hp,
        split_info=info,
        num_classes=2,
    )
    metrics = evaluate_model.entrypoint(
        trained_model=model,
        dataset_chips=str(toy_chips),
        dataset_labels=str(toy_labels),
        hyperparameters=hp,
        split_info=info,
        num_classes=2,
    )
    assert set(metrics) == {"iou_building", "precision_building", "recall_building"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
    assert metrics["iou_building"] > 0.5


def test_export_onnx(base_hyperparameters: dict[str, Any]) -> None:
    import numpy as np
    import onnx
    from onnxruntime import InferenceSession

    from models.unet_buildings.pipeline import _build_model, export_onnx

    model = _build_model(num_classes=2)
    onnx_bytes = export_onnx.entrypoint(
        trained_model=model,
        hyperparameters={**base_hyperparameters, "chip_size": 32},
        num_classes=2,
    )
    assert isinstance(onnx_bytes, bytes)
    loaded = onnx.load_from_string(onnx_bytes)
    assert len(loaded.graph.input) == 1
    assert len(loaded.graph.output) == 1

    session = InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    out = np.asarray(session.run(None, {name: np.random.randn(1, 3, 32, 32).astype(np.float32)})[0])
    assert out.shape == (1, 2, 32, 32)
