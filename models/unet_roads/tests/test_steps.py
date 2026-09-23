"""Step tests for unet-roads.

Each test runs the real @step entrypoint against toy OAM chips + road-line labels.
Telemetry sinks (zenml/mlflow) are no-ops via models/conftest.py.
"""

from pathlib import Path
from typing import Any

import pytest

RESNET34_IMAGENET_URL = "https://download.pytorch.org/models/resnet34-b627a593.pth"


@pytest.fixture(scope="session")
def pretrained_weights() -> str:
    return RESNET34_IMAGENET_URL


def test_split_dataset(toy_chips: Path, toy_labels: Path, base_hyperparameters: dict[str, Any]) -> None:
    from models.unet_roads.pipeline import split_dataset

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
    from models.unet_roads.pipeline import split_dataset, train_model

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
    from models.unet_roads.pipeline import evaluate_model, split_dataset, train_model

    # The toy pattern reliably converges to a near-perfect IoU by ~epoch 10-15
    # (verified manually); 20 epochs gives comfortable margin against init variance.
    hp = {**base_hyperparameters, "val_ratio": 0.34, "block_size": 1, "epochs": 20, "batch_size": 4}
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
    assert set(metrics) == {"iou_road", "precision_road", "recall_road"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
    assert metrics["iou_road"] > 0.7


def test_export_onnx(base_hyperparameters: dict[str, Any]) -> None:
    import numpy as np
    import onnx
    from onnxruntime import InferenceSession

    from models.unet_roads.pipeline import _build_model, export_onnx

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


def test_mask_to_lines_produces_linestrings() -> None:
    """A horizontal strip mask should skeletonise + vectorise to a roughly horizontal line."""
    import numpy as np
    from rasterio.transform import from_bounds

    from models.unet_roads.pipeline import _mask_to_lines

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30:34, 5:59] = 1
    transform = from_bounds(85.5, 27.6, 85.501, 27.601, 64, 64)

    lines = _mask_to_lines(mask, transform)
    assert len(lines) >= 1
    total_length = sum(line.length for line in lines)
    assert total_length > 0
