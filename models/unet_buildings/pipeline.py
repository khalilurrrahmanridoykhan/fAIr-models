"""unet-buildings: a U-Net with an ImageNet-pretrained ResNet-34 encoder for binary
building segmentation, trained from scratch (decoder) or finetuned end to end on
each fAIr project's own OAM chips + OSM building labels.

Unlike `dinov3s_buildings` (frozen foundation-model encoder, decoder-only finetune),
this model updates every layer during training, so it is the right choice when a
project has enough labelled chips to benefit from adapting the encoder itself, and
does not depend on a third-party finetuning package -- the architecture, training
loop and ONNX export below are self-contained standard PyTorch.
"""

import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.utils.data import resolve_directory
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import ONNXMaterializer

MODEL_NAME = "unet-buildings"
CLASS_NAMES = ("background", "building")
CHIP_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Model definition -- a standard U-Net decoder over a torchvision ResNet-34 encoder.
# --------------------------------------------------------------------------- #


def _build_model(num_classes: int = 2) -> Any:
    from torch import nn
    from torchvision.models import resnet34

    class DecoderBlock(nn.Module):
        def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any, skip: Any | None) -> Any:
            import torch

            x = self.up(x)
            if skip is not None:
                if x.shape[-2:] != skip.shape[-2:]:
                    x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
                x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    class UNetResNet34(nn.Module):
        """Encoder: torchvision resnet34 stem + 4 residual stages (skip connections
        taken after the stem and after each of the first three stages). Decoder:
        4 transposed-conv blocks mirroring the encoder, plus a final 4x upsample
        head back to input resolution. Output: per-pixel class logits."""

        def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
            super().__init__()
            backbone = resnet34(weights=None)
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.pool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.pretrained = pretrained

            self.dec4 = DecoderBlock(512, 256, 256)
            self.dec3 = DecoderBlock(256, 128, 128)
            self.dec2 = DecoderBlock(128, 64, 64)
            self.dec1 = DecoderBlock(64, 64, 32)
            self.head = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, num_classes, kernel_size=1),
            )

        def forward(self, x: Any) -> Any:
            s0 = self.stem(x)  # H/2
            p0 = self.pool(s0)  # H/4
            s1 = self.layer1(p0)  # H/4
            s2 = self.layer2(s1)  # H/8
            s3 = self.layer3(s2)  # H/16
            s4 = self.layer4(s3)  # H/32

            d4 = self.dec4(s4, s3)  # H/16
            d3 = self.dec3(d4, s2)  # H/8
            d2 = self.dec2(d3, s1)  # H/4
            d1 = self.dec1(d2, s0)  # H/2
            return self.head(d1)  # H

    return UNetResNet34(num_classes=num_classes)


def _load_imagenet_encoder(model: Any, checkpoint_path: Path) -> None:
    """Load a torchvision resnet34 ImageNet state dict into the encoder submodules only."""
    import torch

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    stem_map = {"conv1": "0", "bn1": "1"}
    remapped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("fc."):
            continue
        prefix = key.split(".", 1)[0]
        rest = key.split(".", 1)[1] if "." in key else ""
        if prefix in stem_map:
            remapped[f"stem.{stem_map[prefix]}.{rest}"] = value
        elif prefix in {"layer1", "layer2", "layer3", "layer4"}:
            remapped[key] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    non_decoder_missing = [m for m in missing if not m.startswith(("dec", "head"))]
    if non_decoder_missing or unexpected:
        msg = f"Unexpected encoder state_dict mismatch: missing={non_decoder_missing} unexpected={unexpected}"
        raise RuntimeError(msg)


def _download_checkpoint(url: str) -> Path:
    from upath import UPath

    local = Path(tempfile.mkdtemp()) / (UPath(url).name or "checkpoint.pth")
    local.write_bytes(UPath(url).read_bytes())
    return local


# --------------------------------------------------------------------------- #
# Data loading -- OAM RGB chips + OSM building polygons burned to raster masks.
# --------------------------------------------------------------------------- #


def _chip_paths(dataset_chips: str) -> list[Path]:
    return sorted(resolve_directory(dataset_chips, "*.tif*").rglob("*.tif"))


def _label_geoms(dataset_labels: str) -> list[Any]:
    import json

    from shapely.geometry import shape

    labels = resolve_directory(dataset_labels)
    label_file = labels if labels.is_file() else sorted(labels.rglob("*.geojson"))[0]
    data = json.loads(label_file.read_text())
    return [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]


def _read_chip(chip_path: Path) -> tuple[Any, Any, Any]:
    """RGB float32 array in [0, 1], the chip's affine transform, and its CRS."""
    import numpy as np
    import rasterio

    with rasterio.open(chip_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
        return rgb, src.transform, src.crs


def _burn_mask(geoms: list[Any], crs: Any, transform: Any, shape_hw: tuple[int, int]) -> Any:
    import numpy as np
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.ops import transform as shapely_transform

    if not geoms:
        return np.zeros(shape_hw, dtype=np.uint8)
    to_chip = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    local_geoms = [shapely_transform(lambda x, y, _z=None, t=to_chip: t.transform(x, y), g) for g in geoms]
    return rasterize([(g, 1) for g in local_geoms], out_shape=shape_hw, transform=transform, dtype="uint8")


def _normalize(rgb: Any) -> Any:
    import numpy as np

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    return (rgb - mean) / std


def _resize_chw(array: Any, size: int) -> Any:
    """Nearest-neighbour resize of a (C, H, W) or (H, W) array to (size, size), used
    only when a project's chips are not the platform's standard 256x256."""
    import numpy as np

    if array.ndim == 2:
        h, w = array.shape
        if (h, w) == (size, size):
            return array
        row_idx = (np.arange(size) * h / size).astype(np.int64).clip(max=h - 1)
        col_idx = (np.arange(size) * w / size).astype(np.int64).clip(max=w - 1)
        return array[row_idx][:, col_idx]
    _, h, w = array.shape
    if (h, w) == (size, size):
        return array
    row_idx = (np.arange(size) * h / size).astype(np.int64).clip(max=h - 1)
    col_idx = (np.arange(size) * w / size).astype(np.int64).clip(max=w - 1)
    return array[:, row_idx][:, :, col_idx]


def _load_batch(chip_paths: list[Path], geoms: list[Any], chip_size: int = CHIP_SIZE) -> tuple[Any, Any]:
    """Stack (N, 3, H, W) normalised chips and (N, H, W) int64 building masks."""
    import numpy as np

    images, masks = [], []
    for chip_path in chip_paths:
        rgb, transform, crs = _read_chip(chip_path)
        mask = _burn_mask(geoms, crs, transform, rgb.shape[1:])
        images.append(_normalize(_resize_chw(rgb, chip_size)))
        masks.append(_resize_chw(mask, chip_size).astype(np.int64))
    return np.stack(images), np.stack(masks)


# --------------------------------------------------------------------------- #
# preprocess / postprocess / predict -- referenced by stac-item.json.
# --------------------------------------------------------------------------- #


def preprocess(image_path: Any) -> Any:
    """Read an RGB chip, normalise with ImageNet stats, return an NCHW float32 tensor."""
    import numpy as np

    rgb, _transform, _crs = _read_chip(Path(image_path))
    rgb = _resize_chw(rgb, CHIP_SIZE)
    return _normalize(rgb)[np.newaxis, ...].astype(np.float32)


def postprocess(logits: Any, confidence_threshold: float = 0.5) -> Any:
    """Softmax the 2-channel logits and threshold the building-class probability."""
    import numpy as np

    building_idx = CLASS_NAMES.index("building")
    arr = np.asarray(logits)[0]  # (2, H, W)
    exp = np.exp(arr - arr.max(axis=0, keepdims=True))
    probs = exp / exp.sum(axis=0, keepdims=True)
    return (probs[building_idx] >= confidence_threshold).astype(np.uint8)


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    """Run ONNX inference over every chip in `input_images`, vectorise the building
    mask of each into WGS84 polygons, and return one merged FeatureCollection."""
    from pyproj import Transformer
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import transform as shapely_transform

    threshold = float(params.get("confidence_threshold", 0.5))
    input_name = session.get_inputs()[0].name

    input_path = resolve_directory(input_images)
    chip_paths = [input_path] if input_path.is_file() else _chip_paths(input_images)
    if not chip_paths:
        msg = f"No georeferenced (.tif) chips found in {input_path}"
        raise FileNotFoundError(msg)

    features: list[dict[str, Any]] = []
    for chip_path in chip_paths:
        _rgb, transform, crs = _read_chip(chip_path)
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        tensor = preprocess(chip_path)
        logits = session.run(None, {input_name: tensor})[0]
        mask = postprocess(logits, threshold)
        for geom, _value in shapes(mask, mask=mask.astype(bool), transform=transform):
            polygon = shapely_transform(lambda x, y, _z=None, t=to_wgs84: t.transform(x, y), shape(geom))
            features.append({"type": "Feature", "properties": {"label": "building"}, "geometry": mapping(polygon)})
    return {"type": "FeatureCollection", "features": features}


# --------------------------------------------------------------------------- #
# ZenML steps + pipelines.
# --------------------------------------------------------------------------- #


@step
def split_dataset(
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
) -> Annotated[dict[str, Any], "split_info"]:
    """Spatial block split on OAM-{x}-{y}-{z} tile coordinates: chips grouped into
    (x // block_size, y // block_size) blocks, whole blocks assigned to train or val
    so nearby chips never leak across the split. Falls back to a seeded random split
    for non-OAM filenames."""
    import random
    import re

    val_ratio = hyperparameters.get("val_ratio", 0.2)
    seed = hyperparameters.get("split_seed", 42)
    block_size = hyperparameters.get("block_size", 4)

    names = [p.name for p in _chip_paths(dataset_chips)]
    oam_pattern = re.compile(r"OAM-(\d+)-(\d+)-\d+")

    blocks: dict[tuple[int, int], list[str]] = {}
    fallback: list[str] = []
    for name in names:
        m = oam_pattern.match(name)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            blocks.setdefault((x // block_size, y // block_size), []).append(name)
        else:
            fallback.append(name)

    rng = random.Random(seed)
    block_keys = sorted(blocks.keys())
    rng.shuffle(block_keys)
    n_val_blocks = max(1, round(len(block_keys) * val_ratio)) if block_keys else 0
    val_block_keys = set(block_keys[:n_val_blocks])

    train_names = [n for k, names_in_block in blocks.items() for n in names_in_block if k not in val_block_keys]
    val_names = [n for k, names_in_block in blocks.items() for n in names_in_block if k in val_block_keys]

    rng.shuffle(fallback)
    n_val_fallback = round(len(fallback) * val_ratio)
    val_names += fallback[:n_val_fallback]
    train_names += fallback[n_val_fallback:]

    info = {
        "strategy": "spatial",
        "val_ratio": val_ratio,
        "seed": seed,
        "block_size": block_size,
        "train_count": len(train_names),
        "val_count": len(val_names),
        "train_chip_names": train_names,
        "val_chip_names": val_names,
        "description": (
            "OAM-x-y-z chips grouped into (x // block_size, y // block_size) blocks; "
            "whole blocks assigned to train or val so adjacent chips never split across "
            "the boundary. Non-OAM filenames fall back to a seeded random split."
        ),
    }
    log_metadata(metadata={"fair/split": info})
    return info


@step
def train_model(
    dataset_chips: str,
    dataset_labels: str,
    base_model_weights: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    model_name: str | None = None,
    base_model_id: str | None = None,
    dataset_id: str | None = None,
) -> Annotated[Any, "trained_model"]:
    """Train the U-Net (encoder + decoder) end to end on the train split only."""
    import mlflow
    import torch
    from torch import nn

    torch.manual_seed(int(split_info.get("seed", 42)))

    model = _build_model(num_classes=num_classes)
    if base_model_weights:
        _load_imagenet_encoder(model, _download_checkpoint(base_model_weights))

    train_names = set(split_info["train_chip_names"])
    chips = [p for p in _chip_paths(dataset_chips) if p.name in train_names]
    geoms = _label_geoms(dataset_labels)
    images, masks = _load_batch(chips, geoms)

    x = torch.from_numpy(images)
    y = torch.from_numpy(masks)

    epochs = int(hyperparameters.get("epochs", 5))
    batch_size = int(hyperparameters.get("batch_size", 4))
    lr = float(hyperparameters.get("learning_rate", 1e-3))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    n = x.shape[0]
    with mlflow_training_context(hyperparameters, model_name, base_model_id, dataset_id):
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                logits = model(x[idx])
                loss = criterion(logits, y[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(idx)
            mlflow.log_metric("train_loss", epoch_loss / max(n, 1), step=epoch)
    log_metadata(metadata={"train_chip_count": int(n)})
    return model.cpu()


@step
def evaluate_model(
    trained_model: Any,
    dataset_chips: str,
    dataset_labels: str,
    hyperparameters: dict[str, Any],
    split_info: dict[str, Any],
    num_classes: int = 2,
    class_names: list[str] | None = None,
) -> Annotated[dict[str, Any], "metrics"]:
    """Pixel precision/recall/IoU for the building class on the held-out val split."""
    import numpy as np
    import torch

    val_names = set(split_info["val_chip_names"])
    chips = [p for p in _chip_paths(dataset_chips) if p.name in val_names]
    geoms = _label_geoms(dataset_labels)
    images, masks = _load_batch(chips, geoms)

    trained_model.eval()
    with torch.no_grad():
        logits = trained_model(torch.from_numpy(images))
        preds = torch.argmax(logits, dim=1).numpy()

    building_idx = CLASS_NAMES.index("building")
    pred_pos = preds == building_idx
    true_pos_mask = masks == building_idx
    intersection = int(np.logical_and(pred_pos, true_pos_mask).sum())
    union = int(np.logical_or(pred_pos, true_pos_mask).sum())
    predicted_positive = int(pred_pos.sum())
    actual_positive = int(true_pos_mask.sum())

    metrics = {
        "iou_building": (intersection / union) if union > 0 else 1.0,
        "precision_building": (intersection / predicted_positive) if predicted_positive > 0 else 0.0,
        "recall_building": (intersection / actual_positive) if actual_positive > 0 else 0.0,
    }
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
    num_classes: int = 2,
) -> Annotated[bytes, "onnx_model"]:
    """Export the trained model to a single-file ONNX graph and validate it."""
    import onnx
    import torch

    chip_size = int(hyperparameters.get("chip_size", CHIP_SIZE))
    model = trained_model.cpu().eval()
    dummy = torch.randn(1, 3, chip_size, chip_size)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "model.onnx")
        torch.onnx.export(
            model,
            (dummy,),
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=18,
            dynamo=False,
        )
        onnx.checker.check_model(path)
        return Path(path).read_bytes()


@step
def run_inference(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any],
) -> Annotated[dict[str, Any], "predictions"]:
    from fair.serve.base import load_session

    return predict(load_session(model_uri), input_images, inference_params)


@pipeline
def training_pipeline(
    base_model_weights: str,
    dataset_chips: str,
    dataset_labels: str,
    num_classes: int,
    hyperparameters: dict[str, Any],
) -> None:
    split_info = split_dataset(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
    )
    trained = train_model(
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        base_model_weights=base_model_weights,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    evaluate_model(
        trained_model=trained,
        dataset_chips=dataset_chips,
        dataset_labels=dataset_labels,
        hyperparameters=hyperparameters,
        split_info=split_info,
        num_classes=num_classes,
    )
    export_onnx(trained_model=trained, hyperparameters=hyperparameters, num_classes=num_classes)


@pipeline
def inference_pipeline(
    model_uri: str,
    input_images: str,
    inference_params: dict[str, Any] | None = None,
) -> None:
    run_inference(
        model_uri=model_uri,
        input_images=input_images,
        inference_params=inference_params or {},
    )
