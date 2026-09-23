"""unet-roads: a U-Net with an ImageNet-pretrained ResNet-34 encoder for binary road
segmentation, full end-to-end finetune. Architecture, training loop, spatial split,
and ONNX export mirror unet_buildings -- the road-specific work is entirely in how
labels become training masks and how predictions become line output.

OSM road data is LineString centrelines, not areas, so two translations happen at
the boundary of an otherwise ordinary segmentation model:
  - training: each centreline is buffered by a width looked up from its `highway`
    tag (a footway and a trunk road are not the same width) before rasterising.
  - inference: the predicted road-area mask is skeletonised back down to a
    1-pixel-wide centreline and vectorised to LineStrings, since fAIr requires the
    declared output geometry type (`line`, here) to match what is actually returned.
"""

import tempfile
from pathlib import Path
from typing import Annotated, Any

from zenml import log_metadata, pipeline, step

from fair.utils.data import resolve_directory
from fair.zenml.instrumentation import log_evaluation_results, mlflow_training_context
from fair.zenml.materializers import ONNXMaterializer

MODEL_NAME = "unet-roads"
CLASS_NAMES = ("background", "road")
CHIP_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Half-widths in metres per OSM `highway` tag, used to buffer centrelines into a
# training mask. Deliberately coarse -- the model only needs "plausible road area",
# not surveyed carriageway width. Unknown/missing tags fall back to _DEFAULT_WIDTH_M.
_HIGHWAY_WIDTH_M: dict[str, float] = {
    "motorway": 12.0,
    "motorway_link": 10.0,
    "trunk": 10.0,
    "trunk_link": 9.0,
    "primary": 8.0,
    "primary_link": 7.0,
    "secondary": 7.0,
    "secondary_link": 6.0,
    "tertiary": 6.0,
    "tertiary_link": 5.0,
    "unclassified": 5.0,
    "residential": 5.0,
    "living_street": 5.0,
    "service": 4.0,
    "track": 3.0,
    "pedestrian": 3.0,
    "path": 2.0,
    "footway": 2.0,
    "cycleway": 2.0,
    "steps": 1.5,
}
_DEFAULT_WIDTH_M = 4.0
_METERS_PER_DEGREE_LAT = 111_320.0


# --------------------------------------------------------------------------- #
# Model definition -- identical U-Net/ResNet-34 to unet_buildings.
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
        def __init__(self, num_classes: int = 2) -> None:
            super().__init__()
            backbone = resnet34(weights=None)
            self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.pool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4

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
            s0 = self.stem(x)
            p0 = self.pool(s0)
            s1 = self.layer1(p0)
            s2 = self.layer2(s1)
            s3 = self.layer3(s2)
            s4 = self.layer4(s3)

            d4 = self.dec4(s4, s3)
            d3 = self.dec3(d4, s2)
            d2 = self.dec2(d3, s1)
            d1 = self.dec1(d2, s0)
            return self.head(d1)

    return UNetResNet34(num_classes=num_classes)


def _load_imagenet_encoder(model: Any, checkpoint_path: Path) -> None:
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
# Data loading -- OAM RGB chips + OSM highway centrelines, buffered by tag.
# --------------------------------------------------------------------------- #


def _chip_paths(dataset_chips: str) -> list[Path]:
    return sorted(resolve_directory(dataset_chips, "*.tif*").rglob("*.tif"))


def _road_geoms(dataset_labels: str) -> list[tuple[Any, str]]:
    """(LineString, highway tag) pairs from the dataset GeoJSON. Non-line geometries
    are skipped; a missing/unknown `highway` tag is treated as its own tag string so
    the width lookup's default still applies."""
    import json

    from shapely.geometry import shape

    labels = resolve_directory(dataset_labels)
    label_file = labels if labels.is_file() else sorted(labels.rglob("*.geojson"))[0]
    data = json.loads(label_file.read_text())
    pairs = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        if not geom or geom.get("type") not in {"LineString", "MultiLineString"}:
            continue
        tags = feat.get("properties", {}).get("tags", {}) or feat.get("properties", {})
        highway = tags.get("highway", "unclassified")
        pairs.append((shape(geom), highway))
    return pairs


def _read_chip(chip_path: Path) -> tuple[Any, Any, Any]:
    import numpy as np
    import rasterio

    with rasterio.open(chip_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
        return rgb, src.transform, src.crs


def _burn_road_mask(
    road_geoms: list[tuple[Any, str]],
    crs: Any,
    transform: Any,
    shape_hw: tuple[int, int],
) -> Any:
    """Buffer each centreline by its highway-tag width (converted from metres to the
    chip CRS's units at the chip's latitude) and rasterise the union as a binary mask."""
    import math

    import numpy as np
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.ops import transform as shapely_transform

    if not road_geoms:
        return np.zeros(shape_hw, dtype=np.uint8)

    to_chip = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    is_geographic = crs is None or (hasattr(crs, "is_geographic") and crs.is_geographic)

    buffered = []
    for geom, highway in road_geoms:
        local_geom = shapely_transform(lambda x, y, _z=None, t=to_chip: t.transform(x, y), geom)
        width_m = _HIGHWAY_WIDTH_M.get(highway, _DEFAULT_WIDTH_M)
        if is_geographic:
            lat = geom.centroid.y
            meters_per_degree = _METERS_PER_DEGREE_LAT * max(math.cos(math.radians(lat)), 0.01)
            buffer_dist = (width_m / 2) / meters_per_degree
        else:
            buffer_dist = width_m / 2
        buffered.append(local_geom.buffer(buffer_dist))

    return rasterize([(g, 1) for g in buffered], out_shape=shape_hw, transform=transform, dtype="uint8")


def _normalize(rgb: Any) -> Any:
    import numpy as np

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    return (rgb - mean) / std


def _resize_chw(array: Any, size: int) -> Any:
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


def _load_batch(
    chip_paths: list[Path], road_geoms: list[tuple[Any, str]], chip_size: int = CHIP_SIZE
) -> tuple[Any, Any]:
    import numpy as np

    images, masks = [], []
    for chip_path in chip_paths:
        rgb, transform, crs = _read_chip(chip_path)
        mask = _burn_road_mask(road_geoms, crs, transform, rgb.shape[1:])
        images.append(_normalize(_resize_chw(rgb, chip_size)))
        masks.append(_resize_chw(mask, chip_size).astype(np.int64))
    return np.stack(images), np.stack(masks)


# --------------------------------------------------------------------------- #
# preprocess / postprocess / predict -- referenced by stac-item.json.
# --------------------------------------------------------------------------- #


def preprocess(image_path: Any) -> Any:
    import numpy as np

    rgb, _transform, _crs = _read_chip(Path(image_path))
    rgb = _resize_chw(rgb, CHIP_SIZE)
    return _normalize(rgb)[np.newaxis, ...].astype(np.float32)


def postprocess(logits: Any, confidence_threshold: float = 0.5) -> Any:
    """Softmax the 2-channel logits and threshold the road-class probability into a
    binary road-area mask (not yet a line -- see `_mask_to_lines`)."""
    import numpy as np

    road_idx = CLASS_NAMES.index("road")
    arr = np.asarray(logits)[0]
    exp = np.exp(arr - arr.max(axis=0, keepdims=True))
    probs = exp / exp.sum(axis=0, keepdims=True)
    return (probs[road_idx] >= confidence_threshold).astype(np.uint8)


def _mask_to_lines(mask: Any, transform: Any) -> list[Any]:
    """Skeletonise a road-area mask to a 1-pixel centreline and vectorise it into
    LineStrings in the chip's own CRS (pixel space -> affine transform).

    Skeleton pixels form a thin, mostly path-like structure; a minimum spanning
    tree over each connected component's 8-neighbourhood graph recovers that path
    (including simple branches) without needing a full topological road-graph
    library, and `linemerge` stitches the resulting 1-pixel edges back into
    contiguous LineStrings.
    """
    import numpy as np
    from scipy.sparse.csgraph import minimum_spanning_tree
    from shapely.geometry import LineString
    from shapely.ops import linemerge
    from skimage.measure import label as cc_label
    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask.astype(bool))
    if not skeleton.any():
        return []

    components = cc_label(skeleton, connectivity=2)
    lines: list[Any] = []
    for component_id in range(1, components.max() + 1):
        ys, xs = np.nonzero(components == component_id)
        if len(xs) < 2:
            continue
        coords = np.stack([xs, ys], axis=1).astype(np.float64)
        # Sparse distance matrix restricted to 8-connected neighbours (skeleton
        # pixels are thin, so only near-adjacent pixels are ever true neighbours).
        n = len(coords)
        rows, cols, dists = [], [], []
        for i in range(n):
            deltas = coords - coords[i]
            dist = np.hypot(deltas[:, 0], deltas[:, 1])
            neighbours = np.where((dist > 0) & (dist <= np.sqrt(2) + 1e-6))[0]
            for j in neighbours:
                rows.append(i)
                cols.append(j)
                dists.append(dist[j])
        if not rows:
            continue
        from scipy.sparse import coo_matrix

        graph = coo_matrix((dists, (rows, cols)), shape=(n, n))
        mst = minimum_spanning_tree(graph).tocoo()
        for i, j in zip(mst.row, mst.col, strict=True):
            px, py = transform * (coords[i][0] + 0.5, coords[i][1] + 0.5)
            qx, qy = transform * (coords[j][0] + 0.5, coords[j][1] + 0.5)
            lines.append(LineString([(px, py), (qx, qy)]))

    if not lines:
        return []
    merged = linemerge(lines)
    if merged.geom_type == "LineString":
        return [merged]
    return list(merged.geoms)


def predict(session: Any, input_images: str, params: dict[str, Any]) -> dict[str, Any]:
    from pyproj import Transformer
    from shapely.geometry import mapping
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
        for line in _mask_to_lines(mask, transform):
            wgs84_line = shapely_transform(lambda x, y, _z=None, t=to_wgs84: t.transform(x, y), line)
            features.append({"type": "Feature", "properties": {"label": "road"}, "geometry": mapping(wgs84_line)})
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
    """Spatial block split on OAM-{x}-{y}-{z} tile coordinates, identical strategy
    to unet_buildings: whole (x // block_size, y // block_size) blocks go to train
    or val so adjacent chips never leak across the boundary."""
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
    import mlflow
    import torch
    from torch import nn

    torch.manual_seed(int(split_info.get("seed", 42)))

    model = _build_model(num_classes=num_classes)
    if base_model_weights:
        _load_imagenet_encoder(model, _download_checkpoint(base_model_weights))

    train_names = set(split_info["train_chip_names"])
    chips = [p for p in _chip_paths(dataset_chips) if p.name in train_names]
    road_geoms = _road_geoms(dataset_labels)
    images, masks = _load_batch(chips, road_geoms)

    x = torch.from_numpy(images)
    y = torch.from_numpy(masks)

    epochs = int(hyperparameters.get("epochs", 5))
    batch_size = int(hyperparameters.get("batch_size", 4))
    lr = float(hyperparameters.get("learning_rate", 1e-3))
    max_class_weight = float(hyperparameters.get("max_class_weight", 8.0))

    # Road pixels are typically a small minority of a chip (a few percent is common),
    # far more imbalanced than e.g. buildings. Unweighted cross-entropy collapses to
    # predicting background everywhere in that regime (empirically confirmed: 0.0 IoU
    # on a real ~5%-road sample). Weight the road class by its own inverse frequency
    # in *this* training batch -- every project's road density differs, so the weight
    # must adapt per run rather than being a fixed constant -- capped so a very sparse
    # project can't push the weight so high training destabilises the other way
    # (over-predicting road everywhere, cratering precision).
    road_idx = CLASS_NAMES.index("road")
    road_fraction = float((y.numpy() == road_idx).mean())
    road_weight = min((1.0 - road_fraction) / max(road_fraction, 1e-6), max_class_weight)
    class_weights = torch.tensor([1.0, road_weight], dtype=torch.float32)
    log_metadata(metadata={"fair/road_pixel_fraction": road_fraction, "fair/road_class_weight": road_weight})

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
    import numpy as np
    import torch

    val_names = set(split_info["val_chip_names"])
    chips = [p for p in _chip_paths(dataset_chips) if p.name in val_names]
    road_geoms = _road_geoms(dataset_labels)
    images, masks = _load_batch(chips, road_geoms)

    trained_model.eval()
    with torch.no_grad():
        logits = trained_model(torch.from_numpy(images))
        preds = torch.argmax(logits, dim=1).numpy()

    road_idx = CLASS_NAMES.index("road")
    pred_pos = preds == road_idx
    true_pos_mask = masks == road_idx
    intersection = int(np.logical_and(pred_pos, true_pos_mask).sum())
    union = int(np.logical_or(pred_pos, true_pos_mask).sum())
    predicted_positive = int(pred_pos.sum())
    actual_positive = int(true_pos_mask.sum())

    metrics = {
        "iou_road": (intersection / union) if union > 0 else 1.0,
        "precision_road": (intersection / predicted_positive) if predicted_positive > 0 else 0.0,
        "recall_road": (intersection / actual_positive) if actual_positive > 0 else 0.0,
    }
    log_evaluation_results(metrics)
    return metrics


@step(output_materializers={"onnx_model": ONNXMaterializer})
def export_onnx(
    trained_model: Any,
    hyperparameters: dict[str, Any],
    num_classes: int = 2,
) -> Annotated[bytes, "onnx_model"]:
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
