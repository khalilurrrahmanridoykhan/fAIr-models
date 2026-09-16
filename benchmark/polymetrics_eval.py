"""Polygon-level quality (per fAIr's contributing guide's recommendation): vectorise
both the Banepa-trained and same-distribution-trained model's predictions on the
same 400-tile eval sample, and score them with polymetrics against the real OSM
ground-truth polygons (label_geojson) shipped in the vhr-building-segmentation
dataset.
"""

import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import polymetrics
import pyarrow.parquet as pq
import torch
from PIL import Image
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape

from models.unet_buildings import pipeline as unet_pipeline
from models.unet_buildings.pipeline import CLASS_NAMES, _build_model, _normalize

patch.object(unet_pipeline, "log_metadata", lambda **_kwargs: None).start()

REPO_ROOT = Path(__file__).resolve().parent
TEST_SHARDS = [
    REPO_ROOT / "data/data/test-00000-of-00002.parquet",
    REPO_ROOT / "data/data/test-00001-of-00002.parquet",
]
EVAL_SAMPLE_SIZE = 400
EVAL_SEED = 42


def load_eval_rows() -> list[dict]:
    rows: list[dict] = []
    for shard in TEST_SHARDS:
        rows.extend(
            pq.ParquetFile(shard)
            .read(columns=["image", "mask", "tile_id", "country", "label_geojson", "bbox_west", "bbox_south", "bbox_east", "bbox_north"])
            .to_pylist()
        )
    idx = np.random.default_rng(EVAL_SEED).choice(len(rows), size=EVAL_SAMPLE_SIZE, replace=False)
    return [rows[i] for i in idx]


def predict_polygons(model, rows: list[dict]) -> list[dict]:
    """Run the model on every tile and vectorise the building mask to WGS84 polygons."""
    features = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            image = np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))
            rgb_chw = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
            tensor = torch.from_numpy(_normalize(rgb_chw)[np.newaxis, ...])
            logits = model(tensor)
            pred = torch.argmax(logits, dim=1)[0].numpy().astype("uint8")
            transform = from_bounds(row["bbox_west"], row["bbox_south"], row["bbox_east"], row["bbox_north"], 256, 256)
            for geom, val in shapes(pred, mask=pred.astype(bool), transform=transform):
                if val == CLASS_NAMES.index("building"):
                    features.append({"geometry": shape(geom), "tile_id": row["tile_id"]})
    return features


def ground_truth_polygons(rows: list[dict]) -> list[dict]:
    features = []
    for row in rows:
        if not row["label_geojson"]:
            continue
        fc = json.loads(row["label_geojson"])
        for feat in fc.get("features", []):
            if feat.get("geometry"):
                features.append({"geometry": shape(feat["geometry"]), "tile_id": row["tile_id"]})
    return features


def run_one(label: str, checkpoint_path: Path, rows: list[dict], gt_gdf: gpd.GeoDataFrame) -> None:
    model = _build_model(num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    pred_features = predict_polygons(model, rows)
    print(f"\n=== {label}: {len(pred_features)} predicted polygons across {len(rows)} tiles ===")
    if not pred_features:
        print("No polygons predicted at all -- skipping polymetrics (would be a degenerate 0-precision case).")
        return
    pred_gdf = gpd.GeoDataFrame(pred_features, crs="EPSG:4326")
    result = polymetrics.evaluate(gt_gdf, pred_gdf, iou_threshold=0.5)
    print(result)


def main() -> None:
    rows = load_eval_rows()
    gt_features = ground_truth_polygons(rows)
    gt_gdf = gpd.GeoDataFrame(gt_features, crs="EPSG:4326")
    print(f"Ground truth: {len(gt_gdf)} real OSM building polygons across {len(rows)} tiles")

    run_one("Banepa-trained (single-city)", REPO_ROOT / "unet_buildings_banepa.pt", rows, gt_gdf)
    run_one("Same-distribution-trained (96 diverse chips)", REPO_ROOT / "unet_buildings_same_distribution.pt", rows, gt_gdf)


if __name__ == "__main__":
    main()
