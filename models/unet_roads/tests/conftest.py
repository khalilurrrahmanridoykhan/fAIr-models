"""Deterministic toy chips and road labels for the unet-roads tests.

Each chip is bright along a horizontal strip through its middle, with a matching
`highway=residential` LineString along that strip, so a model with real spatial
context can learn to separate the two classes.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

GRID = 3
CHIP_PIXELS = 64
STEP_DEG = 0.001
BASE_LON, BASE_LAT = 85.5, 27.6
ROAD_TAG = "trunk"  # _HIGHWAY_WIDTH_M["trunk"] == 10.0m -> matches the visual band below
_METERS_PER_PIXEL = (STEP_DEG / CHIP_PIXELS) * 111_320.0 * math.cos(math.radians(BASE_LAT))
_BAND_HALF_WIDTH_PX = max(2, round((10.0 / _METERS_PER_PIXEL) / 2))
_EAST, _NORTH = BASE_LON + GRID * STEP_DEG, BASE_LAT + GRID * STEP_DEG
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [[BASE_LON, BASE_LAT], [_EAST, BASE_LAT], [_EAST, _NORTH], [BASE_LON, _NORTH], [BASE_LON, BASE_LAT]]
    ],
}
_BBOX = [BASE_LON, BASE_LAT, _EAST, _NORTH]


def create_toy_data(root: Path) -> dict[str, Path]:
    chips_dir = root / "chips"
    chips_dir.mkdir(parents=True)
    road_features = []

    for row in range(GRID):
        for col in range(GRID):
            west = BASE_LON + col * STEP_DEG
            south = BASE_LAT + row * STEP_DEG
            east, north = west + STEP_DEG, south + STEP_DEG
            transform = from_bounds(west, south, east, north, CHIP_PIXELS, CHIP_PIXELS)
            pixels = np.full((3, CHIP_PIXELS, CHIP_PIXELS), 32, dtype=np.uint8)
            mid_row = CHIP_PIXELS // 2
            pixels[:, mid_row - _BAND_HALF_WIDTH_PX : mid_row + _BAND_HALF_WIDTH_PX, :] = 224
            with rasterio.open(
                chips_dir / f"OAM-{col:02d}-{row:02d}-18.tif",
                "w",
                driver="GTiff",
                width=CHIP_PIXELS,
                height=CHIP_PIXELS,
                count=3,
                dtype="uint8",
                crs=CRS.from_epsg(4326),
                transform=transform,
            ) as dst:
                dst.write(pixels)
            mid_lat = south + (north - south) / 2
            road_features.append(
                {
                    "type": "Feature",
                    "properties": {"tags": {"highway": ROAD_TAG}},
                    "geometry": {"type": "LineString", "coordinates": [[west, mid_lat], [east, mid_lat]]},
                }
            )

    labels_dir = root / "labels"
    labels_dir.mkdir()
    feature_collection = {"type": "FeatureCollection", "features": road_features}
    (labels_dir / "labels.geojson").write_text(json.dumps(feature_collection))

    stac_path = root / "dataset-stac-item.json"
    stac_path.write_text(json.dumps(_build_dataset_stac_item(chips_dir, labels_dir), indent=2))
    return {"chips": chips_dir, "labels": labels_dir, "dataset_stac_item": stac_path}


@pytest.fixture(scope="session")
def generate_toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return create_toy_data(tmp_path_factory.mktemp("toy_unet_roads"))


def _build_dataset_stac_item(chips_dir: Path, labels_dir: Path) -> dict[str, Any]:
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": ["https://stac-extensions.github.io/label/v1.0.1/schema.json"],
        "id": "toy-unet-roads",
        "geometry": _GEOMETRY,
        "bbox": _BBOX,
        "properties": {
            "datetime": "2026-05-18T00:00:00Z",
            "description": "Toy unet-roads dataset",
            "label:type": "vector",
            "label:tasks": ["segmentation"],
            "label:classes": [{"name": "road", "classes": ["yes"]}],
            "label:description": "Segmentation labels",
            "keywords": ["road"],
            "fair:user_id": "test",
            "version": "1",
            "deprecated": False,
            "license": "CC-BY-4.0",
            "providers": [{"name": "HOTOSM", "roles": ["producer"], "url": "https://www.hotosm.org"}],
        },
        "assets": {
            "chips": {"href": str(chips_dir), "type": "image/tiff", "roles": ["data"]},
            "labels": {"href": str(labels_dir), "type": "application/geo+json", "roles": ["labels"]},
        },
        "links": [],
    }
