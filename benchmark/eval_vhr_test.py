"""Out-of-domain evaluation: the Banepa-finetuned unet_buildings model, scored
against a random sample of hotosm/vhr-building-segmentation's real test split
(93 mapping projects across 21 countries -- none of which is Banepa/Nepal).
This is the comparison HOT's fAIr maintainer (Kshitij Sharma) asked for.
"""

import io
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from models.unet_buildings.pipeline import CLASS_NAMES, _build_model, _normalize

REPO_ROOT = Path(__file__).resolve().parent
TEST_SHARDS = [
    REPO_ROOT / "data/data/test-00000-of-00002.parquet",
    REPO_ROOT / "data/data/test-00001-of-00002.parquet",
]
MODEL_PATH = REPO_ROOT / "unet_buildings_banepa.pt"
SAMPLE_SIZE = 400
SEED = 42
BATCH_SIZE = 8

FIELDS = [
    "image",
    "mask",
    "tile_id",
    "project_name",
    "country",
    "num_buildings",
]


def load_sample() -> list[dict]:
    rows: list[dict] = []
    for shard in TEST_SHARDS:
        pf = pq.ParquetFile(shard)
        table = pf.read(columns=FIELDS)
        rows.extend(table.to_pylist())
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(rows), size=min(SAMPLE_SIZE, len(rows)), replace=False)
    return [rows[i] for i in idx]


def decode(row: dict) -> tuple[np.ndarray, np.ndarray]:
    image = np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))  # (H, W, 3) uint8
    mask = np.array(Image.open(io.BytesIO(row["mask"]["bytes"])))  # (H, W) uint8, 0/255
    rgb_chw = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0  # (3, H, W)
    building_idx = CLASS_NAMES.index("building")
    binary_mask = (mask > 127).astype(np.int64) * building_idx
    return _normalize(rgb_chw), binary_mask


def main() -> None:
    print(f"Loading {SAMPLE_SIZE} random tiles from the real hotosm/vhr-building-segmentation test split...")
    rows = load_sample()
    countries = sorted({r["country"] for r in rows})
    print(f"Sampled {len(rows)} tiles across {len(countries)} countries: {countries}")

    images = np.zeros((len(rows), 3, 256, 256), dtype=np.float32)
    masks = np.zeros((len(rows), 256, 256), dtype=np.int64)
    for i, row in enumerate(rows):
        images[i], masks[i] = decode(row)

    model = _build_model(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    all_preds = np.zeros_like(masks)
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch = torch.from_numpy(images[start : start + BATCH_SIZE])
            logits = model(batch)
            all_preds[start : start + BATCH_SIZE] = torch.argmax(logits, dim=1).numpy()

    building_idx = CLASS_NAMES.index("building")
    pred_pos = all_preds == building_idx
    true_pos = masks == building_idx
    intersection = int(np.logical_and(pred_pos, true_pos).sum())
    union = int(np.logical_or(pred_pos, true_pos).sum())
    pred_sum = int(pred_pos.sum())
    true_sum = int(true_pos.sum())
    iou = intersection / union if union else 1.0
    precision = intersection / pred_sum if pred_sum else 0.0
    recall = intersection / true_sum if true_sum else 0.0

    print("\n=== Out-of-domain result: Banepa-trained unet_buildings vs. real vhr-building-segmentation test tiles ===")
    print(f"Sample size: {len(rows)} tiles, {len(countries)} countries")
    print(f"Ground-truth building-pixel fraction: {masks.mean():.4f} (dataset-wide reference: 0.1353)")
    print(f"Pixel IoU:       {iou:.4f}")
    print(f"Pixel precision: {precision:.4f}")
    print(f"Pixel recall:    {recall:.4f}")

    # Per-tile accuracy, so a handful of catastrophic tiles don't hide behind a pooled average.
    tile_ious = []
    for i in range(len(rows)):
        u = int(np.logical_or(pred_pos[i], true_pos[i]).sum())
        inter = int(np.logical_and(pred_pos[i], true_pos[i]).sum())
        if true_pos[i].sum() == 0 and pred_pos[i].sum() == 0:
            continue  # empty-empty tiles are trivially "correct"; exclude from the per-tile distribution
        tile_ious.append(inter / u if u else 0.0)
    tile_ious_arr = np.array(tile_ious)
    print(f"\nTiles with any building (predicted or true): {len(tile_ious_arr)}/{len(rows)}")
    print(f"Per-tile IoU: mean={tile_ious_arr.mean():.4f} median={np.median(tile_ious_arr):.4f}")

    by_country: dict[str, list[float]] = defaultdict(list)
    for row, i in zip(rows, range(len(rows))):
        u = int(np.logical_or(pred_pos[i], true_pos[i]).sum())
        inter = int(np.logical_and(pred_pos[i], true_pos[i]).sum())
        if true_pos[i].sum() == 0 and pred_pos[i].sum() == 0:
            continue
        by_country[row["country"]].append(inter / u if u else 0.0)
    print("\nPer-country mean IoU (tiles with any building):")
    for country, vals in sorted(by_country.items(), key=lambda kv: -len(kv[1])):
        print(f"  {country:25s} n={len(vals):3d}  mean_iou={np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
