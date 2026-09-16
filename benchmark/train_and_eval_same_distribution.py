"""Isolates whether the Banepa-trained model's poor cross-country score (see
eval_vhr_test.py) is a domain-shift effect or a capability limit of the
architecture/training recipe itself.

Trains on a small sample drawn from the SAME pool as the evaluation set (a
disjoint slice of the real hotosm/vhr-building-segmentation test parquet --
not the dataset's official train split, which was too large to download
locally; documented here for transparency) and evaluates on the same held-out
400-tile sample used for the Banepa comparison.
"""

import io
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from models.unet_buildings import pipeline as unet_pipeline
from models.unet_buildings.pipeline import (
    CLASS_NAMES,
    _build_model,
    _load_imagenet_encoder,
    _normalize,
)

LOCAL_CHECKPOINT = Path(__file__).resolve().parent / "resnet34-b627a593.pth"

patch.object(unet_pipeline, "log_metadata", lambda **_kwargs: None).start()

REPO_ROOT = Path(__file__).resolve().parent
TEST_SHARDS = [
    REPO_ROOT / "data/data/test-00000-of-00002.parquet",
    REPO_ROOT / "data/data/test-00001-of-00002.parquet",
]
EVAL_SAMPLE_SIZE = 400  # must match eval_vhr_test.py's sample for a fair comparison
EVAL_SEED = 42
TRAIN_SAMPLE_SIZE = 96  # comparable to Banepa's 92 training chips
TRAIN_SEED = 777  # different seed -> disjoint pool from the eval sample with overwhelming probability
EPOCHS = 20
BATCH_SIZE = 4


def load_all_rows() -> list[dict]:
    rows: list[dict] = []
    for shard in TEST_SHARDS:
        rows.extend(pq.ParquetFile(shard).read(columns=["image", "mask", "tile_id", "country"]).to_pylist())
    return rows


def decode(row: dict) -> tuple[np.ndarray, np.ndarray]:
    image = np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))
    mask = np.array(Image.open(io.BytesIO(row["mask"]["bytes"])))
    rgb_chw = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    building_idx = CLASS_NAMES.index("building")
    binary_mask = (mask > 127).astype(np.int64) * building_idx
    return _normalize(rgb_chw), binary_mask


def main() -> None:
    rows = load_all_rows()
    eval_idx = set(np.random.default_rng(EVAL_SEED).choice(len(rows), size=EVAL_SAMPLE_SIZE, replace=False).tolist())

    train_rng = np.random.default_rng(TRAIN_SEED)
    train_idx: list[int] = []
    while len(train_idx) < TRAIN_SAMPLE_SIZE:
        candidate = int(train_rng.integers(0, len(rows)))
        if candidate not in eval_idx and candidate not in train_idx:
            train_idx.append(candidate)
    assert set(train_idx).isdisjoint(eval_idx), "train/eval leakage!"

    train_rows = [rows[i] for i in train_idx]
    eval_rows = [rows[i] for i in sorted(eval_idx)]
    print(f"train: {len(train_rows)} tiles (countries: {sorted({r['country'] for r in train_rows})})")
    print(f"eval:  {len(eval_rows)} tiles (same 400-tile sample as eval_vhr_test.py, seed={EVAL_SEED})")

    train_images = np.zeros((len(train_rows), 3, 256, 256), dtype=np.float32)
    train_masks = np.zeros((len(train_rows), 256, 256), dtype=np.int64)
    for i, row in enumerate(train_rows):
        train_images[i], train_masks[i] = decode(row)

    torch.manual_seed(42)
    model = _build_model(num_classes=2)
    _load_imagenet_encoder(model, LOCAL_CHECKPOINT)

    x = torch.from_numpy(train_images)
    y = torch.from_numpy(train_masks)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    n = x.shape[0]
    print("=== training on same-distribution sample ===")
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(x[idx])
            loss = criterion(logits, y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(idx)
        print(f"epoch {epoch + 1}/{EPOCHS}  loss={epoch_loss / n:.4f}")

    torch.save(model.state_dict(), REPO_ROOT / "unet_buildings_same_distribution.pt")

    print("=== evaluating on the held-out 400-tile sample ===")
    eval_images = np.zeros((len(eval_rows), 3, 256, 256), dtype=np.float32)
    eval_masks = np.zeros((len(eval_rows), 256, 256), dtype=np.int64)
    for i, row in enumerate(eval_rows):
        eval_images[i], eval_masks[i] = decode(row)

    model.eval()
    all_preds = np.zeros_like(eval_masks)
    with torch.no_grad():
        for start in range(0, len(eval_rows), 8):
            batch = torch.from_numpy(eval_images[start : start + 8])
            logits = model(batch)
            all_preds[start : start + 8] = torch.argmax(logits, dim=1).numpy()

    building_idx = CLASS_NAMES.index("building")
    pred_pos = all_preds == building_idx
    true_pos = eval_masks == building_idx
    intersection = int(np.logical_and(pred_pos, true_pos).sum())
    union = int(np.logical_or(pred_pos, true_pos).sum())
    pred_sum = int(pred_pos.sum())
    true_sum = int(true_pos.sum())
    iou = intersection / union if union else 1.0
    precision = intersection / pred_sum if pred_sum else 0.0
    recall = intersection / true_sum if true_sum else 0.0

    print("\n=== Same-distribution-trained model vs. the identical 400-tile eval sample ===")
    print(f"Pixel IoU:       {iou:.4f}")
    print(f"Pixel precision: {precision:.4f}")
    print(f"Pixel recall:    {recall:.4f}")
    print("\n(Compare to eval_vhr_test.py's Banepa-trained result on this same eval sample.)")


if __name__ == "__main__":
    main()
