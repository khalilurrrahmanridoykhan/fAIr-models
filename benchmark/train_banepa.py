"""Real finetune of unet_buildings on fAIr's own local Banepa reference dataset,
then in-domain evaluation on the Banepa test split. Saves the trained torch model
to benchmark/unet_buildings_banepa.pt for the out-of-domain HF benchmark to reuse.
"""

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.unet_buildings import pipeline as unet_pipeline
from models.unet_buildings.pipeline import (
    _build_model,
    _download_checkpoint,
    _label_geoms,
    _load_batch,
    _load_imagenet_encoder,
)


@contextmanager
def _noop_context(*_args, **_kwargs):
    yield


_PATCHES = [
    patch.object(unet_pipeline, "log_metadata", lambda **_kwargs: None),
    patch.object(unet_pipeline, "log_evaluation_results", lambda *_a, **_k: None),
    patch.object(unet_pipeline, "mlflow_training_context", _noop_context),
]
for _p in _PATCHES:
    _p.start()

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CHIPS = REPO_ROOT / "data/sample/train/oam"
TRAIN_LABELS = REPO_ROOT / "data/sample/train/osm"
TEST_CHIPS = REPO_ROOT / "data/sample/test/oam"
TEST_LABELS = REPO_ROOT / "data/sample/test/osm"

HYPERPARAMETERS = {
    "epochs": 20,
    "batch_size": 4,
    "learning_rate": 1e-3,
    "val_ratio": 0.15,
    "split_seed": 42,
    "block_size": 4,
}


def main() -> None:
    from models.unet_buildings.pipeline import split_dataset

    print("=== split ===")
    split_info = split_dataset.entrypoint(
        dataset_chips=str(TRAIN_CHIPS),
        dataset_labels=str(TRAIN_LABELS),
        hyperparameters=HYPERPARAMETERS,
    )
    print(split_info["strategy"], "train:", split_info["train_count"], "val:", split_info["val_count"])

    print("=== build model + load ImageNet encoder ===")
    torch.manual_seed(HYPERPARAMETERS["split_seed"])
    model = _build_model(num_classes=2)
    ckpt = _download_checkpoint("https://download.pytorch.org/models/resnet34-b627a593.pth")
    _load_imagenet_encoder(model, ckpt)

    train_names = set(split_info["train_chip_names"])
    chip_paths = sorted(TRAIN_CHIPS.glob("*.tif"))
    train_chips = [p for p in chip_paths if p.name in train_names]
    geoms = _label_geoms(str(TRAIN_LABELS))
    print(f"train chips: {len(train_chips)}, building polygons in labels: {len(geoms)}")

    print("=== loading + rasterizing train batch ===")
    t0 = time.time()
    images, masks = _load_batch(train_chips, geoms)
    print(f"loaded {images.shape} in {time.time() - t0:.1f}s; building-pixel fraction: {masks.mean():.4f}")

    x = torch.from_numpy(images)
    y = torch.from_numpy(masks)

    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    n = x.shape[0]
    batch_size = HYPERPARAMETERS["batch_size"]
    print("=== training ===")
    t0 = time.time()
    for epoch in range(HYPERPARAMETERS["epochs"]):
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
        elapsed = time.time() - t0
        print(f"epoch {epoch + 1}/{HYPERPARAMETERS['epochs']}  loss={epoch_loss / n:.4f}  elapsed={elapsed:.1f}s")

    out_path = Path(__file__).resolve().parent / "unet_buildings_banepa.pt"
    torch.save(model.state_dict(), out_path)
    print(f"saved trained model to {out_path}")

    print("=== in-domain eval on Banepa test split ===")
    test_chip_paths = sorted(TEST_CHIPS.glob("*.tif"))
    test_geoms = _label_geoms(str(TEST_LABELS))
    print(f"test chips: {len(test_chip_paths)}, building polygons in test labels: {len(test_geoms)}")
    test_images, test_masks = _load_batch(test_chip_paths, test_geoms)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_images))
        preds = torch.argmax(logits, dim=1).numpy()

    import numpy as np

    pred_pos = preds == 1
    true_pos = test_masks == 1
    intersection = int(np.logical_and(pred_pos, true_pos).sum())
    union = int(np.logical_or(pred_pos, true_pos).sum())
    pred_sum = int(pred_pos.sum())
    true_sum = int(true_pos.sum())
    iou = intersection / union if union else 1.0
    precision = intersection / pred_sum if pred_sum else 0.0
    recall = intersection / true_sum if true_sum else 0.0
    print(f"Banepa test (in-domain): IoU={iou:.4f} precision={precision:.4f} recall={recall:.4f}")
    print(f"ground-truth building-pixel fraction in test set: {test_masks.mean():.4f}")


if __name__ == "__main__":
    main()
