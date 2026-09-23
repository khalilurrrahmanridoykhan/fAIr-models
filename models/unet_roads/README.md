# U-Net Roads (ResNet-34 encoder)

A U-Net for binary road segmentation with a torchvision ResNet-34 encoder,
initialised from ImageNet weights, finetuned end to end. It is the road
counterpart to `unet_buildings`, sharing the same architecture, training loop,
and ONNX export; the road-specific work is entirely in how OSM's road data is
turned into training masks and how predictions are turned back into road lines.

## Task

Binary semantic segmentation. The model reads a 3-band RGB chip, classifies every
pixel as road or background, and returns georeferenced road centrelines as
GeoJSON LineStrings.

## Why a segmentation model outputs lines

OpenStreetMap represents roads as centrelines, not areas, so two translations
happen at the boundary of an otherwise ordinary segmentation model. During
training, each centreline is buffered by a width looked up from its OSM
`highway` tag (a footway and a trunk road are not the same width) before being
rasterised into a training mask -- the model learns "plausible road area", not a
surveyed carriageway width. During inference, the predicted road-area mask is
skeletonised back down to a one-pixel-wide centreline and vectorised into
LineStrings, using a minimum-spanning-tree pass over each connected
skeleton component to recover simple branching paths without a full
topological road-graph library.

## Inputs and outputs

| Stage | Shape | Notes |
| ----- | ----- | ----- |
| Input | `[batch, 3, 256, 256]` float32 | RGB chip normalised with ImageNet mean/std |
| Output | `[batch, 2, 256, 256]` float32 | Background and road class logits |
| Prediction | GeoJSON | Road centrelines (LineStrings) in EPSG:4326 |

## Training

`split_dataset` groups chips into spatial blocks by their OAM tile coordinates
and holds out whole blocks for validation, identical to `unet_buildings`.
`train_model` buffers the training labels' road centrelines into a raster mask
sized by each line's `highway` tag, then trains the full network end to end
with Adam and cross-entropy loss. Road pixels are typically a small minority of
a chip -- far more imbalanced than buildings usually are -- so the loss weights
the road class by its own inverse frequency in the current training batch, capped
by the `max_class_weight` hyperparameter; without this, training on a sparse
project collapses to predicting no road at all. `evaluate_model` reports
road-class pixel IoU, precision, and recall on the held-out validation chips.

## Limitations

The width-by-tag lookup is a coarse approximation, not a surveyed carriageway
width, so predicted road extent should not be read as an exact width estimate.
The centreline-recovery step (skeletonise, then minimum-spanning-tree per
connected component) handles simple branching reasonably but is not a full
road-network topology extractor; complex interchanges or tightly parallel roads
in the same chip can produce noisy or merged lines. Like `unet_buildings`, this
model has not been benchmarked against a held-out reference dataset, and a
checkpoint finetuned on one region's imagery should not be expected to
generalise as a universal model elsewhere without refinetuning.

## Citation

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks
for Biomedical Image Segmentation. https://doi.org/10.48550/arXiv.1505.04597

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image
Recognition. https://doi.org/10.48550/arXiv.1512.03385

## License

Apache-2.0.
