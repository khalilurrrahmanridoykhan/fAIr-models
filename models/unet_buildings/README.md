# U-Net Buildings (ResNet-34 encoder)

A U-Net for binary building segmentation built on a torchvision ResNet-34 encoder.
The encoder starts from standard ImageNet weights and, unlike `dinov3s_buildings`,
every layer of the network (encoder and decoder alike) is updated during
finetuning rather than only a decoder head. It targets projects that have enough
labelled OAM chips to make full finetuning worthwhile, and it depends on nothing
beyond PyTorch and torchvision, so its training code, ONNX export, and inference
path are all plain, auditable, self-contained code rather than a wrapped external
package.

## Task

Binary semantic segmentation. The model reads a 3-band RGB chip, classifies every
pixel as building or background, and vectorises the building pixels into GeoJSON
polygons. Any dataset of RGB chips with matching building labels can train it,
though the default hyperparameters assume the platform's standard 256x256 chips.

## Architecture

The encoder is a torchvision `resnet34` stem and four residual stages. The decoder
mirrors it with four transposed-convolution blocks, each concatenating the
matching encoder stage as a skip connection before two 3x3 convolutions, followed
by a final upsampling head back to input resolution. This is the classic U-Net
skip-connection pattern applied to a ResNet encoder rather than the original
paper's plain convolutional stack, which in practice converges faster and needs
fewer labelled chips because the encoder already knows general image features.

## Pretrained source

The encoder is initialised from torchvision's ResNet-34 weights trained on
ImageNet-1k (`resnet34-b627a593.pth`, the standard torchvision v1 checkpoint). The
decoder always starts from a fresh random initialisation, since no pretrained
decoder for this exact architecture and label taxonomy exists. Both weights are
downloaded directly from `download.pytorch.org` at train time, so the model
adapts freely to whatever building-labelled dataset a project provides.

## Training

`split_dataset` groups chips into spatial blocks by their OAM tile coordinates and
holds out whole blocks for validation, so nearby chips never leak between the
train and validation sets. `train_model` burns the GeoJSON building polygons onto
each chip's raster grid, then trains the full network end to end with Adam and
cross-entropy loss. `evaluate_model` reports building-class IoU, precision, and
recall on the held-out validation chips. `export_onnx` traces the trained network
to a single self-contained ONNX file.

## Limitations

Full finetuning needs meaningfully more labelled chips than a frozen-encoder
model to avoid overfitting, so on very small datasets (a few dozen chips or
fewer) `dinov3s_buildings` is likely to generalise better. The model has not been
benchmarked against a held-out reference dataset such as
`hotosm/vhr-building-segmentation`; per-project results should be checked before
relying on its predictions for mapping decisions. Like any RGB segmentation
model, it will underperform on chips with atypical roof materials, dense
vegetation occlusion, or resolutions well outside the ~30cm range it expects.

## Citation

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks
for Biomedical Image Segmentation. https://doi.org/10.48550/arXiv.1505.04597

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image
Recognition. https://doi.org/10.48550/arXiv.1512.03385

## License

Apache-2.0.
