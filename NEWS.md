# geodl (development)

* `defineUNet()` now subsumes the former `defineCBAMUNet()` and `defineConvNeXtUNet()`
  variants, which have been removed:
  * New `useCBAM` argument applies a Convolutional Block Attention Module (CBAM) to each
    encoder skip connection. When `useCBAM = TRUE`, `useSE` and `useAttn` are forced to FALSE.
  * New `encBlocks`, `decBlocks`, and `btnBlock` arguments let each encoder block, each
    decoder block, and the bottleneck independently use a standard double convolution block
    (`"double"`, the default), a depthwise-separable double convolution (`"dws"`), or a
    ConvNeXt-style block (`"convnext"`). `btnBlock` also accepts `"aspp"`.
  * `useASPP` is deprecated in favor of `btnBlock = "aspp"`.
* Added `dwsDoubleConv()`: a double convolution block whose second 3x3 convolution is
  replaced with a depthwise separable convolution.
* Added `dwsDoubleConvR()`, the residual counterpart to `dwsDoubleConv()`. `useRes = TRUE`
  in `defineUNet()` and `defineDLab()` now also applies to `"dws"` encoder/decoder/bottleneck
  blocks (previously it only affected `"double"` blocks; `"convnext"` blocks remain always
  residual regardless of `useRes`).
* Added `defineResNetUNet()`: a UNet with a torchvision ResNet encoder (ResNet-18, -34,
  -50, -101, or -152, selected via `backbone`), following the `defineMobileUNet()` design
  with optional ImageNet pre-training, stem-weight averaging for non-RGB inputs, attention
  gates, deep supervision, encoder freezing, and per-stage learning rates.
* Renamed `defineDeepLabV3Plus()` to `defineDLab()` (the encoder is a plain max-pooled
  CNN rather than an atrous backbone, so the DeepLabv3+ name was misleading). It gains
  `encBlocks` and `decBlock` arguments for choosing `"double"` / `"dws"` / `"convnext"`
  blocks per encoder block and for the decoder block, plus an `expRatio` argument for the
  ConvNeXt blocks.
* `defineUNet3p()`:
  * The full-scale skip connections now upsample with bilinear interpolation instead of
    transpose convolutions, matching the original UNet3+ and reducing the parameter count.
  * New `skipProj` argument (`"conv3x3"`, the default, or `"conv1x1"`) selects the
    per-branch channel-unification convolution.
  * New `encBlocks` and `btnBlock` arguments choose `"double"` / `"dws"` / `"convnext"`
    blocks per encoder block and for the bottleneck (`btnBlock` also accepts `"aspp"`),
    plus an `expRatio` argument for the ConvNeXt blocks. `useASPP` is deprecated in favor
    of `btnBlock = "aspp"`.
  * New `useRes` argument adds residual connections to the `"double"`/`"dws"` encoder
    blocks, the `"double"`/`"dws"`/`"aspp"` bottleneck, and the four per-node decoder
    feature-aggregation blocks (`d1Blk`-`d4Blk`, using a parameter-free identity skip
    since their input and output channel counts always match). Note this changes the
    default `btnBlock = "aspp"` bottleneck from always-residual (`asppBlkR`) to
    residual only when `useRes = TRUE`, matching `defineUNet()`/`defineDLab()`.

# geodl 0.3.3

* Updated torch-dependent examples so they are skipped when the torch runtime
  dependencies, including Lantern, are unavailable.
* Fixed example typo in `defineUnifiedFocalLoss()` where `pred2` was used twice.

# geodl 0.3.0

* Support for three model architectures: UNet, UNet with MobileNetv2 encoder, and UNet3+
* UNet with MobileNetv2 encoder is no longer limited to three input predictor variables
* Assessment and prediction functions now expect a nn_module object as opposed to a luz fitted object
* Dynamically generate chips during training process as opposed to saving them to disk beforehand (still experimental)
* Ignore outer rows and columns of cells when calculating loss or assessment metrics if desired
* Use R torch to calculate several different land surface parameters (LSPs) from a digital terrain model: slope, hillshade, aspect, northwardness, eastwardness, transformed solar radiation aspect index (TRASP), site exposure index (SEI), topographic position index (TPI), and surface curvatures (mean, profile, and planform)
* Calculate three-band terrain visualization raster grid from a DTM using torch or terra
* New specialized model for extracting geomorphic features from digital terrain models (DTMs)
* New function to count the number of trainable parameters in a model
* Fixed issue with chip generation pipeline that caused some chips with NA cells to be written
* Updated atrous spatial pyramid pooling (ASPP) module to align with the version used within DeepLabv3+

# geodl 0.2.0

* Initial CRAN submission.
