# defineUNet

Define a UNet architecture for geospatial semantic segmentation.

## Usage

``` r
defineUNet(
  inChn = 3,
  nCls = 3,
  actFunc = "relu",
  encBlocks = "double",
  btnBlock = "double",
  decBlocks = "double",
  useAttn = FALSE,
  useSE = FALSE,
  useCBAM = FALSE,
  useRes = FALSE,
  useASPP = FALSE,
  useDS = FALSE,
  enChn = c(16, 32, 64, 128),
  dcChn = c(128, 64, 32, 16),
  btnChn = 256,
  dilRates = c(6, 12, 18),
  dilChn = c(256, 256, 256, 256),
  negative_slope = 0.01,
  seRatio = 8,
  cbamRatio = 8,
  cbamKernelSize = 7,
  expRatio = 4,
  stageLRs = NULL
)
```

## Arguments

- inChn:

  Number of channels, bands, or predictor variables in the input image
  or raster data. Default is 3.

- nCls:

  Number of classes being differentiated. For a binary classification,
  this can be either 1 or 2. If 2, the problem is treated as a
  multiclass problem, and a multiclass loss metric should be used.
  Default is 3.

- actFunc:

  Defines activation function to use throughout the network for the
  non-ConvNeXt blocks. "relu" = rectified linear unit (ReLU); "lrelu" =
  leaky ReLU; "swish" = swish; "gelu" = GELU. Default is "relu".

- encBlocks:

  Block implementation for the four encoder blocks. Either a single
  value applied to all four blocks or a length-4 character vector, with
  entries "double", "dws", or "convnext". Default is "double".

- btnBlock:

  Block implementation for the bottleneck: "double", "dws", "convnext",
  or "aspp". Default is "double".

- decBlocks:

  Block implementation for the four decoder blocks. Either a single
  value applied to all four blocks or a length-4 character vector, with
  entries "double", "dws", or "convnext". Default is "double".

- useAttn:

  TRUE or FALSE. Whether to add attention gates along the skip
  connections. Default is FALSE or no attention gates are added. Ignored
  (forced FALSE) when `useCBAM = TRUE`.

- useSE:

  TRUE or FALSE. Whether or not to include squeeze and excitation
  modules in the encoder. Default is FALSE or no squeeze and excitation
  modules are used. Ignored (forced FALSE) when `useCBAM = TRUE`.

- useCBAM:

  TRUE or FALSE. Whether to apply a Convolutional Block Attention Module
  (CBAM) to each encoder skip connection before it is concatenated with
  the upsampled decoder features. When TRUE, `useSE` and `useAttn` are
  forced to FALSE. Default is FALSE.

- useRes:

  TRUE or FALSE. Whether to include residual connections in the "double"
  and "dws" encoder, decoder, and bottleneck blocks and in the ASPP
  bottleneck. Has no effect on "convnext" blocks (always residual).
  Default is FALSE or no residual connections are included.

- useASPP:

  Deprecated. Use `btnBlock = "aspp"` instead. If TRUE and `btnBlock` is
  left at its default, the bottleneck is set to "aspp". Default is
  FALSE.

- useDS:

  TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four
  predictions are made, one at each decoder block resolution, and the
  predictions are returned as a list object containing the 4
  predictions. If FALSE, only the final prediction at the original
  resolution is returned. Default is FALSE or deep supervision is not
  implemented.

- enChn:

  Vector of 4 integers defining the number of output feature maps for
  each of the four encoder blocks. Default is 16, 32, 64, and 128.

- dcChn:

  Vector of 4 integers defining the number of output feature maps for
  each of the 4 decoder blocks. Default is 128, 64, 32, and 16.

- btnChn:

  Number of output feature maps from the bottleneck block. Default is
  256.

- dilRates:

  Vector of 3 values specifying the dilation rates used in the ASPP
  module. Default is 6, 12, and 18.

- dilChn:

  Vector of 4 values specifying the number of channels to produce at
  each dilation rate within the ASPP module. Default is 256 for each
  dilation rate.

- negative_slope:

  If actFunc = "lrelu", specifies the negative slope term to use.
  Default is 0.01.

- seRatio:

  Ratio to use in the squeeze and excitation modules. The default is 8.

- cbamRatio:

  Channel reduction ratio used in the CBAM channel attention MLP.
  Default is 8.

- cbamKernelSize:

  Kernel size for the CBAM spatial attention convolution. Must be odd.
  Default is 7.

- expRatio:

  Inverted-bottleneck expansion ratio for "convnext" blocks. Default is
  4.

- stageLRs:

  Optional numeric vector of length 9 specifying a base learning rate
  for each stage of the network: encoder stages e1, e2, e3, e4 (each
  paired with its squeeze-and-excitation or CBAM module when enabled),
  the bottleneck, and decoder stages d1, d2, d3, d4 (in that order).
  When provided, call `model$get_param_groups()` to obtain a list of
  optimizer parameter groups with per-stage learning rates. Default is
  NULL (single learning rate).

## Value

Unet model instance as torch nn_module

## Details

Define a UNet architecture with 4 blocks in the encoder, a bottleneck
block, and 4 blocks in the decoder. UNet can accept a variable number of
input channels, and the user can define the number of feature maps
produced in each encoder and decoder block and the bottleneck.

Each encoder block, each decoder block, and the bottleneck can
independently use one of several block implementations via `encBlocks`,
`decBlocks`, and `btnBlock`:

- `"double"` — the standard double 3x3 convolution block (default
  everywhere). Becomes a residual double convolution block when
  `useRes = TRUE`.

- `"dws"` — a double convolution block whose second 3x3 convolution is
  replaced with a depthwise separable convolution (`dwsDoubleConv`).
  Becomes a residual variant (`dwsDoubleConvR`) when `useRes = TRUE`.

- `"convnext"` — a ConvNeXt-style block (depthwise 7x7, LayerNorm, 1x1
  inverted-bottleneck expansion, GELU, 1x1 projection, residual;
  `doubleConvBlkCNX`). Always residual and always GELU, so `useRes` and
  `actFunc` do not affect these blocks.

- `"aspp"` — (`btnBlock` only) a modified atrous spatial pyramid pooling
  module for multi-scale context.

Users can also choose to (1) replace all ReLU activation functions with
leaky ReLU, swish, or GELU in the non-ConvNeXt blocks, (2) implement
attention gates along the skip connections (`useAttn`), (3) implement
squeeze-and-excitation modules within the encoder (`useSE`), (4) apply
Convolutional Block Attention Modules (CBAM) to the skip connections
(`useCBAM`), (5) add residual connections within the "double" blocks and
ASPP bottleneck (`useRes`), and/or (6) implement deep supervision using
predictions generated at each stage in the decoder (`useDS`). Because
CBAM already performs both channel and spatial recalibration, setting
`useCBAM = TRUE` forces `useSE` and `useAttn` to FALSE.
