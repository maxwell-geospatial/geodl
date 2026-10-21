# defineUNet3p

Define a UNet3+ architecture for use in a luz training loop.

## Usage

``` r
defineUNet3p(
  inChn = 3,
  nCls = 2,
  actFunc = "lrelu",
  encBlocks = "double",
  btnBlock = "double",
  skipProj = "conv3x3",
  useRes = FALSE,
  enChn = c(16, 32, 64, 128),
  outChn = 64,
  btnChn = 256,
  useASPP = FALSE,
  dilRates = c(6, 12, 18),
  dilChn = c(256, 256, 256, 256),
  negative_slope = 0.01,
  expRatio = 4,
  useDS = FALSE,
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

  Activation function to use throughout the non-ConvNeXt blocks. "relu"
  = ReLU; "lrelu" = leaky ReLU; "swish" = SiLU/Swish; "gelu" = GELU.
  Default is "lrelu".

- encBlocks:

  Block implementation for the four encoder blocks. Either a single
  value applied to all four blocks or a length-4 character vector, with
  entries "double", "dws", or "convnext". Default is "double".

- btnBlock:

  Block implementation for the bottleneck: "double", "dws", "convnext",
  or "aspp". Default is "double".

- skipProj:

  Channel-unification convolution used on each branch of the full-scale
  skip connections: "conv3x3" (original UNet3+) or "conv1x1" (lighter).
  Default is "conv3x3".

- useRes:

  TRUE or FALSE. Whether to include residual connections in the "double"
  and "dws" encoder blocks, the "double"/"dws"/"aspp" bottleneck, and
  the four per-node decoder feature-aggregation blocks
  (`d1Blk`-`d4Blk`). Has no effect on "convnext" blocks (always
  residual). Default is FALSE.

- enChn:

  Vector of 4 integers defining the number of output feature maps for
  each of the four encoder blocks. Default is 16, 32, 64, and 128.

- outChn:

  Number of channels each skip-connection branch is projected to before
  concatenation. Each decoder node therefore operates on 5 \* outChn
  channels. Default is 64.

- btnChn:

  Number of output feature maps from the bottleneck block. Default is
  256.

- useASPP:

  Deprecated. Use `btnBlock = "aspp"` instead. If TRUE and `btnBlock` is
  left at its default, the bottleneck is set to "aspp". Default is
  FALSE.

- dilRates:

  Vector of 3 values specifying the dilation rates used in the ASPP
  module. Default is 6, 12, and 18.

- dilChn:

  Vector of 4 values specifying the number of channels to produce at
  each dilation rate within the ASPP module. Default is 256 for each
  dilation rate.

- negative_slope:

  Specifies the negative slope term for leaky ReLU activation. Default
  is 0.01.

- expRatio:

  Inverted-bottleneck expansion ratio for "convnext" blocks. Default is
  4.

- useDS:

  TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four
  predictions are made, one at each decoder block resolution, and the
  predictions are returned as a list object containing the 4
  predictions. If FALSE, only the final prediction at the original
  resolution is returned. Default is FALSE or deep supervision is not
  implemented.

- stageLRs:

  Optional numeric vector of length 9 specifying a base learning rate
  for each stage: encoder stages encoder1, encoder2, encoder3, encoder4
  (including their cross-scale projections), the bottleneck (including
  its decoder projections), and decoder stages d1, d2, d3, d4 (in that
  order). Call `model$get_param_groups()` to obtain optimizer parameter
  groups. Default is NULL (single learning rate).

## Value

UNet3+ model using nn_module().

## Details

Define a UNet3+-like architecture with full-scale skip connections. The
user can specify the number of output feature maps from each encoder
block, the per-branch channel count used in the full-scale skip
connections (`outChn`), and the bottleneck. Deep supervision can also be
implemented. A variable number of input predictor variables and output
classes can be defined.

As in the original UNet3+, downsampling within the full-scale skip
connections uses non-overlapping max pooling and upsampling uses
bilinear interpolation (there are no transpose convolutions). Each
resampled branch is passed through a channel-unification convolution
before the branches are concatenated; `skipProj` selects a 3x3
convolution ("conv3x3", the original UNet3+ design) or a lighter 1x1
convolution ("conv1x1"). The per-node feature aggregation block that
follows each concatenation is always a 3x3 convolution.

Each of the four encoder blocks and the bottleneck can independently use
one of three block implementations via `encBlocks` and `btnBlock`,
matching
[`defineUNet`](https://maxwell-geospatial.github.io/geodl/reference/defineUNet.md):

- `"double"` - the standard double 3x3 convolution block (default).
  Becomes a residual double convolution block when `useRes = TRUE`.

- `"dws"` - a double convolution block whose second 3x3 convolution is
  replaced with a depthwise separable convolution (`dwsDoubleConv`).
  Becomes a residual variant (`dwsDoubleConvR`) when `useRes = TRUE`.

- `"convnext"` - a ConvNeXt-style block (`doubleConvBlkCNX`); always
  residual and always GELU, so `useRes` and `actFunc` do not affect
  these blocks.

`btnBlock` additionally accepts `"aspp"` for an atrous spatial pyramid
pooling bottleneck, which also respects `useRes` (`asppBlk` vs.
`asppBlkR`). The four per-node feature-aggregation blocks (`d1Blk`
through `d4Blk`) likewise become residual when `useRes = TRUE`; since
their input and output channel counts are always equal, the residual
skip there is a parameter-free identity.

The architecture was inspired by:

Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X.,
Chen, Y.W. and Wu, J., 2020, May. Unet 3+: A full-scale connected unet
for medical image segmentation. In ICASSP 2020-2020 IEEE international
conference on acoustics, speech and signal processing (ICASSP) (pp.
1055-1059). IEEE.
