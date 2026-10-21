# defineEfficientUNetB2

Define a UNet architecture for geospatial semantic segmentation with an
EfficientNet-B2 backbone.

## Usage

``` r
defineEfficientUNetB2(
  inChn = 3,
  nCls = 3,
  pretrainedEncoder = TRUE,
  freezeEncoder = FALSE,
  avgImNetWeights = FALSE,
  actFunc = "relu",
  useAttn = FALSE,
  useASPP = FALSE,
  useDS = FALSE,
  dcChn = c(256, 128, 64, 32, 16),
  btnChn = 256,
  dilRates = c(6, 12, 18),
  dilChn = c(256, 256, 256, 256),
  negative_slope = 0.01,
  stageLRs = NULL
)
```

## Arguments

- inChn:

  Number of input channels or predictor variables. Default is 3.

- nCls:

  Number of classes being differentiated. For a binary classification,
  this can be either 1 or 2. If 2, the problem is treated as a
  multiclass problem, and a multiclass loss metric should be used.
  Default is 3.

- pretrainedEncoder:

  TRUE or FALSE. Whether or not to initialize using pre-trained ImageNet
  weights for the EfficientNet-B2 encoder. Default is TRUE.

- freezeEncoder:

  TRUE or FALSE. Whether or not to freeze the encoder during training.
  The default is FALSE. If TRUE, only the decoder component is trained.

- avgImNetWeights:

  TRUE or FALSE. If three predictor variables are provided and ImageNet
  weights are used, whether or not to use the original weights or
  average them. If the input has more or fewer than 3 channels, the stem
  weights are always averaged regardless of this argument. Default is
  FALSE.

- actFunc:

  Defines the activation function used in the decoder and bottleneck
  (note that EfficientNet-B2 encoder layers are not impacted; they use
  SiLU internally). "relu" = rectified linear unit (ReLU); "lrelu" =
  leaky ReLU; "swish" = SiLU/swish; "gelu" = GELU. Default is "relu".

- useAttn:

  TRUE or FALSE. Whether to add attention gates along the skip
  connections. Default is FALSE.

- useASPP:

  TRUE or FALSE. Whether to replace the sequential bottleneck
  (EfficientNet-B2 stages 6-8) with an Atrous Spatial Pyramid Pooling
  (ASPP) module. If TRUE, the ASPP output channel count is set by
  `btnChn`. Default is FALSE.

- useDS:

  TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four
  predictions are made, one at each of the four largest decoder block
  resolutions, and the predictions are returned as a list object
  containing the 4 predictions. If FALSE, only the final prediction at
  the original resolution is returned. Default is FALSE.

- dcChn:

  Vector of 5 integers defining the number of output feature maps for
  each of the 5 decoder blocks. Default is 256, 128, 64, 32, and 16.

- btnChn:

  Number of output channels from the ASPP bottleneck when
  `useASPP = TRUE`. Ignored when `useASPP = FALSE` (the sequential
  bottleneck always outputs 352 channels). Default is 256.

- dilRates:

  Vector of 3 dilation rates for the ASPP module. Default is 6, 12, and
  18.

- dilChn:

  Vector of 4 channel counts for each ASPP branch. Default is 256 for
  each.

- negative_slope:

  If `actFunc = "lrelu"`, specifies the negative slope term. Default is
  0.01.

- stageLRs:

  Optional numeric vector of length 11 specifying a base learning rate
  for each stage: encoder stages e1, e2, e3, e4, e5, the bottleneck, and
  decoder stages d1, d2, d3, d4, d5 (in that order). Call
  `model$get_param_groups()` to obtain optimizer parameter groups with
  per-stage learning rates. Default is NULL (single learning rate).

## Value

EfficientUNetB2 model instance as torch nn_module.

## Details

Define a UNet architecture with an EfficientNet-B2 backbone or encoder.
The architecture has 5 encoder blocks (e1-e5), a bottleneck, and 5
decoder blocks (d1-d5). The final decoder block (d5) uses the original
input image as its skip connection, matching the design of
`defineMobileUNet`. The user can load ImageNet weights, apply averaged
ImageNet weights when the input channel count differs from 3, replace
the bottleneck with an ASPP module, add attention gates along the skip
connections, freeze the encoder, and enable deep supervision.
