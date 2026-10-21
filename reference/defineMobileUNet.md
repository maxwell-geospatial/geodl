# defineMobileUNet

Define a UNet architecture for geospatial semantic segmentation with a
MobileNet-v2 backbone.

## Usage

``` r
defineMobileUNet(
  inChn = 3,
  nCls = 3,
  pretrainedEncoder = TRUE,
  freezeEncoder = TRUE,
  avgImNetWeights = FALSE,
  actFunc = "relu",
  useAttn = FALSE,
  useDS = FALSE,
  dcChn = c(256, 128, 64, 32, 16),
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

  TRUE or FALSE. Whether or not to initialized using pre-trained
  ImageNet weights for the MobileNet-v2 encoder. Default is TRUE.

- freezeEncoder:

  TRUE or FALSE. Whether or not to freeze the encoder during training. T
  he default is TRUE. If TRUE, only the decoder component is trained.

- avgImNetWeights:

  TRUE or FALSE. If three predictor variables are provided and ImageNet
  weights are used, whether or not to use the original weights or
  average them. Default is FALSE.

- actFunc:

  Defines activation function to use throughout the network (note that
  MobileNet-v2 layers are not impacted). "relu" = rectified linear unit
  (ReLU); "lrelu" = leaky ReLU; "swish" = swish; "gelu" = GELU. Default
  is "relu".

- useAttn:

  TRUE or FALSE. Whether to add attention gates along the skip
  connections. Default is FALSE or no attention gates are added.

- useDS:

  TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four
  predictions are made, one at each of the four largest decoder block
  resolutions, and the predictions are returned as a list object
  containing the 4 predictions. If FALSE, only the final prediction at
  the original resolution is returned. Default is FALSE or deep
  supervision is not implemented.

- dcChn:

  Vector of 4 integers defining the number of output feature maps for
  each of the 4 decoder blocks. Default is 128, 64, 32, and 16.

- negative_slope:

  If actFunc = "lrelu", specifies the negative slope term to use.
  Default is 0.01.

- stageLRs:

  Optional numeric vector of length 11 specifying a base learning rate
  for each stage: encoder stages e1, e2, e3, e4, e5, the bottleneck, and
  decoder stages d1, d2, d3, d4, d5 (in that order). Call
  `model$get_param_groups()` to obtain optimizer parameter groups with
  per-stage learning rates. Default is NULL (single learning rate).

## Value

ModileUNet model instance as torch nn_module

## Details

Define a UNet architecture with a MobileNet-v2 backbone or encoder. This
UNet implementation was inspired by a blog post by Sigrid Keydana
available
[here](https://blogs.rstudio.com/ai/posts/2021-10-29-segmentation-torch-android/).
This architecture has 6 blocks in the encoder (including the bottleneck)
and 5 blocks in the decoder. The user is able to implement deep
supervision (useDS = TRUE) and attention gates along the skip
connections (useAttn = TRUE). If ImageNet weights are used and more then
three predictor variables are provided, ImageNet weights in the layer of
the encoder block are averaged. If three channels or predictor variables
are provided, the user can specify to user the ImageNet weights or
average them.
