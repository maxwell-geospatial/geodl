# defineResNetUNet

Define a UNet architecture for geospatial semantic segmentation with a
ResNet backbone.

## Usage

``` r
defineResNetUNet(
  inChn = 3,
  nCls = 3,
  backbone = "resnet34",
  pretrainedEncoder = TRUE,
  freezeEncoder = FALSE,
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

- backbone:

  ResNet variant to use as the encoder: "resnet18", "resnet34",
  "resnet50", "resnet101", or "resnet152". Default is "resnet34".

- pretrainedEncoder:

  TRUE or FALSE. Whether to initialize the encoder with pre-trained
  ImageNet weights. Default is TRUE.

- freezeEncoder:

  TRUE or FALSE. Whether to freeze the encoder during training. If TRUE,
  only the decoder is trained. Default is FALSE.

- avgImNetWeights:

  TRUE or FALSE. If three predictor variables are provided and ImageNet
  weights are used, whether to average the stem weights rather than use
  the originals. If the input has a number of channels other than 3, the
  stem weights are always averaged regardless of this argument. Default
  is FALSE.

- actFunc:

  Activation function used in the decoder (the ResNet encoder layers are
  not affected; they use ReLU internally). "relu" = rectified linear
  unit (ReLU); "lrelu" = leaky ReLU; "swish" = swish; "gelu" = GELU.
  Default is "relu".

- useAttn:

  TRUE or FALSE. Whether to add attention gates along the skip
  connections. Default is FALSE.

- useDS:

  TRUE or FALSE. Whether to use deep supervision. If TRUE, four
  predictions are made, one at each of the four largest decoder block
  resolutions, and the predictions are returned as a list object. If
  FALSE, only the final prediction at the original resolution is
  returned. Default is FALSE.

- dcChn:

  Vector of 5 integers defining the number of output feature maps for
  each of the 5 decoder blocks. Default is 256, 128, 64, 32, and 16.

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

ResNetUNet model instance as torch nn_module.

## Details

Define a UNet architecture with a torchvision ResNet encoder (ResNet-18,
-34, -50, -101, or -152), following the same design as
[`defineMobileUNet`](https://maxwell-geospatial.github.io/geodl/reference/defineMobileUNet.md):
5 encoder stages (the shallowest is an identity pass-through of the
input image, so the final decoder block skips from the raw input), a
bottleneck, and 5 decoder blocks with transpose-convolution upsampling.
Optional attention gates along the skip connections (`useAttn`) and deep
supervision (`useDS`) are available.

The encoder is tapped as: stem (`conv1 + bn1 + relu`, stride 2),
`maxpool + layer1` (stride 4), `layer2` (stride 8), `layer3` (stride
16), and `layer4` as the bottleneck (stride 32). Channel counts are
taken from the chosen backbone (64/64/128/256/512 for ResNet-18/34;
64/256/512/1024/2048 for ResNet-50/101/152).

If ImageNet weights are used and the number of input channels differs
from 3, or if `avgImNetWeights = TRUE`, the stem convolution weights are
averaged over the input-channel dimension and expanded to `inChn`
channels.
