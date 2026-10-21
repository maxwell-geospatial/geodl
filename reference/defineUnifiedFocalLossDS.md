# defineUnifiedFocalLossDS

Define a loss for geospatial semantic segmentation using a modified
unified focal loss framework as a subclass of torch::nn_module() when
using deep supervision.

## Usage

``` r
defineUnifiedFocalLossDS(
  nCls = 3,
  cropFactorMsk = 0,
  cropFactorPred = 0,
  dsWghts = c(0.6, 0.2, 0.1, 0.1),
  weight_env = NULL,
  lambda = 0.5,
  gamma = 0.5,
  delta = 0.6,
  smooth = 1e-08,
  zeroStart = TRUE,
  clsWghtsDist = 1,
  clsWghtsReg = 1,
  useLogCosH = FALSE,
  useEdgeDist = FALSE,
  prioritizeEdge = TRUE,
  edgeMode = "threshold",
  distThreshold = 10,
  minWght = 0.5,
  maxWght = 1,
  applyDistWght = "both",
  labelSmoothing = 0,
  classAvgDist = FALSE,
  device = "cuda"
)
```

## Arguments

- nCls:

  Number of classes being differentiated.

- cropFactorMsk:

  Number of rows and columns of cells to not include for mask in
  assessment to minimize edge effects. Default is 0 or no cropping.

- cropFactorPred:

  Number of rows and columns of cells to not include for prediction in
  assessment to minimize edge effects. Default is 0 or no cropping.

- dsWghts:

  Vector of 4 weights applied to the losses at each spatial resolution.
  Used only when `weight_env` is `NULL`. The default is c(.6, .2, .1,
  .1) where larger weights are placed on the results at a higher spatial
  resolution.

- weight_env:

  Optional environment created by
  [`make_ds_weights()`](https://maxwell-geospatial.github.io/geodl/reference/make_ds_weights.md).
  When supplied, the loss reads `weight_env$values` on every forward
  pass so that
  [`callback_ds_weight_decay()`](https://maxwell-geospatial.github.io/geodl/reference/callback_ds_weight_decay.md)
  can decay the auxiliary weights during training without rebuilding the
  loss. Takes precedence over `dsWghts`. Default is `NULL`.

- lambda:

  Term used to control the relative weighting of the distribution- and
  region-based losses. Default is 0.5, or equal weighting between the
  losses. If lambda = 1, only the distribution- based loss is
  considered. If lambda = 0, only the region-based loss is considered.
  Values between 0.5 and 1 put more weight on the distribution-based
  loss while values between 0 and 0.5 put more weight on the
  region-based loss.

- gamma:

  Parameter that controls weighting applied to difficult-to-predict
  pixels (for distribution-based losses) or difficult-to-predict classes
  (for region-based losses). Smaller values increase the weight applied
  to difficult samples or classes. Default is 1, or no focal weighting
  is applied. Value must be less than or equal to 1 and larger than 0.

- delta:

  Parameter that controls the relative weightings of false positive and
  false negative errors for each class. Different weightings can be
  provided for each class. The default is 0.6, which results in
  prioritizing false negative errors relative to false positive errors.

- smooth:

  Smoothing factor to avoid divide-by-zero errors and provide numeric
  stability. Default is 1e-8. Recommend using the default.

- zeroStart:

  TRUE or FALSE. If class indices start at 0 as opposed to 1, this
  should be set to TRUE. This is required to implement one-hot encoding
  since R starts indexing at 1. Default is TRUE.

- clsWghtsDist:

  Vector of class weights for use in calculating a weighted version of
  the CE loss. Default is for all classes to be equally weighted.

- clsWghtsReg:

  Vector of class weights for use in calculating a weighted version of
  the region-based loss. Default is for all classes to be equally
  weighted.

- useLogCosH:

  TRUE or FALSE. Whether or not to apply a logCosH transformation to the
  region-based loss. Default is FALSE.

- useEdgeDist:

  TRUE or FALSE. Whether to apply distance-to-class-boundary weights
  during loss computation. When TRUE a `dist` tensor must be supplied to
  `forward`. Default is FALSE.

- prioritizeEdge:

  TRUE or FALSE. When `useEdgeDist = TRUE`, controls which pixels
  receive the higher weight. If TRUE (default), pixels near class
  boundaries receive `maxWght` and pixels far from boundaries receive
  `minWght`. If FALSE, the weighting is reversed.

- edgeMode:

  Character string specifying how the distance is converted to a
  per-pixel weight. One of `"threshold"`, `"linear"`, or `"gaussian"`.
  Default is `"threshold"`.

- distThreshold:

  Numeric. The distance value that defines the boundary of the
  high-weight zone. Default is 10.

- minWght:

  Numeric. The minimum per-pixel weight. Must be less than `maxWght`.
  Default is 0.5.

- maxWght:

  Numeric. The maximum per-pixel weight. Must be greater than `minWght`.
  Default is 1.0.

- applyDistWght:

  Character string controlling which loss component the edge weights are
  applied to. One of `"distribution"`, `"region"`, or `"both"`
  (default).

- labelSmoothing:

  Numeric value in \[0, 1). Amount of label smoothing applied to the
  cross-entropy component of the distribution loss. The focal modifier
  is derived from the true-class softmax probability and is not affected
  by smoothing. Default is 0 (no smoothing). See
  `defineUnifiedFocalLoss` for full details.

- classAvgDist:

  TRUE or FALSE. When TRUE, the distribution loss is computed as a
  weighted average of per-class focal cross-entropy means rather than a
  pixel-level weighted mean. See `defineUnifiedFocalLoss` for a full
  description. Default is FALSE.

- device:

  Define device being used for computation. Define using torch_device().

## Value

Loss metric for use in training process.

## Details

Implementation of modified version of the unified focal loss after:

Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal
loss: Generalising Dice and cross entropy-based losses to handle class
imbalanced medical image segmentation. Computerized Medical Imaging and
Graphics, 95, p.102026.

Modifications include (1) allowing users to define class weights for
both the distribution- based and region-based losses, (2) using class
weights as opposed to the symmetric and asymmetric methods implemented
by the authors, (3) including an option to apply a logcosh transform to
the region-based loss, (4) optionally incorporating
distance-to-class-boundary weights computed by `makeBoundaryDistance`,
(5) an option to compute the distribution loss as a weighted average of
per-class means rather than as a single pixel-level weighted mean, and
(6) optional label smoothing on the distribution loss.

This loss has three key hyperparameters that control its implementation.
Lambda controls the relative weight of the distribution- and
region-based losses. Default is 0.5, or equal weighting between the
losses is applied. If lambda = 1, only the distribution- based loss is
considered. If lambda = 0, only the region-based loss is considered.
Values between 0.5 and 1 put more weight on the distribution-based loss
while values between 0 and 0.5 put more weight on the region-based loss.

Gamma controls the application of focal loss and the application of
increased weight to difficult-to-predict pixels (for distribution-based
losses) or difficult-to-predict classes (region-based losses). Lower
gamma values put increased weight on difficult samples or classes. Using
a value of 1 equates to not using a focal adjustment.

The delta term controls the relative weight of false positive and false
negative errors for each class. The default is 0.6 for each class, which
results in placing a higher weight on false negative as opposed to false
positive errors relative to that class.

By adjusting the lambda, gamma, delta, and class weight terms, the user
can implement a variety of different loss metrics including cross
entropy loss, weighted cross entropy loss, focal cross entropy loss,
focal weighted cross entropy loss, Dice loss, focal Dice loss, Tversky
loss, and focal Tversky loss.

When `useEdgeDist = TRUE` a distance-to-class-boundary tensor must be
passed as the third argument to `forward(pred, target, dist)`. The same
distance tensor is applied to all decoder-output losses. See
`defineUnifiedFocalLoss` for a full description of the edge-distance
weighting options.
