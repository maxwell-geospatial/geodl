# defineUnifiedFocalLoss

Define a loss for semantic segmentation using a modified unified focal
loss framework as a subclass of torch::nn_module()

## Usage

``` r
defineUnifiedFocalLoss(
  nCls = 3,
  cropFactorMsk = 0,
  cropFactorPred = 0,
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
  boundaries (small distance) receive `maxWght` and pixels far from
  boundaries receive `minWght`. If FALSE, the weighting is reversed so
  that non-edge pixels are emphasised.

- edgeMode:

  Character string specifying how the distance is converted to a
  per-pixel weight. One of `"threshold"`, `"linear"`, or `"gaussian"`.
  Default is `"threshold"`. See the Details section for a full
  description of each mode.

- distThreshold:

  Numeric. The distance value (in map units or pixel units, matching the
  distance raster) that defines the boundary of the high-weight zone.
  Default is 10.

- minWght:

  Numeric. The minimum per-pixel weight applied to loss contributions.
  Must be less than `maxWght`. Default is 0.5.

- maxWght:

  Numeric. The maximum per-pixel weight applied to loss contributions.
  Must be greater than `minWght`. Default is 1.0.

- applyDistWght:

  Character string controlling which loss component the edge weights are
  applied to. One of `"distribution"` (distribution-based/CE loss only),
  `"region"` (region-based/Tversky loss only), or `"both"` (default).

- labelSmoothing:

  Numeric value in \[0, 1). When greater than 0, this amount of
  probability mass is spread uniformly across all classes when computing
  the cross-entropy, preventing the model from becoming overconfident.
  The focal modifier is always derived from the true-class softmax
  probability gathered directly from the network output, so focal
  weighting is not distorted by the smoothed targets. Default is 0 (no
  smoothing).

- classAvgDist:

  TRUE or FALSE. When FALSE (default), the distribution loss is computed
  as a pixel-level weighted mean of the focal cross-entropy, with
  `clsWghtsDist` rescaling each pixel's contribution by its true class
  weight. When TRUE, the focal cross-entropy is first averaged within
  each class and the resulting per-class means are then combined as a
  weighted average using `clsWghtsDist`. The per-class approach mirrors
  the macro-averaging philosophy already used by the region-based loss
  and can give better gradient signal to under-represented classes under
  severe class imbalance. Note that if a class has no pixels in a given
  batch its contribution is effectively zero for that step, consistent
  with the behaviour of the region-based loss.

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

When `labelSmoothing > 0` the cross-entropy is computed against soft
targets that spread `labelSmoothing` probability mass uniformly across
all classes, reducing the model's incentive to become overconfident. The
focal modifier is kept anchored to the true-class softmax probability
gathered directly from the network output, so the focal weighting
continues to reflect genuine model confidence rather than being
distorted by the smoothed targets. Label smoothing is applied only to
the distribution loss; the region-based (Tversky) loss is not affected.

When `useEdgeDist = TRUE` a distance-to-class-boundary tensor must be
passed as the third argument to `forward(pred, target, dist)`. The
tensor should have shape (N, 1, H, W) and contain the per-pixel
distances produced by `makeBoundaryDistance`, stacked into a batch.
Pixels at a class boundary have distance 0; distance increases away from
boundaries.

Three edge-weighting modes are available via `edgeMode`:

- `"threshold"`: pixels within `distThreshold` receive one weight;
  pixels beyond receive the other. When `prioritizeEdge = TRUE` the
  inside weight is `maxWght` and the outside weight is `minWght`; the
  relationship is reversed when `prioritizeEdge = FALSE`.

- `"linear"`: weight changes linearly from `maxWght` at distance 0 to
  `minWght` at `distThreshold` (when `prioritizeEdge = TRUE`), or from
  `minWght` to `maxWght` (when `prioritizeEdge = FALSE`). Beyond
  `distThreshold` the weight is clamped to the far-end value.

- `"gaussian"`: weight follows a Gaussian curve centred at distance 0.
  The curve uses sigma = distThreshold / 3 so that the weight is
  approximately `minWght` at `distThreshold` and falls to exactly
  `minWght` beyond it. Direction is controlled by `prioritizeEdge` as
  above.
