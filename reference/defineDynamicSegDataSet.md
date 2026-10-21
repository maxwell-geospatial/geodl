# defineDynamicSegDataSet

Instantiate a subclass of torch::dataset() for geospatial semantic
segmentation using dynamically generated chips

## Usage

``` r
defineDynamicSegDataSet(
  chipsSF,
  chipSize = 512,
  cellSize = 2,
  normalize = FALSE,
  rescaleFactor = 1,
  mskRescale = 1,
  mskAdd = 0,
  bands = c(1, 2, 3),
  bMns = 1,
  bSDs = 1,
  doAugs = FALSE,
  maxAugs = 0,
  probVFlip = 0,
  probHFlip = 0,
  probRotate = 0,
  probBrightness = 0,
  probContrast = 0,
  probGamma = 0,
  probHue = 0,
  probSaturation = 0,
  brightFactor = c(0.8, 1.2),
  contrastFactor = c(0.8, 1.2),
  gammaFactor = c(0.8, 1.2, 1),
  hueFactor = c(-0.2, 0.2),
  saturationFactor = c(0.8, 1.2)
)

defineDynmamicSegDataSet(
  chipsSF,
  chipSize = 512,
  cellSize = 2,
  normalize = FALSE,
  rescaleFactor = 1,
  mskRescale = 1,
  mskAdd = 0,
  bands = c(1, 2, 3),
  bMns = 1,
  bSDs = 1,
  doAugs = FALSE,
  maxAugs = 0,
  probVFlip = 0,
  probHFlip = 0,
  probRotate = 0,
  probBrightness = 0,
  probContrast = 0,
  probGamma = 0,
  probHue = 0,
  probSaturation = 0,
  brightFactor = c(0.8, 1.2),
  contrastFactor = c(0.8, 1.2),
  gammaFactor = c(0.8, 1.2, 1),
  hueFactor = c(-0.2, 0.2),
  saturationFactor = c(0.8, 1.2)
)
```

## Arguments

- chipsSF:

  sf object created by makeDynamicChipsSF().

- chipSize:

  Size of desired image chips. Default is 512 (512x512 cells)

- cellSize:

  Cells size of input data. Default is 1 m.

- normalize:

  TRUE or FALSE. Whether to apply normalization. If FALSE, bMns and bSDs
  are ignored. Default is FALSE. If TRUE, you must provide bMns and
  bSDs.

- rescaleFactor:

  A rescaling factor to rescale the bands to 0 to 1. For example, this
  could be set to 255 to rescale 8-bit data. Default is 1 or no
  rescaling.

- mskRescale:

  Can be used to rescale binary masks that are not scaled from 0 to 1.
  For example, if masks are scaled from 0 and 255, you can divide by 255
  to obtain a 0 to 1 scale. Default is 1 or no rescaling.

- mskAdd:

  Value to add to mask class numeric codes. For example, if class
  indices start are 0, 1 can be added so that indices start at 1.
  Default is 0 (return original class codes). Note that several other
  functions in this package have a zeroStart parameter. If class codes
  start at 0, this argument should be set to TRUE. If they start at 1,
  this argument should be set to FALSE. The importance of this arises
  from the use of one-hot encoding internally, which requires that class
  indices start at 1.

- bands:

  Vector of bands to include. The default is to only include the first 3
  bands. If you want to use a different subset of bands, you must
  provide a vector of band indices here to override the default.

- bMns:

  Vector of band means. Length should be the same as the number of
  bands. Normalization is applied before any rescaling within the
  function.

- bSDs:

  Vector of band standard deviations. Length should be the same as the
  number of bands. Normalization is applied before any rescaling.

- doAugs:

  TRUE or FALSE. Whether or not to apply data augmentations to combat
  overfitting. If FALSE, all augmentations parameters are ignored. Data
  augmentations are generally only applied to the training set. Default
  is FALSE.

- maxAugs:

  0 to 7. Maximum number of random augmentations to apply. Default is 0
  or no augmentations. Must be changed if augmentations are desired.

- probVFlip:

  0 to 1. Probability of applying vertical flips. Default is 0 or no
  augmentations. Must be changed if augmentations are desired.

- probHFlip:

  0 to 1. Probability of applying horizontal flips. Default is 0 or no
  augmentations. Must be changed if augmentations are desired.

- probRotate:

  0 to 1. Probability of applying rotation by 0-, 90-, 180- or
  270-degrees. Default is 0 or no augmentations. Must be changed if
  augmentations are desired.

- probBrightness:

  0 to 1. Probability of applying brightness augmentation. Default is 0
  or no augmentations. Must be changed if augmentations are desired.

- probContrast:

  0 to 1. Probability of applying contrast augmentations. Default is 0
  or no augmentations. Must be changed if augmentations are desired.

- probGamma:

  0 to 1. Probability of applying gamma augmentations. Default is 0 or
  no augmentations. Must be changed if augmentations are desired.

- probHue:

  0 to 1. Probability of applying hue augmentations. Default is 0 or no
  augmentations. Must be changed if augmentations are desired. This is
  only applicable to RGB data.

- probSaturation:

  0 to 1. Probability of applying saturation augmentations. Default is 0
  or no augmentations. Must be changed if augmentations are desired.
  This is only applicable to RGB data.

- brightFactor:

  Vector of smallest and largest brightness adjustment factors. Random
  value will be selected between these extremes. The default is 0.8 to
  1.2. Can be any non negative number. For example, 0 gives a black
  image, 1 gives the original image, and 2 increases the brightness by a
  factor of 2.

- contrastFactor:

  Vector of smallest and largest contrast adjustment factors. Random
  value will be selected between these extremes. The default is 0.8 to
  1.2. Can be any non negative number. For example, 0 gives a solid gray
  image, 1 gives the original image, and 2 increases the contrast by a
  factor of 2.

- gammaFactor:

  Vector of smallest and largest gamma values and gain value for a total
  of 3 values. Random value will be selected between these extremes. The
  default gamma value range is 0.8 to 1.2 and the default gain is 1. The
  gain is not randomly altered, only the gamma. Non negative real
  number. A gamma larger than 1 makes the shadows darker while a gamma
  smaller than 1 makes dark regions lighter.

- hueFactor:

  Vector of smallest and largest hue adjustment factors. Random value
  will be selected between these extremes. The default is -0.2 to 0.2.
  Should be in range -0.5 to 0.5. 0.5 and -0.5 give complete reversal of
  hue channel in HSV space in positive and negative direction,
  respectively. 0 means no shift. Therefore, both -0.5 and 0.5 will give
  an image with complementary colors while 0 gives the original image.

- saturationFactor:

  Vector of smallest and largest saturation adjustment factors. Random
  value will be selected between these extremes. The default is 0.8 to
  1.2. For example, 0 will give a black-and-white image, 1 will give the
  original image, and 2 will enhance the saturation by a factor of 2.

## Value

A dataset object that can be provided to torch::dataloader().

## Details

This function instantiates a subclass of torch::dataset() that
dynmaically generates chips using the output from makeDynamicChipsSF().
Can also define random augmentations to combat overfitting. Note that
horizontal and vertical flips will affect the alignment of the image and
associated mask chips. As a result, the same augmentation will be
applied to both the image and the mask. Changes in brightness, contrast,
gamma, hue, and saturation will not be applied to the masks since
alignment is not impacted by these transformations. Predictor variables
are generated with three dimensions (channel/variable, width, height)
regardless of the number of channels/variables. Masks are generated as
three dimensional tensors (class index, width, height).

`defineDynmamicSegDataSet()` is a misspelled alias retained for backward
compatibility. Use `defineDynamicSegDataSet()` instead.
