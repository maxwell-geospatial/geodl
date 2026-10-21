# defineTerrainSeg

CNN-based semantic segmentation wrapper that generates LSPs from a DTM
and passes them to a user-supplied trainable segmentation model.

## Usage

``` r
defineTerrainSeg(
  segMod,
  cellSize = 1,
  spatDim = 640,
  tCrop = 64,
  doGP = FALSE,
  innerRadius = 2,
  outerRadius = 10,
  hsRadius = 50,
  smoothRadius = 11
)
```

## Arguments

- segMod:

  An already-instantiated trainable segmentation model (torch
  `nn_module`). Must accept the number of input channels produced by the
  LSP module: 6 when `doGP = FALSE`, 31 when `doGP = TRUE`.

- cellSize:

  Input resolution of DTM data in map units. Default is 1.

- spatDim:

  Spatial dimension (height = width) of the input chip in cells. Default
  is 640.

- tCrop:

  Number of rows and columns to crop from each side of the LSP output
  before passing to the segmentation model. This removes edge artefacts
  introduced by the convolution-based LSP calculations. Default is 64.

- doGP:

  Whether to compute Gaussian Pyramids of the DTM and derive LSPs at
  multiple scales. Default is FALSE. If FALSE, 6 LSPs are passed to the
  model. If TRUE, 31 LSPs are passed.

- innerRadius:

  Inner radius (cells) of the annulus window used for local TPI
  calculation. Default is 2.

- outerRadius:

  Outer radius (cells) of the annulus window used for local TPI
  calculation. Default is 10.

- hsRadius:

  Radius (cells) of the circular moving window for hillslope TPI
  calculation. Default is 50.

- smoothRadius:

  Radius (cells) of the circular moving window used to smooth the DTM
  before curvature calculations. Default is 11.

## Value

A `terrainSeg` `nn_module` wrapping the LSP pipeline and the supplied
segmentation model.

## Details

Define a CNN-based semantic segmentation model for landform extraction
or classification. The module generates land surface parameters (LSPs)
from the input DTM, crops the LSP tensor to remove edge artefacts, and
then passes the result to an externally instantiated trainable
segmentation model. The model assumes a single-band DTM of elevation
measurements as input.

When `doGP = FALSE` the LSP module produces 6 channels; when
`doGP = TRUE` it produces 31 channels. The segmentation model supplied
via `segMod` must be instantiated with a matching `inChn` value and with
an `inChn` that accounts for the spatial reduction caused by `tCrop`
(input chip size minus 2 \* tCrop).
