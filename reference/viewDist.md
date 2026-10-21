# viewDist

Visualise a mini-batch of image chips, masks, distance-to-boundary
grids, and the resulting edge-weight surface for a DataLoader that
supplies distance grids.

## Usage

``` r
viewDist(
  dataLoader,
  nCols = 3,
  r = 1,
  g = 2,
  b = 3,
  cCodes,
  cNames,
  cColors,
  edgeMode = "gaussian",
  distThreshold = 10,
  minWght = 0.5,
  maxWght = 1,
  prioritizeEdge = TRUE,
  padding = 10
)
```

## Arguments

- dataLoader:

  Instantiated DataLoader created with
  [`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html).
  The batch must contain named elements `image`, `mask`, and `dist`.

- nCols:

  Number of chip columns in each image grid. Default is 3.

- r:

  Index of the image channel mapped to the red display channel. Default
  is 1. For single-band imagery, assign the same index to `r`, `g`, and
  `b`.

- g:

  Index of the image channel mapped to the green display channel.
  Default is 2.

- b:

  Index of the image channel mapped to the blue display channel. Default
  is 3.

- cCodes:

  Integer class codes as a vector, one per class.

- cNames:

  Character vector of class names, in the same order as `cCodes`.

- cColors:

  Character vector of display colours for the mask panel, in the same
  order as `cCodes`.

- edgeMode:

  Character string specifying how distances are converted to weights.
  One of `"threshold"`, `"linear"`, or `"gaussian"` (default). Must
  match the `edgeMode` used in `defineUnifiedFocalLoss`.

- distThreshold:

  Positive numeric. The distance value (in the same units as the
  distance raster) that defines the boundary of the high-weight zone.
  Default is 10.

- minWght:

  Numeric. Minimum per-pixel weight. Must be less than `maxWght`.
  Default is 0.5.

- maxWght:

  Numeric. Maximum per-pixel weight. Must be greater than `minWght`.
  Default is 1.0.

- prioritizeEdge:

  Logical. When `TRUE` (default), pixels near class boundaries (small
  distance) receive `maxWght` and pixels far from boundaries receive
  `minWght`. When `FALSE`, the relationship is reversed.

- padding:

  Integer. Number of pixels of padding between chips in the grid.
  Default is 10.

## Value

Called for its side effect (plots to the active graphics device).
Returns `invisible(NULL)`.

## Details

Plots four panels stacked vertically:

1.  RGB image chips arranged in a grid.

2.  Corresponding class masks.

3.  Distance-to-class-boundary grids produced by `makeBoundaryDistance`.

4.  The per-pixel edge-weight surface derived from those distances,
    computed with the same formula used by `defineUnifiedFocalLoss` when
    `useEdgeDist = TRUE`.

The weight surface mirrors the three modes available in
`defineUnifiedFocalLoss`:

- `"threshold"`:

  Pixels within `distThreshold` receive one weight (`maxWght` when
  `prioritizeEdge = TRUE`); all others receive the opposite weight.

- `"linear"`:

  Weight decreases linearly from `maxWght` at distance 0 to `minWght` at
  `distThreshold` (when `prioritizeEdge = TRUE`), clamped beyond the
  threshold.

- `"gaussian"`:

  Weight follows a Gaussian curve centred at distance 0 with sigma =
  distThreshold / 3, so the weight equals approximately `minWght` at the
  threshold and is clamped to `minWght` beyond it (when
  `prioritizeEdge = TRUE`).

The DataLoader must return batches with three named elements: `image`,
`mask`, and `dist`. The `dist` element should be a tensor of shape (N,
1, H, W) containing the per-pixel distances produced by
`makeBoundaryDistance`.

## Examples

``` r
if (FALSE) { # \dontrun{
viewDist(
  dataLoader    = trainDL,
  nCols         = 4,
  r = 1, g = 2, b = 3,
  cCodes        = c(1, 2, 3, 4, 5),
  cNames        = c("Background", "Forest", "Water", "Urban", "Agriculture"),
  cColors       = c("gray", "darkgreen", "blue", "red", "yellow"),
  edgeMode      = "gaussian",
  distThreshold = 10,
  minWght       = 0.4,
  maxWght       = 1.0,
  prioritizeEdge = TRUE
)
} # }
```
