# makeCrv

Calculate surface curvatures from input digital terrain model (DTM)
using torch

## Usage

``` r
makeCrv(
  dtm,
  cellSize = 1,
  mode = "mean",
  smoothRadius = 7,
  writeRaster = FALSE,
  outName,
  device = "cpu"
)
```

## Arguments

- dtm:

  Input SpatRaster object representing bare earth surface elevations.

- cellSize:

  Resolution of raster grid. Default is 1 m.

- mode:

  "mean", "profile", or "planform". Default is "mean".

- smoothRadius:

  radius of circular moving window. Default is 7 cells.

- writeRaster:

  TRUE or FALSE. Save output to disk. Default is TRUE.

- outName:

  Name of output raster with full file path and extension.

- device:

  "cpu" or "cuda". Use "cuda" for GPU computation. Without using the
  GPU, implementation will not be significantly faster than using
  non-tensor-based computation. Defaults is "cpu".

## Details

Calculate surface curvatures from input digital terrain model (DTM)
using torch. "mean" = mean curvature or average of profile and planform
curvature; "profile" = curvature in the direction of maximum slope;
"planform" = curvature in the direction perpendicular to the direction
of maximum slope. Radii are specified using cell counts as opposed to
map distance. Gaussian smoothing using a circular window of a
user-defined radius is applied prior to calculating curvatures to
minimize the impact of local noise/variability.

## Examples

``` r
if (FALSE) { # \dontrun{
pth <- "OUTPUT PATH"
dtm <- rast(paste0(pth, "dtm.tif"))
crvPro7 <- makeCrv(dtm,
cellSize=1,
mode="profile",
smoothRadius=7,
writeRaster=TRUE,
outName=paste0(pth, "crvPro7.tif"),
device="cuda")
} # }
```
