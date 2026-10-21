# makeSlope

Calculate slope from a digital terrain model (DTM) using torch

## Usage

``` r
makeSlope(dtm, cellSize = 1, writeRaster = FALSE, outName, device = "cpu")
```

## Arguments

- dtm:

  Input SpatRaster object representing bare earth surface elevations.

- cellSize:

  Resolution of raster grid. Default is 1 m.

- writeRaster:

  TRUE or FALSE. Save output to disk. Default is TRUE.

- outName:

  Name of output raster with full file path and extension.

- device:

  "cpu" or "cuda". Use "cuda" for GPU computation. Without using the
  GPU, implementation will not be significantly faster than using
  non-tensor-based computation. Defaults is "cpu".

## Details

Calculate topographic slope using torch in degree units. Processing on
the GPU can be much faster than using base R and non-tensor-based
calculations.

## Examples

``` r
if (FALSE) { # \dontrun{
pth <- "OUTPUT PATH"
dtm <- rast(paste0(pth, "dtm.tif"))
slpR <- makeSlope(dtm, cellSize=1, writeRaster=TRUE, outName=paste0(pth, "slp.tif"), device="cuda")
} # }
```
