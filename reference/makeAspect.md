# makeAspect

Calculate aspect or slope orientation or a related metric from a digital
terrain model (DTM) using torch

## Usage

``` r
makeAspect(
  dtm,
  cellSize = 1,
  flatThreshold = 1,
  mode = "aspect",
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

- flatThreshold:

  Maximum slope to be re-coded to flat and assigned an aspect value of
  -1. Default is 1-degree.

- mode:

  "aspect", "northness", "eastness", "trasp", or "sei". Default is
  "aspect".

- writeRaster:

  TRUE or FALSE. Save output to disk. Default is TRUE.

- outName:

  Name of output raster with full file path and extension.

- device:

  "cpu" or "cuda". Use "cuda" for GPU computation. Without using the
  GPU, implementation will not be significantly faster than using
  non-tensor-based computation. Defaults is "cpu".

## Details

Calculate topographic aspect or slope orientation or a related metric
using torch in degree units. Processing on the GPU can be much faster
than using base R and non-tensor-based calculations. "aspect" =
topographic aspect in degree units where flat slopes are coded to -1;
"northness" = cosine of aspect in radians; "eastness" = sine of aspect
in radians; "trasp" = topographic radiation aspect index; "sei" = site
exposure index.

## Examples

``` r
if (FALSE) { # \dontrun{
pth <- "OUTPUT PATH"
dtm <- rast(paste0(pth, "dtm.tif"))
aspR <- makeAspect(dtm, cellSize=1,
flatThreshold=1,
mode= "aspect",
writeRaster=TRUE,
outName=paste0(pth, "asp.tif"),
device="cuda")
} # }
```
