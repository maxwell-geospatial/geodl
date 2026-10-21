# makeTRI

Calculate a topographic roughness index (TRI) from a digital terrain
model (DTM) using torch

## Usage

``` r
makeTRI(
  dtm,
  cellSize = 1,
  roughRadius = 7,
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

- roughRadius:

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

Calculate a topographic roughness index (TPI) from a digital terrain
model (DTM) using torch. Us calculated as the square root of the
standard deviation of slope in a local moving window. Output can be
noisy. Radii are specified using cell counts as opposed to map distance.

## Examples

``` r
if (FALSE) { # \dontrun{
pth <- "OUTPUT PATH"
dtm <- rast(paste0(pth, "dtm.tif"))
tri11 <- makeTRI(dtm,
cellSize=1,
roughRadius=11,
writeRaster=TRUE,
outName=paste0(pth, "tri11f.tif"),
device="cuda")
} # }
```
