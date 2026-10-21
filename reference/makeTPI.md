# makeTPI

Calculate a topographic position index (TPI) from a digital terrain
model (DTM) using torch

## Usage

``` r
makeTPI(
  dtm,
  cellSize = 1,
  innerRadius = 2,
  outerRadius = 10,
  mode = "circle",
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

- innerRadius:

  = Inner radius when using annulus moving window. Default is 2 cells.

- outerRadius:

  = Outer radius when using a circle or annulus moving window. Default
  is 10 cells.

- mode:

  Either "circle" or "annulus". Default is "circle".

- writeRaster:

  TRUE or FALSE. Save output to disk. Default is TRUE.

- outName:

  Name of output raster with full file path and extension.

- device:

  "cpu" or "cuda". Use "cuda" for GPU computation. Without using the
  GPU, implementation will not be significantly faster than using
  non-tensor-based computation. Defaults is "cpu".

## Details

Calculate a topographic position index (TPI) from a digital terrain
model (DTM) using torch. TPI is calculated as the center cell elevation
minus the mean elevation in a local moving window. Large, positive
values indicate local topographic high points while large, negative
values indicate topographic low points. Near zero values indicate flat
areas. Larger windows can characterize the hillslope position of a cell
while smaller windows can capture local topographic variability. Radii
are specified using cell counts as opposed to map distance.

## Examples

``` r
if (FALSE) { # \dontrun{
pth <- "OUTPUT PATH"
dtm <- rast(paste0(pth, "dtm.tif"))
tpiA3_11 <- makeTPI(dtm,cellSize=1,
innerRadius=3,
outerRadius=11,
mode="circle",
writeRaster=TRUE,
outName=paste0(pth, "tpiA3_11.tif"),
device="cuda")
} # }
```
