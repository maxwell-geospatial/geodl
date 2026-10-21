# makeTerrainVisTerra

Make three band terrain stack from input digital terrain model using
torch

## Usage

``` r
makeTerrainVisTorch(
  dtm,
  cellSize = 1,
  innerRadius = 2,
  outerRadius = 10,
  hsRadius = 50,
  writeRaster = FALSE,
  outName,
  device = "cpu"
)
```

## Arguments

- dtm:

  Input SpatRaster object representing bare earth surface elevations.

- cellSize:

  Resolution of the grid relative to coordinate reference system units
  (e.g., meters).

- innerRadius:

  inner radius of annulus moving window used for local TPI calculation.
  Default is 2.

- outerRadius:

  outer radius of annulus moving window used for local TPI calculation.

- hsRadius:

  outer radisu for circular moving window used for hillslope TPI
  calculation.

- writeRaster:

  TRUE or FALSE. Save output to disk. Default is TRUE.

- outName:

  Name of output raster with full file path and extension.

- device:

  Device on which to perform calculations. "cpu" or "cuda". Default is
  "cpu". Recommend "cuda". Recommend "cuda" to speed up calculations.

## Value

Three-band raster grid written to disk in TIFF format and spatRaster
object.

## Details

This function creates a three-band raster stack from an input digital
terrain model (DTM) of bare earth surface elevations using torch. This
implementation is faster than the terra-based implementation, especially
when using GPU-based computation.

The first band is a topographic position index (TPI) calculated using a
moving window with a 50 m circular radius. The second band is the square
root of slope calculated in degrees. The third band is a TPI calculated
using an annulus moving window with an inner radius of 2 and outer
radius of 5 meters. The TPI values are clamped to a range of -10 to 10
then linearly rescaled from 0 and 1. The square root of slope is clamped
to a ange of 0 to 10 then linearly rescaled from 0 to 1. Values are
provided in floating point.

The stack is described in the following publication and was originally
proposed by William Odom of the United States Geological Survey (USGS):

Maxwell, A.E., W.E. Odom, C.M. Shobe, D.H. Doctor, M.S. Bester, and T.
Ore, 2023. Exploring the influence of input feature space on CNN-based
geomorphic feature extraction from digital terrain data, Earth and Space
Science, 10: e2023EA002845. https://doi.org/10.1029/2023EA002845.

## Examples

``` r
if (FALSE) { # \dontrun{
pth <- "OUTPUT PATH"
dtm <- rast(paste0(pth, "dtm.tif"))
tVis <- makeTerrainVisTorch(dtm,
cellSize=1,
innerRadius=2,
outerRadius=5,
hsRadius=50,
writeRaster=TRUE,
outName=paste0(pth, "tVisTorch.tif"),
device="cuda")
} # }
```
