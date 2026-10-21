# makeDynamicChipsSF

Generate a vector point sf object of chip center locations for use with
defineDynamicSegDataSet().

## Usage

``` r
makeDynamicChipsSF(
  pntPth,
  pntName,
  featPth,
  featName,
  extentPth,
  extentName,
  extentCrop = 50,
  imgPth,
  imgName,
  doBackground = FALSE,
  backgroundCnt,
  backgroundDist,
  useSeed = FALSE,
  seed,
  doShuffle = FALSE
)
```

## Arguments

- pntPth:

  Full folder path or relative path to vector point layer defining chip
  centers. Must include final forward slash.

- pntName:

  File name of point layer including file extension. Must contain "code"
  (numeric class code) and "class" (text class name) columns.

- featPth:

  Full folder path or relative path to vector polygon features used as
  the mask source. Must include final forward slash.

- featName:

  File name of polygon feature layer including file extension.

- extentPth:

  Full folder path or relative path to vector extent. Must include final
  forward slash.

- extentName:

  File name of extent including file extension.

- extentCrop:

  Amount to crop extent to avoid edge effects. We recommend cropping the
  extent more than the size of the chips you will use to avoid chips
  with NA cells. Default is 50 cells.

- imgPth:

  Path to image or predictor variable raster data. Must include the
  final forward slash.

- imgName:

  File name of image or predictor variable raster data including file
  extension.

- doBackground:

  Whether or not to randomly select background cells. Default is FALSE.

- backgroundCnt:

  Number of background cells to select.

- backgroundDist:

  Minimum allowed distance between randomly selected background
  locations and boundaries of reference features.

- useSeed:

  Whether or not to set a random seed for replicability. Default is
  FALSE.

- seed:

  Random seed value.

- doShuffle:

  Whether or not to shuffle the rows in the resulting table. Default is
  FALSE.

## Value

sf dataframe object of center locations for chip creation with
associated information as attributes.

## Details

Requires an input vector point object defining chip center locations, an
input vector polygon object of reference features used to generate
segmentation masks, a boundary extent, and the path and name of the
associated raster predictor variables. Note that the raster predictor
variable must cover the full spatial extent defined by the extent
object. If not, a smaller extent should be defined for which predictor
variables are available.

The input point layer must contain a "code" column of numeric class
codes and a "class" column of text class names. The input polygon
feature layer is used only as the mask source during training and is not
required to share attributes with the point layer, but must also contain
"code" and "class" columns if class-specific masking is expected
downstream.

The extent can be cropped to avoid generating chips without a full set
of cells available. We recommend cropping by a factor larger than 50%
the width/height of the desired chip size.

The tool returns a sf dataframe with class name ("class"), class numeric
code ("code"), path to the image or predictor variables ("imgPth"), name
of the image or predictor variables ("imgName"), the path to the input
point layer ("pntPth"), the file name of the input point layer
("pntName"), the path to the input polygon features ("featPth"), the
file name of the input polygon features ("featName"), the file path to
the extent ("extentPth"), the file name for the extent ("extentName"),
and the point geometry ("geometry").
