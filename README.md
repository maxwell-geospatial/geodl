# geodl <img src="geodlHex.png" align="right" height="132"/>

## Utility functions for geospatial deep learning

This small package provides a set of utility functions for geospatial
deep learning. Functions can generate raster masks from vector data,
image chips and masks from larger spatial extents, chip summary metrics,
lists of chips and masks, and accuracy assessment metrics. This package
relies heavily on the terra package.

We would like to expand this package to include more deep learning
tools. For example, we would like to generate functions to train and use
deep learning models using the torch package. This package is a work in
progress. If you are interested in collaborating, please reach out.

## To Install

``` r
library(devtools)
install_github("maxwell-geospatial/geodl")
```

## Functions

1.  **makeMasks():** generate raster masks from vector data
2.  **makeChips():** create image chips and associated masks from
    geospatial raster data (binary classification)
3.  **makeChipsMultiClass():** create image chips and associated masks
    from geospatial raster data (multiclass classification)
4.  **describeChips():** obtain summary statistics from image chip bands
    and count pixels in each class.
5.  **makeTerrainDerivatives():** Create a three-band stack of terrain
    derivatives from a digital elevation model
6.  **assessPnts():** assess classification using point samples
7.  **assessRaster():** assess classification using wall-to-wall raster
    reference data and predictions

## Examples

Example data can be downloaded at [this
link](https://figshare.com/articles/dataset/geodl_example_data/23835165).
You may need to update file paths in the code below.

``` r
library(geodl)
#Create terrain derivatives from DTM
inDTM <- terra::rast("data/elev/dtm.tif")
terrOut <- makeTerrainDerivatives(dtm=inDTM, res=2, filename="data/elev/stack.tif")
terra::plotRGB(terrOut*255)

#Make mask
makeMasks(image = "data/toChip/image/KY_Saxton_709705_1970_24000_geo.tif",
          features = "data/toChip/msks/KY_Saxton_709705_1970_24000_geo.shp",
          crop = TRUE,
          extent = "data/toChip/extent/KY_Saxton_709705_1970_24000_geo.shp",
          field = "classvalue",
          background = 0,
          outImage = "data/toChip/output/topoOut.tif",
          outMask = "data/toChip/output/mskOut.tif",
          mode = "Both")

terra::plotRGB(terra::rast("data/toChip/output/topoOut.tif"))
terra::plot(terra::rast("data/toChip/output/mskOut.tif"))

#Make chips
makeChips(image = "data/toChip/output/topoOut.tif",
          mask = "data/toChip/output/mskOut.tif",
          n_channels = 3,
          size = 256,
          stride_x = 256,
          stride_y = 256,
          outDir = "data/toChip/chips/",
          mode = "Positive")

makeChipsMultiClass(image = "data/toChip/output/topoOut.tif",
                    mask = "data/toChip/output/mskOut.tif",
                    n_channels = 3,
                    hasZero = TRUE,
                    size = 256,
                    stride_x = 256,
                    stride_y = 256,
                    outDir = "data/toChip/chips/")

#Describe chips
lstOut <- describeChips(folder = "data/toChip/chips/",
              extension = ".tif",
              mode="Positive",
              subSample = TRUE,
              numChips=200,
              subSamplePix=TRUE,
              sampsPerChip=100)

print(lstOut$ImageStats)
print(lstOut$mskStats)

#Describe chips
chpDF <- makeChipsDF(folder = "data/toChip/chips/",
                    outCSV = "data/toChip/chips/chipsDF.csv",
                    extension = ".tif",
                    mode="Positive",
                    shuffle=FALSE,
                    saveCSV=TRUE)

#Assess using point locations

#Example 1: table that already has the reference and predicted labels for a multiclass classification
mcIn <- readr::read_csv("data/tables/multiClassExample.csv")
myMetrics <- assessPnts(reference=mcIn$ref,
                       predicted=mcIn$pred,
                       multiclass=TRUE)

#Example 2: table that already has the reference and predicted labels for a binary classification
bIn <- readr::read_csv("data/tables/binaryExample.csv")
myMetrics <- assessPnts(reference=bIn$ref,
                       predicted=bIn$pred,
                       multiclass=FALSE,
                       positive_case = "Mine")

#Example 3: Read in point layer and intersect with rater output
pntsIn <- terra::vect("data/topoResult/topoPnts.shp")
refG <- terra::rast("data/topoResult/topoRef.tif")
predG <- terra::rast("data/topoResult/topoPred.tif")
pntsIn2 <- terra::project(pntsIn, terra::crs(refG))
refIsect <- terra::extract(refG, pntsIn2)
predIsect <- terra::extract(predG, pntsIn2)
resultsIn <- data.frame(ref=as.factor(refIsect$topoRef),
                        pred=as.factor(predIsect$topoPred))
myMetrics <- assessPnts(reference=bIn$ref,
                       predicted=bIn$pred,
                       multiclass=FALSE,
                       mappings=c("Mine", "Not Mine"),
                       positive_case = "Mine")

#Assess using raster grids
refG <- terra::rast("data/topoResult/topoRef.tif")
predG <- terra::rast("data/topoResult/topoPred.tif")
refG2 <- crop(project(refG, predG), predG)
myMetrics <- assessRaster(reference = refG2,
                          predicted = predG,
                          multiclass = FALSE,
                          mappings = c("Not Mine", "Mine"),
                          positive_case = "Mine")
```

# `assessPnts`

Assess semantic segmentation model using point locations

## Description

This function will generate a set of summary metrics when provided
reference and predicted classes. For a multiclass classification problem
a confusion matrix is produced with the columns representing the
reference data and the rows representing the predictions. The following
metrics are calculated: overall accuracy (OA), 95% confidence interval
for OA (OAU and OAL), the Kappa statistic, map image classification
efficacy (MICE), average class user’s accuracy (aUA), average class
producer’s accuracy (aPA), average class F1-score, overall error
(Error), allocation disagreement (Allocation), quantity disagreement
(Quantity), exchange disagreement (Exchange), and shift disagreement
(shift). For average class user’s accuracy, producer’s accuracy, and
F1-score, macro-averaging is used where all classes are equally
weighted. For a multiclass classification all class user’s and
producer’s accuracies are also returned.

## Usage

``` r
assessPnts(
  reference,
  predicted,
  multiclass = TRUE,
  mappings = levels(as.factor(reference)),
  positive_case = mappings[1]
)
```

## Arguments

| Argument        | Description                                                                                                                                                                                                             |
|--------------------------------|----------------------------------------|
| `reference`     | Data frame column or vector of reference classes.                                                                                                                                                                       |
| `predicted`     | Data frame column or vector of predicted classes.                                                                                                                                                                       |
| `multiclass`    | TRUE or FALSE. If more than two classes are differentiated, use TRUE. If only two classes are differentiated and there are positive and background/negative classes, use FALSE. Default is TRUE.                        |
| `mappings`      | Vector of factor level names. These must be in the same order as the factor levels so that they are correctly matched to the correct category. If no mappings are provided, then the factor levels are used by default. |
| `positive_case` | Factor level associated with the positive case for a binary classification. Default is the second factor level. This argument is not used for multiclass classification.                                                |

## Details

For a binary classification problem, a confusion matrix is returned
along with the following metrics: overall accuracy (OA), overall
accuracy 95% confidence interval (OAU and OAL), the Kappa statistic
(Kappa), map image classification efficacy (MICE), precision
(Precision), recall (Recall), F1-score (F1), negative predictive value
(NPV), specificity (Specificity), overall error (Error), allocation
disagreement (Allocation), quantity disagreement (Quantity), exchange
disagreement (Exchange), and shift disagreement (shift).

Results are returned as a list object. This function makes use of the
caret, diffeR, and rfUtilities packages.

## Value

List object containing the resulting metrics. For multiclass assessment,
the confusion matrix is provided in the $ConfusionMatrix object, the
aggregated metrics are provided in the $Metrics object, class user’s
accuracies are provided in the $UsersAccs object, class producer’s
accuracies are provided in the $ProducersAccs object, and the list of
classes are provided in the $Classes object. For a binary
classification, the confusion matrix is provided in the $ConfusionMatrix
object, the metrics are provided in the $Metrics object, the classes are
provided in the $Classes object, and the positive class label is
provided in the $PositiveCase object.

# `assessRaster`

Assess semantic segmentation model using categorical raster grids
(wall-to-wall reference data and predictions)

## Description

This function will generate a set of summary metrics when provided
reference and predicted classes. For a multiclass classification problem
a confusion matrix is produced with the columns representing the
reference data and the rows representing the predictions. The following
metrics are calculated: overall accuracy (OA), 95% confidence interval
for OA (OAU and OAL), the Kappa statistic, map image classification
efficacy (MICE), average class user’s accuracy (aUA), average class
producer’s accuracy (aPA), average class F1-score, overall error
(Error), allocation disagreement (Allocation), quantity disagreement
(Quantity), exchange disagreement (Exchange), and shift disagreement
(shift). For average class user’s accuracy, producer’s accuracy, and
F1-score, macro-averaging is used where all classes are equally
weighted. For a multiclass classification all class user’s and
producer’s accuracies are also returned.

## Usage

``` r
assessRaster(
  reference,
  predicted,
  multiclass = TRUE,
  mappings,
  positive_case = mappings[1]
)
```

## Arguments

| Argument        | Description                                                                                                                                                                                                                                                                                                                                                               |
|--------------------------------|----------------------------------------|
| `reference`     | Single-band, categorical spatRaster object representing the reference labels. Note that the reference and predicted data must have the same extent, number of rows and columns, and coordinate reference system.                                                                                                                                                          |
| `predicted`     | Single-band, categorical spatRaster object representing the predicted labels. Note that the reference and predicted data must have the ’ same extent, number of rows and columns, and coordinate reference system.                                                                                                                                                        |
| `mappings`      | Vector of factor level names. These must be in the same order as the factor levels so that they are correctly matched to the correct category. If no mappings are provided, then the factor levels are used by default. This parameter can be especially useful when using raster data as input as it allows the grid codes to be associated with more meaningful labels. |
| `positive_case` | Factor level associated with the positive case for a binary classification. Default is the second factor level. This argument is not used for multiclass classification.                                                                                                                                                                                                  |

## Details

For a binary classification problem, a confusion matrix is returned
along with the following metrics: overall accuracy (OA), overall
accuracy 95% confidence interval (OAU and OAL), the Kappa statistic
(Kappa), map image classification efficacy (MICE), precision
(Precision), recall (Recall), F1-score (F1), negative predictive value
(NPV), specificity (Specificity), overall error (Error), allocation
disagreement (Allocation), quantity disagreement (Quantity), exchange
disagreement (Exchange), and shift disagreement (shift).

Results are returned as a list object. This function makes use of the
caret, diffeR, and rfUtilities packages.

## Value

List object containing the resulting metrics. For multiclass assessment,
the confusion matrix is provided in the $ConfusionMatrix object, the
aggregated metrics are provided in the $Metrics object, class user’s
accuracies are provided in the $UsersAccs object, class producer’s
accuracies are provided in the $ProducersAccs object, and the list of
classes are provided in the $Classes object. For a binary
classification, the confusion matrix is provided in the $ConfusionMatrix
object, the overall metrics are provided in the $Metrics object, the
classes are provided in the $Classes object, and the positive class
label is provided in the $PositiveCase object.

# `describeChips`

Generate data frame of band summary statistics and class pixel counts

## Description

This function will generate a set of summary metrics from image chips
and associated masks stored in a directory. For each band, the minimum,
1st quartile, median, mean, 3rd quartile, and maximum values are
returned. For mask data, the count of pixels in each class are returned.
These summarizations can be useful for data normalization and
determining class weightings in loss calculations.

## Usage

``` r
describeChips(
  folder,
  extension = ".tif",
  mode = "All",
  subSample = TRUE,
  numChips = 200,
  numChipsBack = 200,
  subSamplePix = TRUE,
  sampsPerChip = 100
)
```

## Arguments

| Argument       | Description                                                                                                                                                                                                                                                                                          |
|--------------------------------|----------------------------------------|
| `folder`       | Full folder path or folder path relative to the current working directory that holds the image chips and associated masks. You must include the final forward slash in the folder path (e.g., “C:/data/chips/”).                                                                                     |
| `extension`    | raster file extension (e.g., “.tif”, “.png”, “.jpeg”, or “.img”). The tools in this package generate files in “.tif” format, so this is the default. This option is provided if chips are generated using another method.                                                                            |
| `mode`         | Either “All”, “Positive”, or “Divided”. This should match the settings used in the makeChips() function or be set to “All” if makeChipsMultiClass() is used. Default is “All”.                                                                                                                       |
| `subSample`    | TRUE or FALSE. Whether or not to subsample the image chips to calculate the summary metrics. We recommend using a subset if a large set of chips are being summarized to reduce computational load. The default is TRUE.                                                                             |
| `numChips`     | If subSample is set to TRUE, this parameter defines the number of chips to subset. The default is 200. This parameter will be ignored if subSample is set to FALSE.                                                                                                                                  |
| `numChipsBack` | If subSample is set to TRUE and the mode is “Divided”, this parameter indicates the number of chips to sample from the background-only samples. The default is 200. This parameter will be ignored if subSample is set to FALSE and mode is not “Divided”.                                           |
| `subSamplePix` | TRUE or FALSE. Whether or not to calculate statistics using a subsample of pixels from each image chip as opposed to all pixels. If a large number of chips are available and/or each chip is large, we suggest setting this argument to TRUE to reduce the computational load. The default is TRUE. |
| `sampsPerChip` | If subSamplePix is TRUE, this parameters specifies the number of random pixels to sample per chip. The default is 100. If subSamplePix is set to FALSE, this parameter is ignored.                                                                                                                   |

## Value

List object containing the summary metrics for each band in the
$ImageStats object and the count of pixels by class in the $maskStats
object.

# `hello`

Hello, World!

## Description

Prints ‘Hello, world!’.

## Usage

``` r
hello()
```

## Examples

``` r
hello()
```

# `makeChips`

Generate image chips from images and associated raster masks

## Description

This function will generate image and mask chips from an input image and
associated raster mask. The chips are written into the defined
directory. The number of rows and columns of pixels in each chip are
equal to the size argument. If a stride_x and/or stride_y is used that
is different from the size argument, resulting chips will either overlap
or have gaps between them. In order to not have overlap or gaps, the
stride_x and stride_y arguments should be the same as the size argument.
Both the image chips and associated masks are written to TIFF format
(“.tif”). Input data are not limited to three band images. This function
is specifically for a binary classification where the positive case is
indicated with a cell value of 1 and the background or negative case is
indicated with a cell value of 0. If an irregular shaped raster grid is
provided, only chips and masks that contain no NA or NoDATA cells will
be produced.

## Usage

``` r
makeChips(
  image,
  mask,
  n_channels = 3,
  size = 256,
  stride_x = 256,
  stride_y = 256,
  outDir,
  mode = "All"
)
```

## Arguments

| Argument     | Description                                                                                                                                                                                                                                      |
|--------------------------------|----------------------------------------|
| `image`      | Path to input image. Function will generate a SpatRaster object internally. The image and mask must have the same extent, number of rows and columns of pixels, cell size, and coordinate reference system.                                      |
| `mask`       | Path to single-band mask. Function will generate a SpatRaster object internally. The image and mask must have the same extent, number of rows and columns if pixels, cell size, and coordinate reference system.                                 |
| `n_channels` | Number of channels in the input image. Default is 3.                                                                                                                                                                                             |
| `size`       | Size of image chips as number of rows and columns of pixels. Default is 256.                                                                                                                                                                     |
| `stride_x`   | Stride in the x (columns) direction. Default is 256.                                                                                                                                                                                             |
| `stride_y`   | Stride in the y (rows) direction. Default is 256.                                                                                                                                                                                                |
| `outDir`     | Full or relative path to the current working directory where you want to write the chips to. Subfolders in this directory will be generated by the function. You must include the final forward slash in the file path (e.g., “C:/data/chips/”). |
| `mode`       | Either “All”, “Positive”, or “Divided”. Please see the explanations provided above. The default is “All”.                                                                                                                                        |

## Details

Three modes are available. If “All” is used, all image chips will be
generated even if they do not contain pixels mapped to the positive
case. Within the provided directory, image chips will be written to an
“images” folder and masks will be written to a “masks” folder. If
“Positive” is used, only chips that have at least 1 pixel mapped to the
positive class will be produced. Background- only chips will not be
generated. Within the provided directory, image chips will be written to
an “images” folder and masks will be written to a “masks” folder.
Lastly, if the “Divided” method is used, separate “positive” and
“background” folders will be created with “images” and “masks”
subfolders. Any chip that has at least 1 pixel mapped to the positive
class will be written to the “positive” folder while any chip having
only background pixels will be written to the “background” folder.

## Value

Image and mask files written to disk in TIFF format. Spatial reference
information is not maintained. No R object is returned.

# `makeChipsDF`

Create data frame and CSV file listing image chips and associated masks

## Description

This function creates a dataframe and, optionally, a CSV file that lists
all of the image chips and associated masks in a directory. Three
columns are produced. The chp column provides the name of the chip, the
chpPth column provides the path to the chip, and the chpMsk provides the
path to the associated mask. All paths are relative to the input folder
as opposed to the full file path so that the results can still be used
if the data are copied to a new location on disk or to a new computer.

## Usage

``` r
makeChipsDF(
  folder,
  outCSV,
  extension = ".tif",
  mode = "All",
  shuffle = FALSE,
  saveCSV = FALSE
)
```

## Arguments

| Argument    | Description                                                                                                                                                                                                                                                 |
|--------------------------------|----------------------------------------|
| `folder`    | Full path or path relative to the working directory to the folder containing the image chips and associated masks. You must include the final forward slash in the path (e.g., “C:/data/chips/”).                                                           |
| `outCSV`    | File name and full path or path relative to the working directory for the resulting CSV file with a “.csv” extension.                                                                                                                                       |
| `extension` | The extension of the image and mask raster data (e.g., “.tif”, “.png”, “.jpeg”, or “.img”). The default is “.tif” since this is the file format used by the functions in this package. This option is provided if chips are generated using another method. |
| `mode`      | Either “All”, “Positive”, or “Divided”. This should match the setting used in the makeChips() function. If the makeChipsMultiClass() function was used, this should be set to “All” or left as the default. The default is “All”.                           |
| `shuffle`   | TRUE or FALSE. Whether or not to shuffle the rows in the table. Rows can be shuffled to potentially reduced autocorrelation in the data. The default is FALSE.                                                                                              |
| `saveCSV`   | TRUE or FALSE. Whether or not to save the CSV file or just return the dataframe. If this is set to FALSE then the outCSV parameter is ignored and no CSV file is generated. The default is FALSE.                                                           |

## Value

Data frame with three columns (chp, chpPth, and mskPth) and, optionally,
a CSV file written to disk.

# `makeChipsMultiClass`

Generate image chips from images and associated raster masks for
multiclass classification

## Description

This function will generate image and mask chips from an input image and
associated raster mask. The chips will be written into the defined
directory. The number of rows and columns of pixels per chip are equal
to the size argument. If a stride_x and/or stride_y is used that is
different from the size argument, resulting chips will either overlap or
have gaps between them. In order to not have overlap or gaps, the
stride_x and stride_y arguments should be the same as the size argument.
Both the image chips and associated masks are written to TIFF format
(“.tif”). Input data are not limited to three band images. This function
is specifically for a multiclass classification. For a binary
classification, use the makeChips() function. If an irregular shaped
raster grid is provided, only chips and masks that contain no NA or
NoDATA cells will be produced.

## Usage

``` r
makeChipsMultiClass(
  image,
  mask,
  n_channels = 3,
  hasZero = TRUE,
  size = 256,
  stride_x = 256,
  stride_y = 256,
  outDir
)
```

## Arguments

| Argument     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|--------------------------------|----------------------------------------|
| `image`      | Path to input image. Function will generate a SpatRaster object internally. The image and mask must have the same extent, number of rows and columns of pixels, cell size, and coordinate reference system.                                                                                                                                                                                                                                                                                                                                                 |
| `mask`       | Path to single-band mask. Function will generate a SpatRaster object internally. The image and mask must have the same extent, number of rows and columns of pixels, cell size, and coordinate reference system.                                                                                                                                                                                                                                                                                                                                            |
| `n_channels` | Number of channels in the input image. Default is 3.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `hasZero`    | If the class codes begin at 0 as opposed to 1, this shouls be set to TRUE. If the class codes start at 1 as opposed to 0, this should be set to FALSE. In the case where the class codes start at 1, each code will be reduced by 1 so that codes start at 0. For example, codes 1,2,3,4 would be converted to 0,1,2,3. This is because most deep learning frameworks expect the class codes to be represented as indices from 0 to n-1 where n is the number of classes. Users should be aware of this manipulation and its impact of class code mappings. |
| `size`       | Size of image chips as number of rows and columns of pixels. Default is 256.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `stride_x`   | Stride in the x (columns) direction. Default is 256.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `stride_y`   | Stride in the y (rows) direction. Default is 256.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `outDir`     | Full or relative path to the current working directory where you want to write the chips to. Subfolders in this directory will be generated by the function. You must include the final forward slash in the file path (e.g., “C:/data/chips/”).                                                                                                                                                                                                                                                                                                            |

## Details

Within the provided directory, image chips will be written to an
“images” folder and masks will be written to a “masks” folder.

## Value

Image and mask files written to disk in TIFF format. No R object is
returned.

# `makeMasks`

Make raster mask from input vector data

## Description

This function creates a raster mask from input vector data. The cell
value is indicated by the field parameter. A unique numeric code should
be provided for each class. In the case of a binary classification, 0
should indicate background and 1 should indicate positive. For a
multiclass problem, values should be sequential from 0 to n-1, where n
is the number of classes, or 1 to n. We recommend using 0 to n-1. If no
cropping is applied, the generated raster mask should have the same
spatial extent, number of rows of pixels, number of columns of pixels,
cell size, and coordinate reference system as the input image.

## Usage

``` r
makeMasks(
  image,
  features,
  crop = FALSE,
  extent,
  field,
  background = 0,
  outImage,
  outMask,
  mode = "Both"
)
```

## Arguments

| Argument     | Description                                                                                                                                                                                                                                                                                                                                                                                                  |
|--------------------------------|----------------------------------------|
| `image`      | File name and full path or path relative to working directory for image. Image are converted to a spatRaster internally.                                                                                                                                                                                                                                                                                     |
| `features`   | File name and full path or path relative to working directory for vector mask or label data. A field should be provided that differentiates classes using unique numeric codes as explained above. If the input features use a different coordinate reference system then the input image, the features will be reprojected to match the image. Vector data are converted to a SpatVector object internally. |
| `crop`       | TRUE or FALSE. Whether or not to crop the input image data relative to a defined vector extent. The default is FALSE.                                                                                                                                                                                                                                                                                        |
| `extent`     | File name and full path or path relative to working directory for vector extent data. If the extent uses a different coordinate reference system then the input image, the features will be reprojected to match the image. Vector data are converted to a SpatVector object internally.                                                                                                                     |
| `field`      | The name of the field in the feature vector data that differentiate classes using a unique numeric code with an integer data type. Field name should be provided as a string.                                                                                                                                                                                                                                |
| `background` | The numeric value to assign to the background class. The default is 0. If the full spatial extent has labels in the input feature data, no background value will be applied. For binary classification problems, the background should be coded to 0 and the positive case should be coded to 1. It is not necessary to include the background class in the vector feature data.                             |
| `outImage`   | Image output name in TIFF format (“.tif”) with full path or path relative to working directory for image. This output will only be generated if the mode is set to “Both”.                                                                                                                                                                                                                                   |
| `outMask`    | Mask output name in TIFF format (“.tif”) with full path or path relative to working directory for image. Output will be a single-band raster grid of class numeric codes.                                                                                                                                                                                                                                    |
| `mode`       | Either “Both” or “Mask”. If “Both”, a copy of the image will be made along with the generated raster mask. If “Mask”, only the mask is produced. If you are experiencing issues with alignment between the image and associated mask, setting the mode to “Both” can alleviate this issue. However, this will result in more data being written to disk.                                                     |

## Value

Single-band raster mask written to disk in TIFF format and, optionally,
a copy of the image written to disk. Cropping may be applied as
specified. No R objects are returned.

# `makeTerrainDerivatives`

Make three band terrain stack from input digital terrain model

## Description

This function creates a three-band raster stack from an input digital
terrain model (DTM) of bare earth surface elevations. The first band is
a topographic position index (TPI) calculated using a moving window with
a 50 m circular radius. The second band is the square root of slope
calculated in degrees. The third band is a TPI calculated using an
annulus moving window with an inner radius of 2 and outer radius of 5
meters. The TPI values are clamped to a range of -10 to 10 then linearly
rescaled from 0 and 1. The square root of slope is clamped to a range of
0 to 10 then linearly rescaled from 0 to 1. Values are provided in
floating point.

## Usage

``` r
makeTerrainDerivatives(dtm, res, filename)
```

## Arguments

| Argument   | Description                                                                                                                                                       |
|--------------------------------|----------------------------------------|
| `dtm`      | Input SpatRaster object representing bare earth surface elevations.                                                                                               |
| `res`      | Resolution of the grid relative to coordinate reference system units (e.g., meters).                                                                              |
| `filename` | Name and full path or path relative to working directory for output terrain stack. We recommend saving the data to either TIFF (“.tif”) or Image (“.img”) format. |

## Details

The stack is described in the following publication:

Maxwell, A.E., W.E. Odom, C.M. Shobe, D.H. Doctor, M.S. Bester, and T.
Ore, 2023. Exploring the influence of input feature space on CNN-based
geomorphic feature extraction from digital terrain data, Earth and Space
Science, 10: e2023EA002845. <https://doi.org/10.1029/2023EA002845>.

## Value

Three-band raster mask written to disk in TIFF format and spatRaster
object.
