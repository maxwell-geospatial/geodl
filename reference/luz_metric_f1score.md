# luz_metric_f1score

luz_metric function to calculate the macro-averaged, class aggregated
F1-score

## Usage

``` r
luz_metric_f1score(
  nCls = 1,
  cropFactorMsk = 0,
  cropFactorPred = 0,
  smooth = 1,
  mode = "multiclass",
  biThresh = 0.5,
  clsWghts = rep(1, nCls),
  zeroStart = TRUE,
  usedDS = FALSE
)
```

## Arguments

- nCls:

  Number of classes being differentiated.

- cropFactorMsk:

  Number of rows and columns of cells to not include for mask in
  assessment to minimize edge effects. Default is 0 or no cropping.

- cropFactorPred:

  Number of rows and columns of cells to not include for prediction in
  assessment to minimize edge effects. Default is 0 or no cropping.

- smooth:

  A smoothing factor to avoid divide by zero errors. Default is 1.

- mode:

  Either "binary" or "multiclass". If "binary", only the logit for the
  positive class prediction should be provided. If both the positive and
  negative or background class probability is provided for a binary
  classification, use the "multiclass" mode. Note that this package is
  designed to treat all predictions as multiclass. The "binary" mode is
  only provided for use outside of the standard geodl workflow.

- biThresh:

  Probability threshold to define postive case prediction. Default is
  0.5.

- clsWghts:

  Vector of class weightings loss calculatokn. Default is equal
  weightings.

- zeroStart:

  TRUE or FALSE. If class indices start at 0 as opposed to 1, this
  should be set to TRUE. This is required to implement one-hot encoding
  since R starts indexing at 1. Default is TRUE.

- usedDS:

  TRUE or FALSE. If deep supervision was implemented and masks are
  produced at varying scales using the defineSegDataSetDS() function,
  this should be set to TRUE. Only the original resolution is used to
  calculate assessment metrics. Default is FALSE.

## Value

Calculated metric returned as a base-R vector as opposed to tensor.

## Details

Calculates F1-score based on luz_metric() for use within training and
validation loops.

## Examples

``` r
# \donttest{
library(terra)
library(torch)
#Generate example data as SpatRasters
ref <- terra::rast(matrix(sample(c(1, 2, 3), 625, replace=TRUE), nrow=25, ncol=25))
pred1 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
pred2 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
pred3 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
pred <- c(pred2, pred2, pred3)

#Convert SpatRaster to array
ref <- terra::as.array(ref)
pred <- terra::as.array(pred)

#Convert arrays to tensors and reshape
ref <- torch::torch_tensor(ref, dtype=torch::torch_long())
#> Error: Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
pred <- torch::torch_tensor(pred, dtype=torch::torch_float32())
#> Error: Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
ref <- ref$permute(c(3,1,2))
#> Error in ref$permute: $ operator is invalid for atomic vectors
pred <- pred$permute(c(3,1,2))
#> Error in pred$permute: $ operator is invalid for atomic vectors

#Add mini-batch dimension
ref <- ref$unsqueeze(1)
#> Error in ref$unsqueeze: $ operator is invalid for atomic vectors
pred <- pred$unsqueeze(1)
#> Error in pred$unsqueeze: $ operator is invalid for atomic vectors

#Duplicate tensors to have a batch of two
ref <- torch::torch_cat(list(ref, ref), dim=1)
#> Error: Lantern is not loaded. Please use `install_torch()` to install additional dependencies.
pred <- torch::torch_cat(list(pred, pred), dim=1)
#> Error: Lantern is not loaded. Please use `install_torch()` to install additional dependencies.

#Calculate Macro-Averaged, Class Aggregated F1-Score
metric<-luz_metric_f1score(nCls=3,
                           smooth=1e-8,
                           mode = "multiclass",
                           zeroStart=FALSE,
                           usedDS=FALSE)
metric<-metric$new()
metric$update(pred,ref)
#> Error in inT$size: $ operator is invalid for atomic vectors
metric$compute()
#> [1] 1
# }
```
