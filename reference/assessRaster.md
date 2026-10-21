# assessRaster

Assess semantic segmentation model using categorical raster grids
(wall-to-wall reference data and predictions)

## Usage

``` r
assessRaster(
  reference,
  predicted,
  multiclass = TRUE,
  mappings = levels(as.factor(reference)),
  decimals = 4
)
```

## Arguments

- reference:

  SpatRaster object of reference class codes/indices.

- predicted:

  SpatRaster object of reference class codes/indices.

- multiclass:

  TRUE or FALSE. If more than two classes are differentiated, use TRUE.
  If only two classes are differentiated and there are positive and
  background/negative classes, use FALSE. Default is TRUE.

- mappings:

  Vector of class names. These must be in the same order as the class
  indices or class names so that they are correctly matched to the
  correct category. If no mappings are provided, then the factor levels
  or class indices are used by default. For a binary classification, it
  is assumed that the first class is "Background" and the second class
  is "Positive".

- decimals:

  Number of decimal places to return for assessment metrics. Default is
  4.

## Value

List object containing the resulting metrics and ancillary information.

## Details

This function generates a set of summary assessment metrics when
provided reference and predicted classes. Results are returned as a list
object. For multiclass assessment, the class names (\$Classes), count of
samples per class in the reference data (\$referenceCounts), count of
samples per class in the predictions (\$predictionCounts), confusion
matrix (\$confusionMatrix), aggregated assessment metrics (\$aggMetrics)
(OA = overall accuracy, macroF1 = macro-averaged class aggregated
F1-score, macroPA = macro-averaged class aggregated producer's accuracy
or recall, and macroUA = macro-averaged class aggregated user's accuracy
or precision), class-level user's accuracies or precisions
(\$userAccuracies), class-level producer's accuracies or recalls
(\$producerAccuracies), and class-level F1-scores (\$F1Scores). For a
binary case, the \$Classes, \$referenceCounts, \$predictionCounts, and
\$confusionMatrix objects are also returned; however, the \$aggMets
object is replaced with \$Mets, which stores the following metrics:
overall accuracy, recall, precision, specificity, negative predictive
value (NPV), and F1-score. For binary cases, the second class is assumed
to be the positive case.

## Examples

``` r
if(requireNamespace("terra", quietly = TRUE)){
require(torch)
require(terra)
#Multiclass example

#Generate example data as SpatRasters
ref <- terra::rast(matrix(sample(c(1, 2, 3), 625, replace=TRUE), nrow=25, ncol=25))
pred <- terra::rast(matrix(sample(c(1, 2, 3), 625, replace=TRUE), nrow=25, ncol=25))

#Calculate metrics
metsOut <- assessRaster(reference=ref,
                        predicted=pred,
                        multiclass=TRUE,
                        mappings=c("Class A", "Class B", "Class C"),
                        decimals=4)

print(metsOut)

#Binary example

#Generate example data as SpatRasters
ref <- terra::rast(matrix(sample(c(0, 1), 625, replace=TRUE), nrow=25, ncol=25))
pred <- terra::rast(matrix(sample(c(0, 1), 625, replace=TRUE), nrow=25, ncol=25))

#Calculate metrics
metsOut <- assessRaster(reference=ref,
                        predicted=pred,
                        multiclass=FALSE,
                        mappings=c("Background", "Positive"),
                        decimals=4)

print(metsOut)
}
#> Loading required package: torch
#> Loading required package: terra
#> terra 1.9.50
#> $Classes
#> [1] "Class A" "Class B" "Class C"
#> 
#> $referenceCounts
#> Class A Class B Class C 
#>     214     209     202 
#> 
#> $predictionCounts
#> Class A Class B Class C 
#>     209     184     232 
#> 
#> $confusionMatrix
#>          Reference
#> Predicted Class A Class B Class C
#>   Class A      64      69      76
#>   Class B      70      66      48
#>   Class C      80      74      78
#> 
#> $aggMetrics
#>       OA macroF1 macroPA macroUA
#> 1 0.3328  0.3337  0.3337  0.3337
#> 
#> $userAccuracies
#> Class A Class B Class C 
#>  0.3062  0.3587  0.3362 
#> 
#> $producerAccuracies
#> Class A Class B Class C 
#>  0.2991  0.3158  0.3861 
#> 
#> $f1Scores
#> Class A Class B Class C 
#>  0.3026  0.3359  0.3594 
#> 
#> $Classes
#> [1] "Background" "Positive"  
#> 
#> $referenceCounts
#> Negative Positive 
#>      322      303 
#> 
#> $predictionCounts
#> Negative Positive 
#>      301      324 
#> 
#> $confusionMatrix
#>           Reference
#> Predicted  Negative Positive
#>   Negative      153      148
#>   Positive      169      155
#> 
#> $Mets
#>              OA Recall Precision Specificity    NPV F1Score
#> Positive 0.4928 0.5116    0.4784      0.4752 0.5083  0.4944
#> 
```
