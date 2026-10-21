# assessDL

Assess semantic segmentation model using all samples in a torch
DataLoader.

## Usage

``` r
assessDL(
  dl,
  model,
  multiclass = TRUE,
  batchSize,
  size,
  nCls,
  cCodes,
  cNames,
  cropFactorMsk = 0,
  cropFactorPred = 0,
  usedDS = FALSE,
  useCUDA = FALSE,
  decimals = 4
)
```

## Arguments

- dl:

  torch DataLoader object.

- model:

  instantiated model object as nn_module subclass as opposed to luz
  fitted object. Make sure to place model in evaluation mode.

- multiclass:

  TRUE or FALSE. If more than two classes are differentiated, use TRUE.
  If only two classes are differentiated and there are positive and
  background/negative classes, use FALSE. Default is TRUE. For binary
  cases, the second class is assumed to be the positive case.

- batchSize:

  mini-batch size used in torch DataLoader.

- size:

  size of image chips in spatial dimensions (e.g., 128, 256, 512).

- nCls:

  number of classes being differentiated.

- cCodes:

  class indices as a vector of integer values equal in length to the
  number of classes.

- cNames:

  class names as a vector of character strings with a length equal to
  the number of classes and in the correct order. Class codes and names
  are matched by position in the cCodes and cNames vectors. For binary
  case, this argument is ignored, and the first class is called
  "Negative" while the second class is called "Positive".

- cropFactorMsk:

  Number of rows and columns of cells to not include in assessment to
  minimize edge effects for reference mask. Default is 0 or no cropping.

- cropFactorPred:

  Number of rows and columns of cells to not include in assessment to
  minimize edge effects for prediction. Default is 0 or no cropping.

- usedDS:

  TRUE or FALSE. Whether or not deep supervision was used. Default is
  FALSE, or it is assumed that deep supervision was not used.

- useCUDA:

  TRUE or FALSE. Whether or not to use GPU. Default is FALSE, or GPU is
  not used. We recommend using a CUDA-enabled GPU if one is available
  since this will speed up computation.

- decimals:

  Number of decimal places to return for assessment metrics. Default is
  4.

## Value

List object containing the resulting metrics and ancillary information.

## Details

This function generates a set of summary assessment metrics based on all
samples within a torch data loader. Results are returned as a list
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
if (FALSE) { # \dontrun{
metricsOut <- assessDL(dl=testDL,
                       model=model,
                       batchSize=15,
                       size=256,
                       nCls=2,
                       mode="binary",
                       cCodes=c(1,2),
                       cNames=c("Not Mine", "Mine"),
                       usedDS=FALSE,
                       useCUDA=TRUE,
                       decimals=4)
} # }
```
