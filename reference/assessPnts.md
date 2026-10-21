# assessPnts

Assess semantic segmentation model using point locations

## Usage

``` r
assessPnts(
  reference,
  predicted,
  multiclass = TRUE,
  mappings = levels(as.factor(reference)),
  aliases = NULL,
  decimals = 4
)
```

## Arguments

- reference:

  Data frame column or vector of reference classes.

- predicted:

  Data frame column or vector of predicted classes.

- multiclass:

  TRUE or FALSE. If more than two classes are differentiated, use TRUE.
  If only two classes are differentiated and there are positive and
  background/negative classes, use FALSE. Default is TRUE.

- mappings:

  Vector of class names. These must be in the same order as the factor
  levels so that they are correctly matched to the correct category. If
  no mappings are provided, then the factor levels are used by default.
  For a binary classification, it is assumed that the first class is
  "Background" and the second class is "Positive".

- aliases:

  Optional vector of display names used to label the output (confusion
  matrix row/column names, class name lists, and metric name vectors).
  Must be the same length as `mappings` and provided in the same order.
  If NULL (default), the `mappings` values are used as labels.

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
#Multiclass example

#Generate example data as data frame of class predictions
inDF <- data.frame(ref = sample(c("Class A", "Class B", "Class C"), 1000, replace=TRUE),
pred = sample(c("Class A", "Class B", "Class C"), 1000, replace=TRUE))

#Calculate metrics
metsOut <- assessPnts(reference=inDF$ref,
                     predicted=inDF$pred,
                     multiclass=TRUE,
                     mappings = c("Class A", "Class B", "Class C"),
                     decimals=4)

print(metsOut)
#> $Classes
#> [1] "Class A" "Class B" "Class C"
#> 
#> $referenceCounts
#> Class A Class B Class C 
#>     332     347     321 
#> 
#> $predictionCounts
#> Class A Class B Class C 
#>     342     321     337 
#> 
#> $confusionMatrix
#>          Reference
#> Predicted Class A Class B Class C
#>   Class A     112     112     118
#>   Class B     122     102      97
#>   Class C      98     133     106
#> 
#> $aggMetrics
#>     OA macroF1 macroPA macroUA
#> 1 0.32  0.3202  0.3205  0.3199
#> 
#> $userAccuracies
#> Class A Class B Class C 
#>  0.3275  0.3178  0.3145 
#> 
#> $producerAccuracies
#> Class A Class B Class C 
#>  0.3373  0.2939  0.3302 
#> 
#> $f1Scores
#> Class A Class B Class C 
#>  0.3323  0.3054  0.3222 
#> 

#Multiclass example with aliases to rename output labels
inDF2 <- data.frame(ref  = sample(c("cls_a", "cls_b", "cls_c"), 1000, replace=TRUE),
                    pred = sample(c("cls_a", "cls_b", "cls_c"), 1000, replace=TRUE))

metsOut2 <- assessPnts(reference=inDF2$ref,
                       predicted=inDF2$pred,
                       multiclass=TRUE,
                       mappings = c("cls_a", "cls_b", "cls_c"),
                       aliases  = c("Class A", "Class B", "Class C"),
                       decimals=4)

print(metsOut2)
#> $Classes
#> [1] "Class A" "Class B" "Class C"
#> 
#> $referenceCounts
#> Class A Class B Class C 
#>     326     350     324 
#> 
#> $predictionCounts
#> Class A Class B Class C 
#>     343     351     306 
#> 
#> $confusionMatrix
#>          Reference
#> Predicted Class A Class B Class C
#>   Class A     108     123     112
#>   Class B     119     124     108
#>   Class C      99     103     104
#> 
#> $aggMetrics
#>      OA macroF1 macroPA macroUA
#> 1 0.336  0.3358  0.3355   0.336
#> 
#> $userAccuracies
#> Class A Class B Class C 
#>  0.3149  0.3533  0.3399 
#> 
#> $producerAccuracies
#> Class A Class B Class C 
#>  0.3313  0.3543  0.3210 
#> 
#> $f1Scores
#> Class A Class B Class C 
#>  0.3229  0.3538  0.3302 
#> 

#Binary example

#Generate example data as data frame of class predictions
inDF <- data.frame(ref = sample(c("Background", "Positive"), 1000, replace=TRUE),
                  pred = sample(c("Background", "Positive"), 1000, replace=TRUE))

#Calculate metrics
metsOut <- assessPnts(reference=inDF$ref,
                     predicted=inDF$pred,
                     multiclass=FALSE,
                     mappings = c("Background", "Positive"),
                     decimals=4)

print(metsOut)
#> $Classes
#> [1] "Background" "Positive"  
#> 
#> $referenceCounts
#> Background   Positive 
#>        519        481 
#> 
#> $predictionCounts
#> Background   Positive 
#>        504        496 
#> 
#> $confusionMatrix
#>             Reference
#> Predicted    Background Positive
#>   Background        261      243
#>   Positive          258      238
#> 
#> $Mets
#>             OA Recall Precision Specificity    NPV F1Score
#> Positive 0.499 0.4948    0.4798      0.5029 0.5179  0.4872
#> 
```
