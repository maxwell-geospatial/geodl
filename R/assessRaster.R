#' assessRaster
#'
#' Assess semantic segmentation model using categorical raster grids (wall-to-wall
#' reference data and predictions)
#'
#' This function generates a set of summary assessment metrics when provided
#' reference and predicted classes. Results are returned as a list object. For
#' multiclass assessment, the class names ($Classes), count of samples per class
#' in the reference data ($referenceCounts), count of samples per class in the
#' predictions ($predictionCounts), confusion matrix ($confusionMatrix),
#' aggregated assessment metrics ($aggMetrics) (OA = overall accuracy, macroF1 = macro-averaged
#' class aggregated F1-score, macroPA = macro-averaged class aggregated producer's
#' accuracy or recall, and macroUA = macro-averaged class aggregated user's accuracy or
#' precision), class-level user's accuracies or precisions ($userAccuracies),
#' class-level producer's accuracies or recalls ($producerAccuracies), and class-level
#' F1-scores ($F1Scores). For a binary case, the $Classes, $referenceCounts,
#' $predictionCounts, and $confusionMatrix objects are also returned; however, the $aggMets
#' object is replaced with $Mets, which stores the following metrics: overall accuracy, recall,
#' precision, specificity, negative predictive value (NPV), and F1-score.
#' For binary cases, the second class is assumed to be the positive case.
#'
#' @param reference SpatRaster object of reference class codes/indices.
#' @param predicted SpatRaster object of reference class codes/indices.
#' @param multiclass TRUE or FALSE. If more than two classes are differentiated,
#' use TRUE. If only two classes are differentiated and there are positive and
#' background/negative classes, use FALSE. Default is TRUE.
#' @param mappings Vector of class names. These must be in the same order
#' as the class indices or class names so that they are correctly matched to the correct category.
#' If no mappings are provided, then the factor levels or class indices are used by default.
#' For a binary classification, it is assumed that the first class is "Background" and
#' the second class is "Positive".
#' @param decimals Number of decimal places to return for assessment metrics. Default is 4.
#' @return List object containing the resulting metrics and ancillary information.
#' @examples
#' #Multiclass example
#'
#' #Generate example data as SpatRasters
#'  ref <- terra::rast(matrix(sample(c(1, 2, 3), 625, replace=TRUE), nrow=25, ncol=25))
#'  pred <- terra::rast(matrix(sample(c(1, 2, 3), 625, replace=TRUE), nrow=25, ncol=25))
#'
#'  #Calculate metrics
#'  metsOut <- assessRaster(reference=ref,
#'                         predicted=pred,
#'                         multiclass=TRUE,
#'                         mappings=c("Class A", "Class B", "Class C"),
#'                         decimals=4)
#'
#' print(metsOut)
#'
#'  #Binary example
#'
#'  Generate example data as SpatRasters
#'  ref <- terra::rast(matrix(sample(c(0, 1), 625, replace=TRUE), nrow=25, ncol=25))
#'  pred <- terra::rast(matrix(sample(c(0, 1), 625, replace=TRUE), nrow=25, ncol=25))
#'
#'  #Calculate metrics
#'  metsOut <- assessRaster(reference=ref,
#'                         predicted=pred,
#'                         multiclass=FALSE,
#'                         mappings=c("Background", "Positive"),
#'                         decimals=4)
#'
#' print(metsOut)
#' @export
assessRaster <- function(reference,
                         predicted,
                         multiclass=TRUE,
                         mappings=levels(as.factor(reference)),
                         decimals=4){

    stackG <- c(predicted, reference)
    t1 <- terra::crosstab(stackG, long = FALSE, useNA = FALSE)

    if(multiclass == TRUE){
      colnames(t1) <- mappings
      rownames(t1) <- mappings
      dimnames(t1) <- setNames(dimnames(t1),c("Predicted", "Reference"))

      diag1 <- diag(t1)
      col1 <- colSums(t1)
      row1 <- rowSums(t1)

      pa <- diag1/col1
      ua <- diag1/row1
      names(pa) <- mappings
      names(ua) <- mappings

      f1 <- (2*pa*ua)/(pa+ua)
      names(f1) <- mappings

      aUA <- mean(ua)
      aPA <- mean(pa)
      oa <- sum(diag1)/sum(t1)
      aF1 <- (2*aUA*aPA)/(aUA+aPA)
      results <- list(Classes = mappings,
                      referenceCounts = col1,
                      predictionCounts = row1,
                      confusionMatrix = t1,
                      aggMetrics = data.frame(OA = round(oa, digits=4),
                                              macroF1 = round(aF1, digits=decimals),
                                              macroPA = round(aPA, digits=decimals),
                                              macroUA = round(aUA, digits=decimals)),
                      userAccuracies = round(ua, digits=decimals),
                      producerAccuracies = round(pa, digits=decimals),
                      f1Scores = round(f1, digits=decimals))
    }else{
      colnames(t1) <- c("Negative", "Positive")
      rownames(t1) <- c("Negative", "Positive")
      dimnames(t1) <- setNames(dimnames(t1),c("Predicted", "Reference"))

      diag1 <- diag(t1)
      col1 <- colSums(t1)
      row1 <- rowSums(t1)

      pa <- diag1/col1
      ua <- diag1/row1
      names(pa) <- mappings
      names(ua) <- mappings

      f1 <- (2*pa*ua)/(pa+ua)
      names(f1) <- mappings

      aUA <- mean(ua)
      aPA <- mean(pa)
      oa <- sum(diag1)/sum(t1)
      f1bi <- (2*ua[2]*pa[2])/(ua[2]+pa[2])
      results <- list(Classes = mappings,
                      referenceCounts = col1,
                      predictionCounts = row1,
                      ConfusionMatrix = t1,
                      Mets = data.frame(OA = round(oa, digits=decimals),
                                        Recall = round(pa[2], digits=decimals),
                                        Precision = round(ua[2], digits=decimals),
                                        Specificity = round(pa[1], digits=decimals),
                                        NPV = round(ua[1], digits=decimals),
                                        F1Score = round(f1bi, digits=decimals)
                      )
      )
    }

    return(results)

}

