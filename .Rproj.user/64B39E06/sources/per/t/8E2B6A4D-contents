#' Assess semantic segmentation model using categorical raster grids (wall-to-wall
#' reference data and predictions)
#'
#' This function will generate a set of summary metrics when provided
#' reference and predicted classes. For a multiclass classification problem
#' a confusion matrix is produced with the columns representing the reference
#' data and the rows representing the predictions. The following metrics are
#' calculated: overall accuracy (OA), 95% confidence interval for OA
#' (OAU and OAL), the Kappa statistic, map image classification
#' efficacy (MICE), average class user's accuracy (aUA), average class
#' producer's accuracy (aPA), average class F1-score, overall error (Error),
#' allocation disagreement (Allocation), quantity disagreement (Quantity),
#' exchange disagreement (Exchange), and shift disagreement (shift). For average
#' class user's accuracy, producer's accuracy, and F1-score, macro-averaging
#' is used where all classes are equally weighted. For a multiclass classification
#' all class user's and producer's accuracies are also returned.
#'
#' For a binary classification problem, a confusion matrix is returned
#' along with the following metrics: overall accuracy (OA), overall accuracy
#' 95% confidence interval (OAU and OAL), the Kappa statistic (Kappa), map
#' image classification efficacy (MICE), precision (Precision), recall (Recall),
#' F1-score (F1), negative predictive value (NPV), specificity (Specificity),
#' overall error (Error), allocation disagreement (Allocation), quantity
#' disagreement (Quantity), exchange disagreement (Exchange), and shift
#' disagreement (shift).
#'
#' Results are returned as a list object. This function makes use of the caret,
#' diffeR, and rfUtilities packages.
#'
#' @param reference Single-band, categorical spatRaster object representing
#' the reference labels. Note that the reference and predicted data must have
#' the same extent, number of rows and columns, and coordinate reference system.
#' @param predicted Single-band, categorical spatRaster object representing the
#' predicted labels. Note that the reference and predicted data must have the '
#' same extent, number of rows and columns, and coordinate reference system.
#' @param mappings Vector of factor level names. These must be in the same order
#' as the factor levels so that they are correctly matched to the correct category.
#' If no mappings are provided, then the factor levels are used by default. This
#' parameter can be especially useful when using raster data as input as it
#' allows the grid codes to be associated with more meaningful labels.
#' @param positive_case Factor level associated with the positive case for a
#' binary classification. Default is the second factor level. This argument is
#' not used for multiclass classification.
#' @return List object containing the resulting metrics. For multiclass assessment,
#' the confusion matrix is provided in the $ConfusionMatrix object, the aggregated
#' metrics are provided in the $Metrics object, class user's accuracies are provided
#' in the $UsersAccs object, class producer's accuracies are provided in the
#' $ProducersAccs object, and the list of classes are provided in the $Classes object.
#' For a binary classification, the confusion matrix is provided in the
#' $ConfusionMatrix object, the overall metrics are provided in the $Metrics object,
#' the classes are provided in the $Classes object, and the positive class label is
#' provided in the $PositiveCase object.
#' @export
assessRaster <- function(reference,
                         predicted,
                         multiclass=TRUE,
                         mappings,
                         positive_case=mappings[1]){

    stackG <- c(reference, predicted)
    ctab <- terra::crosstab(stackG, long = FALSE, useNA = FALSE)

    if(multiclass == TRUE){
      colnames(ctab) <- mappings
      rownames(ctab) <- mappings
      dimnames(ctab) <- setNames(dimnames(ctab),c("Predicted", "Reference"))
      cm <- caret::confusionMatrix(ctab, mode ="everything", positive = positive_case)
      rfu <- rfUtilities::accuracy(ctab)
      Error <- diffeR::overallDiff(ctab)/sum(ctab)
      Allocation <- diffeR::overallAllocD(ctab)/sum(ctab)
      Quantity <- diffeR::overallQtyD(ctab)/sum(ctab)
      Exchange <- diffeR::overallExchangeD(ctab)/sum(ctab)
      Shift <- diffeR::overallShiftD(ctab)/sum(ctab)

      toa <- sum(diag(ctab))/sum(ctab)
      t1row <- colSums(ctab)
      t1ra <- (rfu$users.accuracy/100)+.00001
      t1njn <- (t1row/sum(t1row))+.00001
      t1re <- ((t1ra-t1njn)/(1-t1njn))+.00001
      t1njn2 <- t1njn*t1njn
      mice <- (toa-sum(t1njn2))/(1-sum(t1njn2))

      rfu$users.accuracy[is.na(rfu$users.accuracy)] <- 0
      rfu$producers.accuracy[is.na(rfu$producers.accuracy)] <- 0
      rfu$f <- (2*rfu$users.accuracy*rfu$producers.accuracy)/((rfu$users.accuracy+rfu$producers.accuracy)+.00001)

      final_metrics <- list(ConfusionMatrix = cm$table,
                            Metrics = data.frame(OA = unname(cm$overall[1]),
                                                 OAU = unname(cm$overall[3]),
                                                 OAL = unname(cm$overall[4]),
                                                 Kappa = unname(cm$overall[2]),
                                                 MICE = mice,
                                                 aUA = mean(rfu$users.accuracy),
                                                 aPA = mean(rfu$producers.accuracy),
                                                 aFS = mean(rfu$f),
                                                 Error = Error,
                                                 Allocation = Allocation,
                                                 Quantity = Quantity,
                                                 Exchange = Exchange,
                                                 Shift = Shift),
                            UsersAccs = rfu$users.accuracy/100,
                            ProducersAccs = rfu$producers.accuracy/100,
                            Classes = mappings)

    }else{
      colnames(ctab) <- mappings
      rownames(ctab) <- mappings
      dimnames(ctab) <- setNames(dimnames(ctab),c("Predicted", "Reference"))
      cm <- caret::confusionMatrix(ctab, mode="everything", positive=positive_case)
      Error <- diffeR::overallDiff(ctab)/sum(ctab)
      Allocation <- diffeR::overallAllocD(ctab)/sum(ctab)
      Quantity <- diffeR::overallQtyD(ctab)/sum(ctab)
      Exchange <- diffeR::overallExchangeD(ctab)/sum(ctab)
      Shift <- diffeR::overallShiftD(ctab)/sum(ctab)

      rfu <- rfUtilities::accuracy(ctab)

      toa <- sum(diag(ctab))/sum(ctab)
      t1row <- colSums(ctab)
      t1ra <- (rfu$users.accuracy/100)+.00001
      t1njn <- (t1row/sum(t1row))+.00001
      t1re <- ((t1ra-t1njn)/(1-t1njn))+.00001
      t1njn2 <- t1njn*t1njn
      mice <- (toa-sum(t1njn2))/(1-sum(t1njn2))

      final_metrics <- list(ConfusionMatrix=cm$table,
                            Metrics = data.frame(OA = unname(cm$overall[1]),
                                                 OAU = unname(cm$overall[3]),
                                                 OAL = unname(cm$overall[4]),
                                                 Kappa = unname(cm$overall[2]),
                                                 MICE = mice,
                                                 Precision = unname(cm$byClass[5]),
                                                 Recall = unname(cm$byClass[6]),
                                                 F1 = unname(cm$byClass[7]),
                                                 NPV = unname(cm$byClass[4]),
                                                 Specificity = unname(cm$byClass[2]),
                                                 Error = Error,
                                                 Allocation = Allocation,
                                                 Quantity = Quantity,
                                                 Exchange = Exchange,
                                                 Shift = Shift),
                            Classes = mappings,
                            PostiveCase = cm$positive)

    }

    return(final_metrics)

}
