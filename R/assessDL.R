#' assessDL
#'
#' Assess semantic segmentation model using all samples in a torch DataLoader.
#'
#' This function generates a set of summary assessment metrics based on all samples
#' within a torch data loader. Results are returned as a list object. For
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
#'
#' @param dl torch DataLoader object.
#' @param model trained model object.
#' @param multiclass TRUE or FALSE. If more than two classes are differentiated,
#' use TRUE. If only two classes are differentiated and there are positive and
#' background/negative classes, use FALSE. Default is TRUE. For binary cases, the second
#' class is assumed to be the positive case.
#' @param batchSize Batch size used in torch DataLoader.
#' @param size Size of image chips in spatial dimensions (e.g., 128, 256, 512).
#' @param nCls Number of classes being differentiated.
#' @param cCodes Class indices as a vector of integer values equal in length to the number of
#' classes.
#' @param cNames Class names as a vector of character strings with a length equal to the number of
#' classes and in the correct order. Class codes and names are matched by position in the
#' cCodes and cNames vectors. For binary case, this argument is ignored, and the first class is
#' called "Negative" while the second class is called "Positive".
#' @usedDS TRUE or FALSE. Whether or not deep supervision was used. Default is FALSE, or
#' it is assumed that deep supervision was not used.
#' @useCUDA TRUE or FALSE. Whether or not to use GPU. Default is FALSE, or GPU is not used.
#' We recommend using a CUDA-enabled GPU if one is available since this will speed up computation.
#' @return List object containing the resulting metrics and ancillary information.
#' @param decimals Number of decimal places to return for assessment metrics. Default is 4.
#' @export
assessDL <- function(dl,
                     model,
                     multiclass=TRUE,
                     batchSize,
                     size,
                     nCls,
                     cCodes,
                     cNames,
                     usedDS=FALSE,
                     useCUDA=FALSE,
                     decimals=4){

  cm <- data.frame(Prediction=as.character(),
                   Reference=as.character(),
                   n=as.numeric())

  if(multiclass == TRUE){
    clsTbl <- data.frame(id=cCodes,
                         classes = cNames)
  } else {
    clsTbl <- data.frame(id=cCodes,
                         classes = c("Negative", "Positive"))
  }


  if(usedDS == TRUE){
    model2 <- model$model
  }else{
    model <- model
  }

  # disable gradient tracking to reduce memory usage
  with_no_grad({
    coro::loop(for (b in dl) {

          masks <- b$mask
          images <- b$image

        if(useCUDA == TRUE){
          images <- images$to(device="cuda")
        }

        if(usedDS == TRUE){
          preds <- model2(images)
        }else{
          preds <- predict(model, images)
        }

        if(usedDS==TRUE){
          preds <- preds[[1]]
        }

        coro::loop(for(i in 1:batchSize){
          predi <- preds[i,1:size,1:size]$squeeze(dim=1)
          predi <- torch::torch_argmax(predi, dim=1)
          predi <- predi$unsqueeze(1)$permute(c(2,3,1))$cpu()$to(device="cpu")
          predOut <- terra::rast(as.array(predi))

          predOut <- terra::as.factor(predOut)
          levels(predOut) <- clsTbl
          names(predOut) <- "Prediction"

          refi <- masks[i,1:size,1:size]$squeeze(dim=1)
          refi<- refi$unsqueeze(1)$permute(c(2,3,1))$cpu()$to(device="cpu")
          refOut <- terra::rast(as.array(refi))

          refOut <- terra::as.factor(refOut)
          levels(refOut) <- clsTbl
          names(refOut) <- "Reference"

          stk <- c(predOut, refOut)
          cm2 <- terra::crosstab(stk, long=TRUE)

          cm <- dplyr::bind_rows(cm, cm2)
        })
      })
  })



  if(multiclass == TRUE){
    t1 <- xtabs(n ~ Prediction + Reference, data = cm)

    colnames(t1) <- cNames
    rownames(t1) <- cNames
    dimnames(t1) <- setNames(dimnames(t1),c("Predicted", "Reference"))

    diag1 <- diag(t1)
    col1 <- colSums(t1)
    row1 <- rowSums(t1)

    pa <- diag1/col1
    ua <- diag1/row1
    names(pa) <- cNames
    names(ua) <- cNames

    f1 <- (2*pa*ua)/(pa+ua)
    names(f1) <- cNames

    aUA <- mean(ua)
    aPA <- mean(pa)
    oa <- sum(diag1)/sum(t1)
    aF1 <- (2*aUA*aPA)/(aUA+aPA)
    results <- list(Classes = cNames,
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
    t1 <- xtabs(n ~ Prediction + Reference, data = cm)

    colnames(t1) <- c("Negative", "Positive")
    rownames(t1) <- c("Negative", "Positive")
    dimnames(t1) <- setNames(dimnames(t1),c("Predicted", "Reference"))

    diag1 <- diag(t1)
    col1 <- colSums(t1)
    row1 <- rowSums(t1)

    pa <- diag1/col1
    ua <- diag1/row1
    names(pa) <- cNames
    names(ua) <- cNames

    f1 <- (2*pa*ua)/(pa+ua)
    names(f1) <- cNames

    aUA <- mean(ua)
    aPA <- mean(pa)
    oa <- sum(diag1)/sum(t1)
    f1bi <- (2*ua[2]*pa[2])/(ua[2]+pa[2])
    results <- list(Classes = cNames,
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

}


