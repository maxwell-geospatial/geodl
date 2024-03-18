getMets <- function(ctab, mode="multiclass", positive_case="positive", cNames){
  if(mode == "multiclass"){
    rfu <- rfUtilities::accuracy(ctab)

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

    final_metrics <- list(ConfusionMatrix=ctab,
                          Metrics = data.frame(OA = rfu$PCC/100,
                                               Kappa = rfu$kappa,
                                               MICE = mice,
                                               aUA = mean(rfu$users.accuracy)/100,
                                               aPA = mean(rfu$producers.accuracy)/100),
                          UsersAccs = rfu$users.accuracy/100,
                          ProducersAccs = rfu$producers.accuracy/100,
                          Classes = cNames)

    final_metrics$Metrics$aF1 <- (2*
                                   final_metrics$Metrics$aUA*
                                   final_metrics$Metrics$aPA)/
      (final_metrics$Metrics$aUA+
         final_metrics$Metrics$aPA)

  }else{
    rfu <- rfUtilities::accuracy(ctab)

    toa <- sum(diag(ctab))/sum(ctab)
    t1row <- colSums(ctab)
    t1ra <- (rfu$users.accuracy/100)+.00001
    t1njn <- (t1row/sum(t1row))+.00001
    t1re <- ((t1ra-t1njn)/(1-t1njn))+.00001
    t1njn2 <- t1njn*t1njn
    mice <- (toa-sum(t1njn2))/(1-sum(t1njn2))

    final_metrics <- list(ConfusionMatrix=ctab,
                          Metrics = data.frame(OA = rfu$PCC/100,
                                               Kappa = rfu$kappa,
                                               MICE = mice,
                                               Precision = unname(cm$byClass[5][2])/100,
                                               Recall = unname(cm$byClass[6][2])/100,
                                               NPV = unname(cm$byClass[4][1])/100,
                                               Specificity = unname(cm$byClass[2][1])/100),
                          Classes = cNames,
                          PostiveCase = postive_case)

      final_metrics$Metrics$F1 <- (2*
                                     final_metrics$Metrics$Recall*
                                     final_metrics$Metrics$Precision)/
        (final_metrics$Metrics$Recall+
           final_metrics$Metrics$Precision)

  }

  return(final_metrics)

}


#' assessDL
#'
#' Assess semantic segmentation model using all samples in a DataLoader.
#'
#' This function generates a set of summary metrics when provided
#' reference and predicted classes. For a multiclass classification problem
#' a confusion matrix is produced with the columns representing the reference
#' data and the rows representing the predictions. The following metrics are
#' calculated: overall accuracy (OA), the Kappa statistic, map image classification
#' efficacy (MICE), average class user's accuracy (aUA), average class
#' producer's accuracy (aPA), and average class F1-score (aF1). For average
#' class user's accuracy, producer's accuracy, and F1-score, macro-averaging
#' is used where all classes are equally weighted. For a multiclass classification
#' all class user's and producer's accuracies are also returned.
#'
#' For a binary classification problem, a confusion matrix is returned
#' along with the following metrics: overall accuracy (OA), overall accuracy,
#' the Kappa statistic (Kappa), map
#' image classification efficacy (MICE), precision (Precision), recall (Recall),
#' F1-score (F1), negative predictive value (NPV), and specificity (Specificity).

#'
#' Results are returned as a list object. This function makes use of the rfUtilities package.
#'
#' @param dl torch DataLoader object.
#' @param nCls number of classes being differentiated.
#' @param multiclass TRUE or FALSE. If more than two classes are differentiated,
#' use TRUE. If only two classes are differentiated and there are positive and
#' background/negative classes, use FALSE. Default is TRUE.
#' @param cCodes class indices as a vector of integer values equal in length to the number of
#' classes.
#' @param cNames class names as a vector of character strings with a length equal to the number of
#' classes and in the correct order. Class codes and names are matched by position in the
#' cCodes and cNames vectors.
#' @param positive_case Factor level associated with the positive case for a
#' binary classification. Default is the second factor level. This argument is
#' not used for multiclass classification.
#' @return List object containing the resulting metrics. For multiclass assessment,
#' the confusion matrix is provided in the $ConfusionMatrix object, the aggregated
#' metrics are provided in the $Metrics object, class user's accuracies are provided
#' in the $UsersAccs object, class producer's accuracies are provided in the
#' $ProducersAccs object, and the list of classes are provided in the $Classes object.
#' For a binary classification, the confusion matrix is provided in the
#' $ConfusionMatrix object, the metrics are provided in the $Metrics object,
#' the classes are provided in the $Classes object, and the positive class label is
#' provided in the $PositiveCase object.
#' @export
assessDL <- function(dl,
                     model,
                     batchSize,
                     size,
                     nCls,
                     mode,
                     cCodes,
                     cNames,
                     usedDS,
                     useCUDA,
                     positive_case){

  cm <- data.frame(Prediction=as.character(),
                   Reference=as.character(),
                   n=as.numeric())

  clsTbl <- data.frame(id=cCodes,
                       classes = cNames)
  if(usedDS == TRUE){
    model2 <- model$model
  }else{
    model <- model
  }

  # disable gradient tracking to reduce memory usage
  with_no_grad({
    coro::loop(for (b in dl) {
        if(usedDS == TRUE){
          masks <- b$mask[[1]]
          images <- b$image
        }else{
          masks <- b$mask
          images <- b$image
        }

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

  cmTable <- xtabs(n ~ Prediction + Reference, data = cm)

  theMets <- getMets(ctab=cmTable,
                      mode=mode,
                      positive_case=positive_case,
                      cNames=cNames)
  return(theMets)

}


