#' luz_metric_recall
#'
#' luz_metric function to calculate macro-averaged, class aggregated recall
#'
#' Calculates recall based on luz_metric() for use within training and validation
#' loops.
#'
#' @param nCls Number of classes being differentiated.
#' @param smooth A smoothing factor to avoid divide by zero errors. Default is 1.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' the positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode. Note that this package is designed to treat all predictions as multiclass.
#' The "binary" mode is only provided for use outside of the standard geodl workflow.
#' @param biThresh Probability threshold to define postive case prediction. Default is 0.5.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param clsWghts Vector of class weightings loss calculatokn. Default is equal weightings.
#' @param usedDS TRUE or FALSE. If deep supervision was implemented and masks are produced at varying scales using
#' the defineSegDataSetDS() function, this should be set to TRUE. Only the original resolution is used
#' to calculate assessment metrics. Default is FALSE.
#' @return Calculated metric returned as a base-R vector as opposed to tensor.
#' @examples
#' \donttest{
#' library(terra)
#' library(torch)
#' #Generate example data as SpatRasters
#' ref <- terra::rast(matrix(sample(c(1, 2, 3), 625, replace=TRUE), nrow=25, ncol=25))
#' pred1 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred2 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred3 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred <- c(pred2, pred2, pred3)
#'
#' #Convert SpatRaster to array
#' ref <- terra::as.array(ref)
#' pred <- terra::as.array(pred)
#'
#' #Convert arrays to tensors and reshape
#' ref <- torch::torch_tensor(ref, dtype=torch::torch_long())
#' pred <- torch::torch_tensor(pred, dtype=torch::torch_float32())
#' ref <- ref$permute(c(3,1,2))
#' pred <- pred$permute(c(3,1,2))
#'
#' #Add mini-batch dimension
#' ref <- ref$unsqueeze(1)
#' pred <- pred$unsqueeze(1)
#'
#' #Duplicate tensors to have a batch of two
#' ref <- torch::torch_cat(list(ref, ref), dim=1)
#' pred <- torch::torch_cat(list(pred, pred), dim=1)
#'
#' #Calculate Macro-Averaged, Class Aggregated Recall
#' metric<-luz_metric_recall(nCls=3,
#'                           smooth=1e-8,
#'                           mode = "multiclass",
#'                           zeroStart=FALSE,
#'                           usedDS=FALSE)
#' metric<-metric$new()
#' metric$update(pred,ref)
#' metric$compute()
#' }
#' @export
luz_metric_recall <- luz::luz_metric(

  abbrev = "recall",

  initialize = function(nCls=3,
                        smooth=1,
                        mode = "multiclass",
                        biThresh = 0.5,
                        zeroStart=TRUE,
                        clsWghts=rep(1.0, nCls),
                        usedDS = TRUE){

    self$nCls <- nCls
    self$smooth <- smooth
    self$mode <- mode
    self$biThresh <- biThresh
    self$zeroStart <- zeroStart
    self$clsWghts <- clsWghts
    self$usedDS <- usedDS

    if(self$mode == "multiclass"){
      self$tps <- rep(0.0, nCls)
      self$fns <- rep(0.0, nCls)
      self$fps <- rep(0.0, nCls)
    }else{
      self$tps <- 0.0
      self$fns <- 0.0
      self$fps <- 0.0
    }
  },

  update = function(preds, target){
    if(self$usedDS == TRUE){
      preds <- preds[[1]]
      target <- target
    }

    if(self$mode == "multiclass"){
      predsMax <- torch::torch_argmax(preds, dim = 2)
      target1 <- torch::torch_tensor(target, dtype=torch::torch_long())

      if(self$zeroStart == TRUE){
        target1 <- torch::torch_tensor(target1+1, dtype=torch::torch_long())
      }

      preds_one_hot <- torch::nnf_one_hot(predsMax, num_classes = self$nCls)
      preds_one_hot <- preds_one_hot$permute(c(1,4,2,3))

      target_one_hot <- torch::nnf_one_hot(target1, num_classes = self$nCls)
      target_one_hot <- target_one_hot$squeeze()
      target_one_hot <- target_one_hot$permute(c(1,4,2,3))

      dims <- c(1, 3, 4)

      self$tps <- self$tps + (torch::torch_sum(preds_one_hot * target_one_hot, dims)$cpu() |>
                                torch::as_array() |>
                                as.vector())
      self$fps <- self$fps + (torch::torch_sum(preds_one_hot * (-target_one_hot + 1.0), dims)$cpu() |>
                                torch::as_array() |>
                                as.vector())
      self$fns <- self$fns + (torch::torch_sum((-preds_one_hot + 1.0) * target_one_hot, dims)$cpu() |>
                                torch::as_array() |>
                                as.vector())

    }else{
      preds <- torch::nnf_sigmoid(preds)
      preds <- preds > self$biThresh
      preds <- torch::torch_tensor(preds, dtype=torch::torch_float32())
      preds <- preds$flatten()
      target <- target$flatten()

      self$tps <- self$tps + sum(preds * target)$cpu() |>
        torch::as_array() |>
        as.vector()
      self$fps <- self$fps + sum((1.0 - target) * preds)$cpu() |>
        torch::as_array() |>
        as.vector()
      self$fns <- self$fns + sum(target * (1.0 - preds))$cpu() |>
        torch::as_array() |>
        as.vector()
    }
  },

  compute = function(){
    #Calculate recall: (tps + smooth)/(tps + fns + smooth)
    stats::weighted.mean(((self$tps + self$smooth)/(self$tps + self$fns + self$smooth)), self$clsWghts)
  }
)


#' luz_metric_precision
#'
#' luz_metric function to calculate macro-averaged, class aggregated precision
#'
#' Calculates precision based on luz_metric() for use within training and validation
#' loops.
#'
#' @param nCls Number of classes being differentiated.
#' @param smooth A smoothing factor to avoid divide by zero errors. Default is 1.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' the positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode. Note that this package is designed to treat all predictions as multiclass.
#' The "binary" mode is only provided for use outside of the standard geodl workflow.
#' @param biThresh Probability threshold to define postive case prediction. Default is 0.5.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param clsWghts Vector of class weightings loss calculatokn. Default is equal weightings.
#' @param usedDS TRUE or FALSE. If deep supervision was implemented and masks are produced at varying scales using
#' the defineSegDataSetDS() function, this should be set to TRUE. Only the original resolution is used
#' to calculate assessment metrics. Default is FALSE.
#' @return Calculated metric returned as a base-R vector as opposed to tensor.
#' @examples
#' \donttest{
#' library(terra)
#' library(torch)
#' #Generate example data as SpatRasters
#' ref <- terra::rast(matrix(sample(c(1, 2, 3), 625, replace=TRUE), nrow=25, ncol=25))
#' pred1 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred2 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred3 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred <- c(pred2, pred2, pred3)
#'
#' #Convert SpatRaster to array
#' ref <- terra::as.array(ref)
#' pred <- terra::as.array(pred)
#'
#' #Convert arrays to tensors and reshape
#' ref <- torch::torch_tensor(ref, dtype=torch::torch_long())
#' pred <- torch::torch_tensor(pred, dtype=torch::torch_float32())
#' ref <- ref$permute(c(3,1,2))
#' pred <- pred$permute(c(3,1,2))
#'
#' #Add mini-batch dimension
#' ref <- ref$unsqueeze(1)
#' pred <- pred$unsqueeze(1)
#'
#' #Duplicate tensors to have a batch of two
#' ref <- torch::torch_cat(list(ref, ref), dim=1)
#' pred <- torch::torch_cat(list(pred, pred), dim=1)
#' #Calculate Macro-Averaged, Class Aggregated Precision
#' metric<-luz_metric_precision(nCls=3,
#'                              smooth=1e-8,
#'                              mode = "multiclass",
#'                              zeroStart=FALSE,
#'                              usedDS=FALSE)
#' metric<-metric$new()
#' metric$update(pred,ref)
#' metric$compute()
#' }
#' @export
luz_metric_precision <- luz::luz_metric(

  abbrev = "precision",

  initialize = function(nCls=3,
                        smooth=1,
                        mode = "multiclass",
                        biThresh=0.5,
                        zeroStart=TRUE,
                        clsWghts = rep(1, nCls),
                        usedDS=TRUE){

    self$nCls <- nCls
    self$smooth <- smooth
    self$mode <- mode
    self$biThresh <- biThresh
    self$zeroStart <- zeroStart
    self$clsWghts <- clsWghts
    self$usedDS <- usedDS

    #initialize R vectors to store true positive, false negative, and false positive counts
    #For a binary and micro-averaged multiclass metric, will obtain a vector with a length of one.
    #For a macro-averaged multiclass, will obtain a vector with a lenght equal to the number of classes.
    if(self$mode == "multiclass"){
      self$tps <- rep(0.0, nCls)
      self$fns <- rep(0.0, nCls)
      self$fps <- rep(0.0, nCls)
    }else{
      self$tps <- 0.0
      self$fns <- 0.0
      self$fps <- 0.0
    }
  },

  update = function(preds, target){
    if(self$usedDS == TRUE){
      preds <- preds[[1]]
      target <- target
    }

    #For multiclass problems
    if(self$mode == "multiclass"){
      #Get index of class with largest logit.
      predsMax <- torch::torch_argmax(preds, dim = 2)
      #Make sure target is in type torch_long()
      target1 <- torch::torch_tensor(target, dtype=torch::torch_long())

      if(self$zeroStart == TRUE){
        #If class indices start at zero, add 1.
        target1 <- torch::torch_tensor(target+1, dtype=torch::torch_long())
      }

      #One-hot encode the prediction results that have been passed through argmax()
      preds_one_hot <- torch::nnf_one_hot(predsMax, num_classes = self$nCls)
      #Permute the results so that the dimension order is [batch, encodings, height, width].
      preds_one_hot <- preds_one_hot$permute(c(1,4,2,3))

      #one-hot encode the targets.
      target_one_hot <- torch::nnf_one_hot(target1, num_classes = self$nCls)
      #Remove channel dimension.
      target_one_hot <- target_one_hot$squeeze()
      #permute the results so that the order is [batch, encoding, height, width]
      target_one_hot <- target_one_hot$permute(c(1,4,2,3))


      #If using macro averaging, calculation of tps, fps, and fns should be performed separately for each class.
      dims <- c(1, 3, 4)

      #Calculate true positives and add to running true positives count.
      self$tps <- self$tps + (torch::torch_sum(preds_one_hot * target_one_hot, dims)$cpu() |>
                                torch::as_array() |>
                                as.vector())
      #Calculate false positives and add to running true positives count.
      self$fps <- self$fps + (torch::torch_sum(preds_one_hot * (-target_one_hot + 1.0), dims)$cpu() |>
                                torch::as_array() |>
                                as.vector())
      #Calculate false negatives and add to running false negative count.
      self$fns <- self$fns + (torch::torch_sum((-preds_one_hot + 1.0) * target_one_hot, dims)$cpu() |>
                                torch::as_array() |>
                                as.vector())

    #Processing for binary classification
    }else{
      #Convert logits to probs using sigmoid function
      preds <- torch::nnf_sigmoid(preds)
      #Round to convert probs to 0 and 1.
      preds <- preds > self$biThresh
      preds <- torch::torch_tensor(preds, dtype=torch::torch_float32())
      #Flatten arrays (this generalizes the problem so that the shape of the input tensors does not matter)
      preds <- preds$flatten()
      target <- target$flatten()

      #Calculate true positives and add to running true positives count.
      self$tps <- self$tps + sum(preds * target)$cpu() |>
        torch::as_array() |>
        as.vector()
      #Calculate false positives and add to running true positives count.
      self$fps <- self$fps + sum((1.0 - target) * preds)$cpu() |>
        torch::as_array() |>
        as.vector()
      #Calculate false negatives and add to running false negative count.
      self$fns <- self$fns + sum(target * (1.0 - preds))$cpu() |>
        torch::as_array() |>
        as.vector()
    }
  },

  compute = function(){
    #Calculate precision: (tps + smooth)/(tps + fps + smooth)
    stats::weighted.mean(((self$tps + self$smooth)/(self$tps + self$fps + self$smooth)), self$clsWghts)
  }
)

#' luz_metric_f1score
#'
#' luz_metric function to calculate the macro-averaged, class aggregated F1-score
#'
#' Calculates F1-score based on luz_metric() for use within training and validation
#' loops.
#'
#' @param nCls Number of classes being differentiated.
#' @param smooth A smoothing factor to avoid divide by zero errors. Default is 1.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' the positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode. Note that this package is designed to treat all predictions as multiclass.
#' The "binary" mode is only provided for use outside of the standard geodl workflow.
#' @param biThresh Probability threshold to define postive case prediction. Default is 0.5.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param clsWghts Vector of class weightings loss calculatokn. Default is equal weightings.
#' @param usedDS TRUE or FALSE. If deep supervision was implemented and masks are produced at varying scales using
#' the defineSegDataSetDS() function, this should be set to TRUE. Only the original resolution is used
#' to calculate assessment metrics. Default is FALSE.
#' @return Calculated metric returned as a base-R vector as opposed to tensor.
#' @examples
#' \donttest{
#' library(terra)
#' library(torch)
#' #Generate example data as SpatRasters
#' ref <- terra::rast(matrix(sample(c(1, 2, 3), 625, replace=TRUE), nrow=25, ncol=25))
#' pred1 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred2 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred3 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred <- c(pred2, pred2, pred3)
#'
#' #Convert SpatRaster to array
#' ref <- terra::as.array(ref)
#' pred <- terra::as.array(pred)
#'
#' #Convert arrays to tensors and reshape
#' ref <- torch::torch_tensor(ref, dtype=torch::torch_long())
#' pred <- torch::torch_tensor(pred, dtype=torch::torch_float32())
#' ref <- ref$permute(c(3,1,2))
#' pred <- pred$permute(c(3,1,2))
#'
#' #Add mini-batch dimension
#' ref <- ref$unsqueeze(1)
#' pred <- pred$unsqueeze(1)
#'
#' #Duplicate tensors to have a batch of two
#' ref <- torch::torch_cat(list(ref, ref), dim=1)
#' pred <- torch::torch_cat(list(pred, pred), dim=1)
#'
#' #Calculate Macro-Averaged, Class Aggregated F1-Score
#' metric<-luz_metric_f1score(nCls=3,
#'                            smooth=1e-8,
#'                            mode = "multiclass",
#'                            zeroStart=FALSE,
#'                            usedDS=FALSE)
#' metric<-metric$new()
#' metric$update(pred,ref)
#' metric$compute()
#' }
#' @export
luz_metric_f1score <- luz::luz_metric(

  abbrev = "F1Score",

  initialize = function(nCls=1,
                        smooth=1,
                        mode = "multiclass",
                        biThresh = 0.5,
                        clsWghts = rep(1, nCls),
                        zeroStart=TRUE,
                        usedDS = FALSE){

    self$nCls <- nCls
    self$smooth <- smooth
    self$mode <- mode
    self$biThresh <- biThresh
    self$clsWghts <- clsWghts
    self$zeroStart <- zeroStart
    self$usedDS <- usedDS

    if(self$mode == "multiclass"){
      self$tps <- rep(0.0, nCls)
      self$fns <- rep(0.0, nCls)
      self$fps <- rep(0.0, nCls)
    }else{
     self$tps <- 0
     self$fns <- 0
     self$fps <- 0
    }
  },

  update = function(preds, target){
    if(self$usedDS == TRUE){
      preds <- preds[[1]]
      target <- target
    }

    if(self$mode == "multiclass"){
      predsMax <- torch::torch_argmax(preds, dim = 2)
      target1 <- torch::torch_tensor(target, dtype=torch::torch_long())

      if(self$zeroStart == TRUE){
        target1 <- torch::torch_tensor(target+1, dtype=torch::torch_long())
      }

      preds_one_hot <- torch::nnf_one_hot(predsMax, num_classes = self$nCls)
      preds_one_hot <- preds_one_hot$permute(c(1,4,2,3))

      target_one_hot <- torch::nnf_one_hot(target1, num_classes = self$nCls)
      target_one_hot <- target_one_hot$squeeze()
      target_one_hot <- target_one_hot$permute(c(1,4,2,3))

      dims <- c(1, 3, 4)

      self$tps <- self$tps + (torch::torch_sum(preds_one_hot * target_one_hot, dims)$cpu() |>
        torch::as_array() |>
        as.vector())
      self$fps <- self$fps + (torch::torch_sum(preds_one_hot * (-target_one_hot + 1.0), dims)$cpu() |>
        torch::as_array() |>
        as.vector())
      self$fns <- self$fns + (torch::torch_sum((-preds_one_hot + 1.0) * target_one_hot, dims)$cpu() |>
        torch::as_array() |>
        as.vector())

    }else{
      preds <- torch::nnf_sigmoid(preds)
      preds <- preds > self$biThresh
      preds <- torch::torch_tensor(preds, dtype=torch::torch_float32())
      preds <- preds$flatten()
      target <- target$flatten()

      self$tps <- self$tps + sum(preds * target)$cpu() |>
        torch::as_array() |>
        as.vector()
      self$fps <- self$fps + sum((1.0 - target) * preds)$cpu() |>
        torch::as_array() |>
        as.vector()
      self$fns <- self$fns + sum(target * (1.0 - preds))$cpu() |>
        torch::as_array() |>
        as.vector()
    }
  },

  compute = function(){
    recalls <- (self$tps + self$smooth)/(self$tps + self$fns + self$smooth)
    precisions <- (self$tps + self$smooth)/(self$tps + self$fps + self$smooth)

    recAgg <- stats::weighted.mean(recalls, self$clsWghts)
    precAgg <- stats::weighted.mean(precisions, self$clsWghts)

    f1s <- (2.0*precAgg*recAgg)/(precAgg + recAgg + 1e-8)
    return(f1s)
  }
)




#' luz_metric_overall_accuracy
#'
#' luz_metric function to calculate overall accuracy ((correct/total)*100)
#'
#' @param nCls Number of classes being differentiated.
#' @param smooth A smoothing factor to avoid divide by zero errors. Default is 1.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' the positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode. Note that this package is designed to treat all predictions as multiclass.
#' The "binary" mode is only provided for use outside of the standard geodl workflow.
#' @param biThresh Probability threshold to define postive case prediction. Default is 0.5.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param usedDS TRUE or FALSE. If deep supervision was implemented and masks are produced at varying scales using
#' the defineSegDataSetDS() function, this should be set to TRUE. Only the original resolution is used
#' to calculate assessment metrics. Default is FALSE.
#' @return Calculated metric returned as a base-R vector as opposed to tensor.
#' @examples
#' \donttest{
#' require(terra)
#' require(torch)
#' #Generate example data as SpatRasters
#' ref <- terra::rast(matrix(sample(c(1, 2, 3), 625, replace=TRUE), nrow=25, ncol=25))
#' pred1 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred2 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred3 <- terra::rast(matrix(sample(c(1:150), 625, replace=TRUE), nrow=25, ncol=25))
#' pred <- c(pred2, pred2, pred3)
#'
#' #Convert SpatRaster to array
#' ref <- terra::as.array(ref)
#' pred <- terra::as.array(pred)
#'
#' #Convert arrays to tensors and reshape
#' ref <- torch::torch_tensor(ref, dtype=torch::torch_long())
#' pred <- torch::torch_tensor(pred, dtype=torch::torch_float32())
#' ref <- ref$permute(c(3,1,2))
#' pred <- pred$permute(c(3,1,2))
#'
#' #Add mini-batch dimension
#' ref <- ref$unsqueeze(1)
#' pred <- pred$unsqueeze(1)
#'
#' #Duplicate tensors to have a batch of two
#' ref <- torch::torch_cat(list(ref, ref), dim=1)
#' pred <- torch::torch_cat(list(pred, pred), dim=1)
#'
#' #Calculate Overall Accuracy
#' metric<-luz_metric_overall_accuracy(nCls=3,
#'                                    smooth=1e-8,
#'                                     mode = "multiclass",
#'                                    zeroStart=FALSE,
#'                                    usedDS=FALSE)
#' metric<-metric$new()
#' metric$update(pred,ref)
#' metric$compute()
#' }
#' @export
luz_metric_overall_accuracy <- luz::luz_metric(

  abbrev = "OverallAcc",

  initialize = function(nCls=1,
                        smooth=1,
                        mode = "multiclass",
                        biThresh=0.5,
                        zeroStart=TRUE,
                        usedDS = FALSE){

    self$nCls <- nCls
    self$smooth <- smooth
    self$mode <- mode
    self$biThresh <- biThresh
    self$zeroStart <- zeroStart
    self$usedDS <- usedDS

    self$correct <- 0
    self$total <- 0
  },

  update = function(preds, target) {

    if(self$usedDS == TRUE){
      preds <- preds[[1]]
      target <- target
    }

    if(self$mode == "multiclass"){

      predsMax <- torch::torch_argmax(preds, dim = 2)
      target1 <- torch::torch_tensor(target, dtype=torch::torch_long())

      if(self$zeroStart == TRUE){
          target1 <- torch::torch_tensor(target1+1, dtype=torch::torch_long())
      }

      preds_one_hot <- torch::nnf_one_hot(predsMax, num_classes = self$nCls)
      preds_one_hot <- preds_one_hot$permute(c(1,4,2,3))

      target_one_hot <- torch::nnf_one_hot(target1, num_classes = self$nCls)
      target_one_hot <- target_one_hot$squeeze()
      target_one_hot <- target_one_hot$permute(c(1,4,2,3))

      currentCnt <- torch::torch_sum(target_one_hot >= 0)$
        to(dtype = torch::torch_float())$cpu() |>
        torch::as_array() |>
        as.vector() |> sum()/self$nCls

      dims <- c(1, 2, 3, 4)

      self$correct <- self$correct + (torch::torch_sum(preds_one_hot * target_one_hot, dims)$cpu() |>
                                torch::as_array() |>
                                as.vector())
      self$total <- self$total + currentCnt
  }else{
    preds <- torch::nnf_sigmoid(preds)
    preds <- preds > self$biThresh
    preds <- torch::torch_tensor(preds, dtype=torch::torch_float32())
    preds <- preds$flatten()
    target <- target$flatten()

    self$correct <- self$correct + torch::torch_sum(preds == target)$
      to(dtype = torch::torch_float())$
      sum()$
      item()
    self$total <- self$total + preds$numel()

  }
  },

  compute = function() {
    oa <- self$correct/self$total
    return(oa)
  }
)
