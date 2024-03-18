#' luz_metric_recall
#'
#' luz_metric function to calculate recall
#'
#' Calculates recall based on luz_metric() for use within training and validation
#' loops.
#'
#' @param preds Tensor of class predicted probabilities with shape
#' (Mini-Batch, Class Logits, Width, Height) and a 32-bit float data type.
#' @param target Tensor of target class labels with shape
#' (Mini-Batch, Class Indices, Width, Height) and a long integer data type. For
#' binary classification, the class index must be 1 for the positive class and
#' 0 for the background case.
#' @param nCLs number of classes being differentiated.
#' @param smooth a smoothing factor to avoid divide by zero errors. Default is 1.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param usedDS TRUE or FALSE. If deep supervision was implemented and masks are produced at varying scales using
#' the defineSegDataSetDS() function, this should be set to TRUE. Only the original resolution is used
#' to calculate assessment metrics. Default is FALSE.
#' @return Calculated metric returned as a base-R vector as opposed to tensor.
#' @export
luz_metric_recall <- luz::luz_metric(

  abbrev = "recall",

  initialize = function(nCls=3,
                        smooth=1,
                        mode = "multiclass",
                        biThresh = 0.5,
                        zeroStart=TRUE,
                        usedDS = TRUE){

    self$nCls <- nCls
    self$smooth <- smooth
    self$mode <- mode
    self$biThresh <- biThresh
    self$zeroStart <- zeroStart
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
      target <- target[[1]]
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
    base::mean((self$tps + self$smooth)/(self$tps + self$fns + self$smooth))
  }
)


#' luz_metric_precision
#'
#' luz_metric function to calculate precision
#'
#' Calculates precision based on luz_metric() for use within training and validation
#' loops.
#'
#' @param preds Tensor of class predicted probabilities with shape
#' (Mini-Batch, Class Logits, Width, Height) and a 32-bit float data type.
#' @param target Tensor of target class labels with shape
#' (Mini-Batch, Class Indices, Width, Height) and a long integer data type. For
#' binary classification, the class index must be 1 for the positive class and
#' 0 for the background case.
#' @param nCLs number of classes being differentiated.
#' @param smooth a smoothing factor to avoid divide by zero errors. Default is 1.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @return Calculated metric returned as a base-R vector as opposed to tensor.
#' @export
luz_metric_precision <- luz::luz_metric(

  abbrev = "precision",

  initialize = function(nCls=3,
                        smooth=1,
                        mode = "multiclass",
                        biThresh=0.5,
                        zeroStart=TRUE,
                        usedDS=TRUE){

    self$nCls <- nCls
    self$smooth <- smooth
    self$mode <- mode
    self$biThresh <- biThresh
    self$zeroStart <- zeroStart
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
      target <- target[[1]]
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
    base::mean((self$tps + self$smooth)/(self$tps + self$fps + self$smooth))
  }
)

#' luz_metric_f1score
#'
#' luz_metric function to calculate the F1-score
#'
#' Calculates F1-score based on luz_metric() for use within training and validation
#' loops.
#'
#' @param preds Tensor of class predicted probabilities with shape
#' (Mini-Batch, Class Logits, Width, Height) and a 32-bit float data type.
#' @param target Tensor of target class labels with shape
#' (Mini-Batch, Class Indices, Width, Height) and a long integer data type. For
#' binary classification, the class index must be 1 for the positive class and
#' 0 for the background case.
#' @param nCLs number of classes being differentiated.
#' @param smooth a smoothing factor to avoid divide by zero errors. Default is 1.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @return Calculated metric returned as a base-R vector as opposed to tensor.
#' @export
luz_metric_f1score <- luz::luz_metric(

  abbrev = "F1Score",

  initialize = function(nCls=1,
                        smooth=1,
                        mode = "multiclass",
                        biThresh = 0.5,
                        zeroStart=TRUE,
                        usedDS = FALSE){

    self$nCls <- nCls
    self$smooth <- smooth
    self$mode <- mode
    self$biThresh <- biThresh
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
      target <- target[[1]]
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
    #Calculate f1-score: (2*tps + smooth)/(2*tps + fns + fps + smooth)
    #Not sure if this is the best way to do this or if it should be calculated direclty from precision and recall as (2*precision*recall)/(precision+recall)
    recalls <- (self$tps + self$smooth)/(self$tps + self$fns + self$smooth)
    precisions <- (self$tps + self$smooth)/(self$tps + self$fps + self$smooth)

    recAgg <- base::mean(recalls)
    precAgg <- base::mean(precisions)

    f1s <- (2.0*precAgg*recAgg)/(precAgg + recAgg + 1e-8)
    return(f1s)
  }
)




#' luz_metric_overall_accuracy
#'
#' luz_metric function to calculate overall accuracy ((correct/total)*100)
#'
#' @param preds Tensor of class predicted probabilities with shape
#' (Mini-Batch, Class Logits, Width, Height) and a 32-bit float data type.
#' @param target Tensor of target class labels with shape
#' (Mini-Batch, Class Indices, Width, Height) and a long integer data type. For
#' binary classification, the class index must be 1 for the positive class and
#' 0 for the background case.
#' @param nCLs number of classes being differentiated.
#' @param smooth a smoothing factor to avoid divide by zero errors. Default is 1.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @return Calculated metric returned as a base-R vector as opposed to tensor.
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
      target <- target[[1]]
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
