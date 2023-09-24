#' luz_metric_recall
#'
#' luz_metric function to calculate recall
#'
#' Calculates recall based on luz_metric() for use within training and validation
#' loops.
#'
#' @param preds Tensor of class predicted probabilities with shape
#' (Batch, Class Logits, Height, Width) for a multiclass classification. For a
#' binary classification, you should provide logits for just the positive class as
#' (Batch, Positive Class Logit, Height, Width) or (Batch, Height, Width).
#' @param target Tensor of target class labels with shape
#' (Batch, Class Indices, Height, Width) for a multiclass classification. For a
#' binary classification, you should provide targets as
#' (Batch, Positive Class Index, Height, Width) or (Batch, Height, Width). For
#' binary classification, the class index must be 1 for the positive class and 0 for the
#' background case.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' positive class prediction should be provided. If both the positive and negative
#' or background class logits are provided for a binary classification, use
#' the "multiclass" mode.
#' @param average Either "micro" or "macro". Whether to use micro- or macro-averaging
#' for multiclass metric calculation. Ignored when mode is "binary". Default is
#' "micro"
#' @return Calculated metric return as a base-R vector as opposed to a tensor
#' @export
luz_metric_recall <- luz::luz_metric(

  abbrev = "Recall",

  initialize = function(mode = "multiclass", average="micro", smooth=1){
    self$tps <- 0.0
    self$fns <- 0.0
    self$fps <- 0.0
    self$smooth <- smooth
    self$mode <- mode
    self$average <- average
  },

  update = function(preds, target){
    if(self$mode == "multiclass"){
      preds <- torch::torch_argmax(preds, dim = 2)
      target1 <- torch::tensor(target+1, dtype=torch::torch_long())

      target_one_hot <- torch::nnf_one_hot(target1$flatten(), num_classes = preds$shape[2])
      target_one_hot <- target_one_hot$permute(c(2,1))$reshape(c(1,2, preds$shape[3],preds$shape[4]))

      dims <- c(3, 4)
      if(self$average == "micro"){
        dims <- c(2, dims)
      }

      self$tps <- self$tps + torch::torch_sum(preds * target_one_hot, dims)$item()
      self$fps <- self$fps + torch::torch_sum(preds * (-target_one_hot + 1.0), dims)$item()
      self$fns <- self$fns + torch::torch_sum((preds + 1.0) * target_one_hot, dims)$item()

    }else{
      preds <- torch::nnf_sigmoid(preds)
      preds <- torch::torch_round(preds)
      preds <- preds$flatten()
      target <- target$flatten()

      self$tps <- self$tps + sum(preds * target)$item()
      self$fps <- self$fps + sum((1.0 - target) * preds)$item()
      self$fns <- self$fns + sum(target * (1.0 - preds))$item()
    }
  },

  compute = function(){
    mean((self$tps + self$smooth)/(self$tps + self$fns + self$smooth))
  }
)

#' luz_metric_recall
#'
#' luz_metric function to calculate precision
#'
#' Calculates precision based on luz_metric() for use within training and validation
#' loops.
#'
#' @param preds Tensor of class predicted probabilities with shape
#' [Batch, Class Logits, Height, Width] for a multiclass classification. For a
#' binary classification, you can provide logits for the positive class as
#' [Batch, Positive Class Logit, Height, Width] or [Batch, Height, Width].
#' @param target Tensor of target class labels with shape
#' [Batch, Class Indices, Height, Width] for a multiclass classification. For a
#' binary classification, you can provide targets as
#' [Batch, Positive Class Index, Height, Width] or [Batch, Height, Width]. For
#' binary classification, the class index must be 1 for the positive class and
#' 0 for the background case.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode.
#' @param average Either "micro" or "macro". Whether to use micro- or macro-averaging
#' for multiclass metric calculation. Ignored when mode is "binary". Default is
#' "micro"
#' @return Calculated metric return as a base-R vector as opposed to tensor
#' @export
luz_metric_precision <- luz::luz_metric(

  abbrev = "Precision",

  initialize = function(mode = "multiclass", average="micro", smooth=1){
    self$tps <- 0.0
    self$fns <- 0.0
    self$fps <- 0.0
    self$smooth <- smooth
    self$mode <- mode
    self$average <- average
  },

  update = function(preds, target){
    if(self$mode == "multiclass"){
      preds <- torch::torch_argmax(preds, dim = 2)
      target1 <- torch::tensor(target+1, dtype=torch::torch_long())

      target_one_hot <- torch::nnf_one_hot(target1$flatten(), num_classes = preds$shape[2])
      target_one_hot <- target_one_hot$permute(c(2,1))$reshape(c(1,2, preds$shape[3],preds$shape[4]))

      dims <- c(3, 4)
      if(self$average == "micro"){
        dims <- c(2, dims)
      }

      self$tps <- self$tps + torch::torch_sum(preds * target_one_hot, dims)$item()
      self$fps <- self$fps + torch::torch_sum(preds * (-target_one_hot + 1.0), dims)$item()
      self$fns <- self$fns + torch::torch_sum((-preds + 1.0) * target_one_hot, dims)$item()

    }else{
      preds <- torch::nnf_sigmoid(preds)
      preds <- torch::torch_round(preds)
      preds <- preds$flatten()
      target <- target$flatten()

      self$tps <- self$tps + sum(preds * target)$item()
      self$fps <- self$fps + sum((1.0 - target) * preds)$item()
      self$fns <- self$fns + sum(target * (1.0 - preds))$item()
    }
  },

  compute = function(){
    mean((self$tps + self$smooth)/(self$tps + self$fps + self$smooth))
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
#' [Batch, Class Logits, Height, Width] for a multiclass classification. For a
#' binary classification, you can provide logits for the positive class as
#' [Batch, Positive Class Logit, Height, Width] or [Batch, Height, Width].
#' @param target Tensor of target class labels with shape
#' [Batch, Class Indices, Height, Width] for a multiclass classification. For a
#' binary classification, you can provide targets as
#' [Batch, Positive Class Index, Height, Width] or [Batch, Height, Width]. For
#' binary classification, the class index must be 1 for the positive class and 0
#' for the background case.
#' @param mode Either "binary" or "multiclass". If "binary", only the logit for
#' positive class prediction should be provided. If both the positive and negative
#' or background class probability is provided for a binary classification, use
#' the "multiclass" mode.
#' @param average Either "micro" or "macro". Whether to use micro- or macro-averaging
#' for multiclass metric calculation. Ignored when mode is "binary". Default is
#' "micro"
#' @return Calculated metric return as a base-R vector as opposed to tensor
#' @export
luz_metric_f1score <- luz::luz_metric(

  abbrev = "F1Score",

  initialize = function(mode = "multiclass", average="micro", smooth=1){
    self$tps <- 0.0
    self$fns <- 0.0
    self$fps <- 0.0
    self$smooth <- smooth
    self$mode <- mode
    self$average <- average
  },

  update = function(preds, target){
    if(self$mode == "multiclass"){
      preds <- torch::torch_argmax(preds, dim = 2)
      target1 <- torch::tensor(target+1, dtype=torch::torch_long())

      target_one_hot <- torch::nnf_one_hot(target1$flatten(), num_classes = preds$shape[2])
      target_one_hot <- target_one_hot$permute(c(2,1))$reshape(c(1,2, preds$shape[3],preds$shape[4]))

      dims <- c(3, 4)
      if(self$average == "micro"){
        dims <- c(2, dims)
      }

      self$tps <- self$tps + torch::torch_sum(preds * target_one_hot, dims)$item()
      self$fps <- self$fps + torch::torch_sum(preds * (-target_one_hot + 1.0), dims)$item()
      self$fns <- self$fns + torch::torch_sum((-preds + 1.0) * target_one_hot, dims)$item()

    }else{
      preds <- torch::nnf_sigmoid(preds)
      preds <- torch::torch_round(preds)
      preds <- preds$flatten()
      target <- target$flatten()

      self$tps <- self$tps + sum(preds * target)$item()
      self$fps <- self$fps + sum((1.0 - target) * preds)$item()
      self$fns <- self$fns + sum(target * (1.0 - preds))$item()
    }
  },

  compute = function(){
    mean(((2.0*self$tps) + self$smooth)/((2.0*self$tps) + self$fns + self$fps + self$smooth))
  }
)
