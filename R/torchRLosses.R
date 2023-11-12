#' dice_loss_family
#'
#' Define a loss metric for binary or multiclass classification based on Dice (function-based).
#'
#' Allows for defining a Dice-based loss metric including Dice, Focal Dice,
#' Tversky, Focal Tversky, or a combo loss that combines Dice or Tversky loss
#' with binary cross entropy or cross entropy loss. This is a functional implementation
#' that is called internally in the defineDiceFamilyLoss() class-based implementation.
#'
#' @param smooth Smooth factor to aid in stability and to prevent divide-by-zero errors.
#' Default is 1.
#' @param mode Either "multiclass" or "binary". Default is "multiclass". If "multiclass",
#' the prediction should be provided as (Batch, Channel, Height, Width), where the channel
#' dimension provides the predicted logit for each class, and the target should be
#' (Batch, Channel, Height, Width), where the channel dimension provides the index
#' for the correct class. Script assumes class indices start at 0 as opposed to 1.
#' If "binary", the prediction should be provided as
#' (Batch, Channel, Height, Width), where the channel dimension provides the
#' predicted logit for the positive class, and the target should be
#' (Batch, Channel, Height, Width), where the channel dimension provides the index
#' for the correct class (0 = negative, 1 = Positive). If the target does not included
#' the channel dimension (i.e. (Batch, Height, Width)), the chnDim argument should be set
#' to FALSE, which will force the script to add the channel dimension. It is best to
#' provide targets in a torch_long data type. If not, the script will convert the targets
#' to long. Predictions should be provided as logits, and a softmax or sigmoid activation
#' should not be applied. The data type for the targets should be torch_float32.
#' @param alpha Alpha parameter for false positives in Tversky calculation. This
#' is ignored if Dice is calculated. The default is 0.5.
#' @param beta Beta parameter for false negatives in Tversky calculation. This
#' is ignored if Dice is calculated. The default is 0.5
#' @param gamma Gamma parameter if Focal Tversky or Focal Dice is calculated.
#' Ignored if focal loss is not used. Default is 1.
#' @param average Either "micro" or "macro". Class averaging method applied for
#' multiclass classification. If "micro", classes are weighted relative to their
#' abundance in the target data. If "macro", classes are equally weighted in the
#' calculation. Default is "micro".
#' @param tversky TRUE or FALSE. Whether to calculate Tversky as opposed to Dice
#' loss. If TRUE, Tversky is calculated. If FALSE, Dice is calculated. Default is FALSE.
#' @param focal TRUE or FALSE. Whether to calculate a Focal Dice or Focal Tversky loss.
#' If FALSE, the gamma parameter is ignored. Default is FALSE.
#' @param combo TRUE or FALSE. Whether to calculate a combo loss using Dice/Tversky +
#' binary entropy/cross entropy. If TRUE, a combo loss is calculated. If FALSE, a
#' combo loss is not calculated. Default is FALSE.
#' @param useWghts TRUE or FALSE. Default is FALSE. If TRUE, class weights will be applied
#' in the calculation of cross entropy loss and macro-average Dice or Tversky loss. This setting
#' does not impact micro-averaged Dice or Tversky loss. If TRUE, the wght argument must be specified.
#' @param wght TRUE or FALSE. Default is FALSE. Must be defined if useWhgts is TRUE. A vector of class
#' weights must be provided that has the same length as the number of classes. The vector is converted to
#' a torch tensor within the script.
#' @param ceWght If combo is TRUE, defines relative weighting of binary cross entropy/
#' cross entropy in the loss as (Dice/Tversky) + ceWght*(binary cross entropy/
#' cross entropy). Ignored if combo is FALSE. Default is 1, or equal weighting in
#' the calculation between the two losses.
#' @param chnDim TRUE or FALSE. Default is TRUE. If TRUE, assumes the target tensor includes the
#' channel dimension: (Batch, Channel, Height, Width). If FALSE, assumes the target tensor does not include the
#' channel dimension: (Channel, Height, Width). The script is written such that it expects the channel dimension.
#' So, if FALSE, the script will add the channel dimension as needed.
#' @param mskLong TRUE or FALSE. Default is TRUE. Data type of target or mask. If the provided target
#' has a data type other than torch_long, this parameter should be set to FALSE. This will cause the
#' script to convert the target to the tensor_long data type as required for the calculations.
#' @return Loss metric for us in training process.
#' @export
dice_loss_family <- function(pred,
                             target,
                             nCls=1,
                             chnDim=TRUE,
                             zeroStart=TRUE,
                             smooth = 1,
                             mode = "multiclass",
                             alpha = 0.5,
                             beta = 0.5,
                             gamma = 1,
                             average = "micro",
                             tversky = FALSE,
                             focal = FALSE,
                             combo = FALSE,
                             useWghts=FALSE,
                             wghts,
                             ceWght,
                             mskLong = TRUE){
  if(chnDim == FALSE){
    target <- target$unsqueeze(dim=2)
  }

  if(mskLong == FALSE){
    target <- torch::torch_tensor(target, dtype=torch::torch_long())
  }

  if(mode == "multiclass"){
    input_soft <- pred |>
      torch::nnf_log_softmax(dim = 2) |>
      torch::torch_exp()

    target1 <- torch::torch_tensor(target, dtype=torch::torch_long())

    if(zeroStart == TRUE){
      target1 <- torch::torch_tensor(target+1, dtype=torch::torch_long())
    }

    target_one_hot <- torch::nnf_one_hot(target1, num_classes = nCls)
    target_one_hot <- target_one_hot$squeeze()
    target_one_hot <- target_one_hot$permute(c(1,4,2,3))

    dims <- c(3, 4)
    if(average == "micro"){
      dims <- c(2,3,4)
    }

    tps <- torch::torch_sum(input_soft * target_one_hot, dims)
    fps <- torch::torch_sum(input_soft * (-target_one_hot + 1.0), dims)
    fns <- torch::torch_sum((-input_soft + 1.0) * target_one_hot, dims)

    if(tversky == TRUE){
      numerator <- tps
      denominator <- tps + (beta * fps) + (alpha * fns)
    }else{
      numerator <- 2.0 * tps
      denominator <- (2.0 * tps) + fps + fns
    }

    metric <- 1.0 - ((numerator + smooth) / (denominator + smooth))

    if(focal == TRUE){
      metric <- metric ^ gamma
    }

    if(useWghts==TRUE & average=="macro"){
      wghtsT <- torch::torch_tensor(wghts, dtype=torch::torch_float32())
      metric <- metric*wghtsT
    }

    metric <- torch::torch_mean(metric)

    if(combo == TRUE & useWghts==TRUE){
      wghtsT <- torch::torch_tensor(wghts, dtype=torch::torch_float32())
      ceLoss <- torch::nnf_cross_entropy(pred, target1, weight=wghtsT)
      metric <- ceWght*ceLoss + metric
    }

    if(combo==TRUE & useWghts==FALSE){
      ceLoss <- torch::nnf_cross_entropy(pred, target1)
      metric <- ceWght*ceLoss + metric
    }

  }else{

    pred <- pred |> torch::nnf_logsigmoid() |> torch::torch_exp()
    pred <- pred$flatten()
    target <- target$flatten()

    tps <- sum(pred * target)
    fps <- sum((1.0 - target) * pred)
    fns <- sum(target * (1.0 - pred))

    if(tversky == TRUE){
      numerator <- tps
      denominator <- tps + (beta * fps) + (alpha * fns)
    }else{
      numerator <- 2.0 * tps
      denominator <- (2.0 * tps) + fps + fns
    }

    metric <- (1.0) - (numerator + smooth) / (denominator + smooth)

    if(focal == TRUE) {
      metric <- metric ^ gamma
    }

    if(combo == TRUE) {
      targetF <- torch::torch_tensor(target, dtype=torch::torch_float32())
      ceLoss <- torch::nnf_binary_cross_entropy_with_logits(pred, targetF)
      metric <- ceWght*ceLoss + metric
    }
  }

  return(metric)
}

#' defineDiceLossFamily
#'
#' Define a loss metric for binary or multiclass classification based on Dice (class-based).
#'
#' Allows for defining a Dice-based loss metric including Dice, Focal Dice,
#' Tversky, Focal Tversky, or a combo loss that combines Dice or Tversky loss
#' with binary cross entropy or cross entropy loss. This is implemented as a subclass
#' of torch::nn_module() that uses the geodl::dice_loss_family() function internally. Must be
#' instantiated before use.
#'
#' @param smooth Smooth factor to aid in stability and to prevent divide-by-zero errors.
#' Default is 1.
#' @param mode Either "multiclass" or "binary". Default is "multiclass". If "multiclass",
#' the prediction should be provided as [Batch, Channel, Height, Width], where the channel
#' dimension provides the predicted logit for each class, and the target should be
#' [Batch, Channel, Height, Width], where the channel dimension provides the index
#' for the correct class. Script assumes class indices start at 0 as opposed to 1.
#' If "binary", the prediction should be provided as
#' [Batch, Channel, Height, Width], where the channel dimension provides the
#' predicted logit for the positive class, and the target should be
#' [Batch, Channel, Height, Width], where the channel dimension provides the index
#' for the correct class (0 = negative, 1 = Positive). If the target does not included
#' the channel dimension (i.e. [Batch, Height, Width]), the chnDim argument should be set
#' to FALSE, which will force the script to add the channel dimension. It is best to
#' provide targets in a torch_long data type. If not, the script will convert the targets
#' to long. Predictions should be provided as logits, and a softmax or sigmoid activation
#' should not be applied. The data type for the targets should be torch_float32.
#' @param alpha Alpha parameter for false positives in Tversky calculation. This
#' is ignored if Dice is calculated. The default is 0.5.
#' @param beta Beta parameter for false negatives in Tversky calculation. This
#' is ignored if Dice is calculated. The default is 0.5
#' @param gamma Gamma parameter if Focal Tversky or Focal Dice is calculated.
#' Ignored if focal loss is not used. Default is 1.
#' @param average Either "micro" or "macro". Class averaging method applied for
#' multiclass classification. If "micro", classes are weighted relative to their
#' abundance in the target data. If "macro", classes are equally weighted in the
#' calculation. Default is "micro".
#' @param tversky TRUE or FALSE. Whether to calculate Tversky as opposed to Dice
#' loss. If TRUE, Tversky is calculated. If FALSE, Dice is calculated. Default is FALSE.
#' @param focal TRUE or FALSE. Whether to calculate a Focal Dice or Focal Tversky loss.
#' If FALSE, the gamma parameter is ignored. Default is FALSE.
#' @param combo TRUE or FALSE. Whether to calculate a combo loss using Dice/Tversky +
#' binary entropy/cross entropy. If TRUE, a combo loss is calculated. If FALSE, a
#' combo loss is not calculated. Default is FALSE.
#' @param useWghts TRUE or FALSE. Default is FALSE. If TRUE, class weights will be applied
#' in the calculation of cross entropy loss and macro-average Dice or Tversky loss. This setting
#' does not impact micro-averaged Dice or Tversky loss. If TRUE, the wght argument must be specified.
#' @param wght TRUE or FALSE. Default is FALSE. Must be defined if useWhgts is TRUE. A vector of class
#' weights must be provided that has the same length as the number of classes. The vector is converted to
#' a torch tensor within the script.
#' @param ceWght If combo is TRUE, defines relative weighting of binary cross entropy/
#' cross entropy in the loss as (Dice/Tversky) + ceWght*(binary cross entropy/
#' cross entropy). Ignored if combo is FALSE. Default is 1, or equal weighting in
#' the calculation between the two losses.
#' @param chnDim TRUE or FALSE. Default is TRUE. If TRUE, assumes the target tensor includes the
#' channel dimension: [Batch, Channel, Height, Width]. If FALSE, assumes the target tensor does not include the
#' channel dimension: [Channel, Height, Width]. The script is written such that it expects the channel dimension.
#' So, if FALSE, the script will add the channel dimension as needed.
#' @param mskLong TRUE or FALSE. Default is TRUE. Data type of target or mask. If the provided target
#' has a data type other than torch_long, this parameter should be set to FALSE. This will cause the
#' script to convert the target to the tensor_long data type as required for the calculations.
#' @return Loss metric for us in training process.
#' @export
defineDiceLossFamily <- torch::nn_module(
  initialize = function(nCls=1,
                        chnDim=TRUE,
                        zeroStart=TRUE,
                        smooth = 1,
                        mode = "multiclass",
                        alpha = 0.5,
                        beta = 0.5,
                        gamma = 1,
                        average = "micro",
                        tversky = FALSE,
                        focal = FALSE,
                        combo = FALSE,
                        useWghts=FALSE,
                        wghts = c(1,1),
                        ceWght=1,
                        mskLong = TRUE){

    self$nCls <- nCls
    self$chnDim <- chnDim
    self$zeroStart <- zeroStart
    self$smooth <- smooth
    self$mode <- mode
    self$alpha <- alpha
    self$beta <- beta
    self$gamma <- gamma
    self$average <- average
    self$tversky <- tversky
    self$focal <- focal
    self$combo <- combo
    self$useWghts <- useWghts
    self$wghts <- wghts
    self$ceWght <- ceWght
    self$mskLong <- mskLong

  },

  forward = function(pred, target) {
    return(dice_loss_family(pred,
                            target,
                            self$nCls,
                            self$chnDim,
                            self$zeroStart,
                            self$smooth,
                            self$mode,
                            self$alpha,
                            self$beta,
                            self$gamma,
                            self$average,
                            self$tversky,
                            self$focal,
                            self$combo,
                            self$useWghts,
                            self$wghts,
                            self$ceWght,
                            self$mskLong))
  }
)


#Assess using raster grids
#efG <- terra::rast("C:/Users/vidcg/Dropbox/code_dev/geodl/data/topoResult/topoRef.tif")
#predGP <- terra::rast("C:/Users/vidcg/Dropbox/code_dev/geodl/data/topoResult/topoPredP.tif")
#predGB <- terra::rast("C:/Users/vidcg/Dropbox/code_dev/geodl/data/topoResult/topoPredB.tif")
#refG2 <- terra::crop(terra::project(refG, predGP), predGP)

#predG <- c(predGB, predGP)

#predG <- terra::as.array(predGP)
#refG2 <- terra::as.array(refG2)

#target <- torch::torch_tensor(refG2, dtype=torch::torch_long())
#pred <- torch::torch_tensor(predG, dtype=torch::torch_float32())
#target <- target$permute(c(3,1,2))
#pred <- pred$permute(c(3,1,2))

#target <- target$unsqueeze(1)
#pred <- pred$unsqueeze(1)

#target <- torch::torch_tensor(target, dtype=torch::torch_long()
