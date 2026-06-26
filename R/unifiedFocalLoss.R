#' defineUnifiedFocalLoss
#'
#' Define a loss for semantic segmentation using a modified unified focal loss framework as a subclass of torch::nn_module()
#'
#' Implementation of modified version of the unified focal loss after:
#'
#' Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal loss:
#' Generalising Dice and cross entropy-based losses to handle class imbalanced
#' medical image segmentation. Computerized Medical Imaging and Graphics, 95, p.102026.
#'
#' Modifications include (1) allowing users to define class weights for both the distribution-
#' based and region-based losses, (2) using class weights as opposed to the symmetric and
#' asymmetric methods implemented by the authors, and (3) including an option to apply
#' a logcosh transform to the region-based loss.
#'
#' This loss has three key hyperparameters that control its implementation. Lambda controls
#' the relative weight of the distribution- and region-based losses. Default is 0.5,
#' or equal weighting between the losses is applied. If lambda = 1, only the distribution-
#' based loss is considered. If lambda = 0, only the region-based loss is considered. Values between 0.5
#' and 1 put more weight on the distribution-based loss while values between 0 and 0.5 put more
#' weight on the region-based loss.
#'
#' Gamma controls the application of focal loss and the application of
#' increased weight to difficult-to-predict pixels (for distribution-based losses) or difficult-to-predict
#' classes (region-based losses). Lower gamma values put increased weight on difficult samples or classes.
#' Using a value of 1 equates to not using a focal adjustment.
#'
#' The delta term controls the relative weight of
#' false positive and false negative errors for each class. The default is 0.6 for each class, which results in
#' placing a higher weight on false negative as opposed to false positive errors relative to that class.
#'
#' By adjusting the lambda, gamma, delta, and class weight terms, the user can implement a variety of
#' different loss metrics including cross entropy loss, weighted cross entropy loss, focal cross entropy
#' loss, focal weighted cross entropy loss, Dice loss, focal Dice loss, Tversky loss, and focal Tversky
#' loss.
#'
#' @param nCls Number of classes being differentiated.
#' @param cropFactorMsk Number of rows and columns of cells to not include for mask in assessment to
#' minimize edge effects. Default is 0 or no cropping.
#' @param cropFactorPred Number of rows and columns of cells to not include for prediction in assessment to
#' minimize edge effects. Default is 0 or no cropping.
#' @param lambda Term used to control the relative weighting of the distribution- and region-based
#' losses. Default is 0.5, or equal weighting between the losses. If lambda = 1, only the distribution-
#' based loss is considered. If lambda = 0, only the region-based loss is considered. Values between 0.5
#' and 1 put more weight on the distribution-based loss while values between 0 and 0.5 put more
#' weight on the region-based loss.
#' @param gamma Parameter that controls weighting applied to difficult-to-predict pixels (for
#' distribution-based losses) or difficult-to-predict classes (for region-based losses). Smaller values increase the
#' weight applied to difficult samples or classes. Default is 1, or no focal weighting is applied. Value must be
#' less than or equal to 1 and larger than 0.
#' @param delta Parameter that controls the relative weightings of false positive and false negative errors for
#' each class. Different weightings can be provided for each class. The default is 0.6, which results in prioritizing
#' false negative errors relative to false positive errors.
#' @param smooth Smoothing factor to avoid divide-by-zero errors and provide numeric stability. Default is 1e-8.
#' Recommend using the default.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param clsWghtsDist Vector of class weights for use in calculating a weighted version of the CE loss.
#' Default is for all classes to be equally weighted.
#' @param clsWghtsReg Vector of class weights for use in calculating a weighted version of the
#' region-based loss. Default is for all classes to be equally weighted.
#' @param useLogCosH TRUE or FALSE. Whether or not to apply a logCosH transformation to the region-based
#' loss. Default is FALSE.
#' @param device Define device being used for computation. Define using torch_device().
#' @return Loss metric for use in training process.
#' @export
defineUnifiedFocalLoss <- torch::nn_module(
  initialize = function(nCls=3,
                        cropFactorMsk=0,
                        cropFactorPred=0,
                        lambda=.5,
                        gamma=.5,
                        delta=0.6,
                        smooth = 1e-8,
                        zeroStart=TRUE,
                        clsWghtsDist=1,
                        clsWghtsReg=1,
                        useLogCosH=FALSE,
                        device="cuda"){

    self$nCls          <- nCls
    self$cropFactorMsk <- cropFactorMsk
    self$cropFactorPred <- cropFactorPred
    self$lambda        <- lambda
    self$gamma         <- gamma
    self$delta         <- delta
    self$smooth        <- smooth
    self$zeroStart     <- zeroStart
    self$useLogCosH    <- useLogCosH
    self$device        <- device

    # Expand scalar delta and class weights to per-class vectors once
    delta2        <- if(length(delta) == 1)       rep(delta,       nCls) else delta
    clsWghtsDist2 <- if(length(clsWghtsDist) == 1) rep(clsWghtsDist, nCls) else clsWghtsDist
    clsWghtsReg2  <- if(length(clsWghtsReg) == 1)  rep(clsWghtsReg,  nCls) else clsWghtsReg

    # Pre-build all constant tensors so they are not recreated on every forward call
    self$smoothT      <- torch::torch_tensor(smooth,
                                             dtype=torch::torch_float32(),
                                             device=device)
    self$lambdaT      <- torch::torch_tensor(lambda,
                                             dtype=torch::torch_float32(),
                                             device=device)
    self$gammaT       <- torch::torch_tensor(gamma,
                                             dtype=torch::torch_float32(),
                                             device=device)
    self$gammaRepT    <- torch::torch_tensor(rep(gamma, nCls),
                                             dtype=torch::torch_float32(),
                                             device=device)
    self$deltaT       <- torch::torch_tensor(delta2,
                                             dtype=torch::torch_float32(),
                                             device=device)
    self$wghtT        <- torch::torch_tensor(clsWghtsDist2,
                                             dtype=torch::torch_float32(),
                                             device=device)
    self$clsWghtsRegT <- torch::torch_tensor(clsWghtsReg2,
                                             dtype=torch::torch_float32(),
                                             device=device)

    # (1, C, 1, 1) weight tensor for broadcasting against (N, C, H, W) one-hot maps
    self$clsWghtsDistMapT <- torch::torch_tensor(clsWghtsDist2,
                                                 dtype=torch::torch_float32(),
                                                 device=device)$view(c(1L, nCls, 1L, 1L))
  },

  forward = function(pred, target){

    pred   <- cropTensor(pred,   crpFactor=self$cropFactorPred)
    target <- cropTensor(target, crpFactor=self$cropFactorMsk)

    # Convert target to long and shift indices if classes start at 0
    target1 <- target$to(dtype=torch::torch_long(), device=self$device)
    if(self$zeroStart) target1 <- target1 + 1L

    # Softmax probabilities for region-based loss
    pred_soft <- torch::nnf_softmax(pred, dim=2L)

    # One-hot encode: (N,1,H,W) -> nnf_one_hot -> (N,1,H,W,C) -> squeeze dim2 -> (N,H,W,C) -> permute -> (N,C,H,W)
    target_one_hot <- torch::nnf_one_hot(target1, num_classes=self$nCls)
    target_one_hot <- target_one_hot$squeeze(2L)
    target_one_hot <- target_one_hot$permute(c(1L, 4L, 2L, 3L))

    # Distribution-based loss ------------------------------------------------
    if(self$lambda > 0){
      # Remove channel dim for CE; squeeze(2) targets dim of size 1 only
      targetCE <- target1$squeeze(2L)

      # Unweighted per-pixel CE for the focal probability estimate
      ceL <- torch::nnf_cross_entropy(pred, targetCE, reduction="none")

      # True class probability: p_t = exp(-CE_unweighted)
      pt <- torch::torch_exp(-ceL)

      # Focal modifier * class-weighted CE
      # Weight applied per-pixel via one-hot broadcast: avoids passing weight to CE
      # (which would distort pt) while still weighting the loss contribution
      wghtsPerPixel <- torch::torch_sum(target_one_hot * self$clsWghtsDistMapT, dim=2L)
      mFL <- ((1.0 - pt)**(1.0 - self$gammaT)) * ceL * wghtsPerPixel

      # Normalise by total weight mass across the batch
      wghtSumT    <- torch::torch_sum(target_one_hot * self$clsWghtsDistMapT)
      distMetric  <- torch::torch_sum(mFL) / wghtSumT
    }

    # Region-based loss -------------------------------------------------------
    if(self$lambda < 1){
      tps <- torch::torch_sum(pred_soft * target_one_hot,          dim=c(1L, 3L, 4L))
      fps <- torch::torch_sum(pred_soft * (1.0 - target_one_hot),  dim=c(1L, 3L, 4L))
      fns <- torch::torch_sum((1.0 - pred_soft) * target_one_hot,  dim=c(1L, 3L, 4L))

      mTI <- (tps + self$smoothT) /
             (tps + ((1.0 - self$deltaT) * fps) + (self$deltaT * fns) + self$smoothT)

      regMetric <- (1.0 - mTI)**self$gammaRepT
      regMetric <- regMetric * self$clsWghtsRegT
      regMetric <- torch::torch_sum(regMetric) / torch::torch_sum(self$clsWghtsRegT)

      if(self$useLogCosH){
        regMetric <- torch::torch_log(torch::torch_cosh(regMetric))
      }
    }

    if(self$lambda == 1){
      comboMetric <- distMetric
    }else if(self$lambda == 0){
      comboMetric <- regMetric
    }else{
      comboMetric <- (self$lambdaT * distMetric) + ((1.0 - self$lambdaT) * regMetric)
    }

    return(comboMetric)
  }
)


#' defineUnifiedFocalLossDS
#'
#' Define a loss for geospatial semantic segmentation using a modified unified focal loss framework as a subclass of torch::nn_module() when using deep supervision.
#'
#' Implementation of modified version of the unified focal loss after:
#'
#' Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal loss:
#' Generalising Dice and cross entropy-based losses to handle class imbalanced
#' medical image segmentation. Computerized Medical Imaging and Graphics, 95, p.102026.
#'
#' Modifications include (1) allowing users to define class weights for both the distribution-
#' based and region-based losses, (2) using class weights as opposed to the symmetric and
#' asymmetric methods implemented by the authors, and (3) including an option to apply
#' a logcosh transform to the region-based loss.
#'
#' This loss has three key hyperparameters that control its implementation. Lambda controls
#' the relative weight of the distribution- and region-based losses. Default is 0.5,
#' or equal weighting between the losses is applied. If lambda = 1, only the distribution-
#' based loss is considered. If lambda = 0, only the region-based loss is considered. Values between 0.5
#' and 1 put more weight on the distribution-based loss while values between 0 and 0.5 put more
#' weight on the region-based loss.
#'
#' Gamma controls the application of focal loss and the application of
#' increased weight to difficult-to-predict pixels (for distribution-based losses) or difficult-to-predict
#' classes (region-based losses). Lower gamma values put increased weight on difficult samples or classes.
#' Using a value of 1 equates to not using a focal adjustment.
#'
#' The delta term controls the relative weight of
#' false positive and false negative errors for each class. The default is 0.6 for each class, which results in
#' placing a higher weight on false negative as opposed to false positive errors relative to that class.
#'
#' By adjusting the lambda, gamma, delta, and class weight terms, the user can implement a variety of different loss metrics
#' including cross entropy loss, weighted cross entropy loss, focal cross entropy loss, focal weighted cross entropy loss,
#' Dice loss, focal Dice loss, Tversky loss, and focal Tversky loss.
#'
#' @param nCls Number of classes being differentiated.
#' @param cropFactorMsk Number of rows and columns of cells to not include for mask in assessment to
#' minimize edge effects. Default is 0 or no cropping.
#' @param cropFactorPred Number of rows and columns of cells to not include for prediction in assessment to
#' minimize edge effects. Default is 0 or no cropping.
#' @param dsWghts Vector of 4 weights applied to the losses at each spatial resolution. Used only
#' when \code{weight_env} is \code{NULL}. The default is c(.6, .2, .1, .1) where larger weights are
#' placed on the results at a higher spatial resolution.
#' @param weight_env Optional environment created by \code{make_ds_weights()}. When supplied, the
#' loss reads \code{weight_env$values} on every forward pass so that
#' \code{callback_ds_weight_decay()} can decay the auxiliary weights during training without
#' rebuilding the loss. Takes precedence over \code{dsWghts}. Default is \code{NULL}.
#' @param lambda Term used to control the relative weighting of the distribution- and region-based
#' losses. Default is 0.5, or equal weighting between the losses. If lambda = 1, only the distribution-
#' based loss is considered. If lambda = 0, only the region-based loss is considered. Values between 0.5
#' and 1 put more weight on the distribution-based loss while values between 0 and 0.5 put more
#' weight on the region-based loss.
#' @param gamma Parameter that controls weighting applied to difficult-to-predict pixels (for
#' distribution-based losses) or difficult-to-predict classes (for region-based losses). Smaller values increase the
#' weight applied to difficult samples or classes. Default is 1, or no focal weighting is applied. Value must be
#' less than or equal to 1 and larger than 0.
#' @param delta Parameter that controls the relative weightings of false positive and false negative errors for
#' each class. Different weightings can be provided for each class. The default is 0.6, which results in prioritizing
#' false negative errors relative to false positive errors.
#' @param smooth Smoothing factor to avoid divide-by-zero errors and provide numeric stability. Default is 1e-8.
#' Recommend using the default.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param clsWghtsDist Vector of class weights for use in calculating a weighted version of the CE loss.
#' Default is for all classes to be equally weighted.
#' @param clsWghtsReg Vector of class weights for use in calculating a weighted version of the
#' region-based loss. Default is for all classes to be equally weighted.
#' @param useLogCosH TRUE or FALSE. Whether or not to apply a logCosH transformation to the region-based
#' loss. Default is FALSE.
#' @param device Define device being used for computation. Define using torch_device().
#' @return Loss metric for use in training process.
#' @export
defineUnifiedFocalLossDS <- torch::nn_module(
  initialize = function(nCls=3,
                        cropFactorMsk=0,
                        cropFactorPred=0,
                        dsWghts=c(.6, .2, .1, .1),
                        weight_env=NULL,
                        lambda=.5,
                        gamma=.5,
                        delta=0.6,
                        smooth=1e-8,
                        zeroStart=TRUE,
                        clsWghtsDist=1,
                        clsWghtsReg=1,
                        useLogCosH=FALSE,
                        device="cuda"){

    self$device     <- device
    self$weight_env <- weight_env

    # Static weight tensors — used only when weight_env is NULL
    self$wght1   <- torch::torch_tensor(dsWghts[1], dtype=torch::torch_float32(), device=device)
    self$wght2   <- torch::torch_tensor(dsWghts[2], dtype=torch::torch_float32(), device=device)
    self$wght4   <- torch::torch_tensor(dsWghts[3], dtype=torch::torch_float32(), device=device)
    self$wght8   <- torch::torch_tensor(dsWghts[4], dtype=torch::torch_float32(), device=device)
    self$wghtSum <- torch::torch_tensor(sum(dsWghts), dtype=torch::torch_float32(), device=device)

    # Single loss instance shared across all four decoder outputs
    self$loss <- defineUnifiedFocalLoss(nCls=nCls,
                                        cropFactorMsk=cropFactorMsk,
                                        cropFactorPred=cropFactorPred,
                                        lambda=lambda,
                                        gamma=gamma,
                                        delta=delta,
                                        smooth=smooth,
                                        zeroStart=zeroStart,
                                        clsWghtsDist=clsWghtsDist,
                                        clsWghtsReg=clsWghtsReg,
                                        useLogCosH=useLogCosH,
                                        device=device)
  },

  forward = function(pred, target){
    if (!is.null(self$weight_env)) {
      w    <- self$weight_env$values
      wSum <- sum(w)
      lossOut <- w[1] * self$loss(pred[[1]], target)
      if (w[2] != 0) lossOut <- lossOut + w[2] * self$loss(pred[[2]], target)
      if (w[3] != 0) lossOut <- lossOut + w[3] * self$loss(pred[[3]], target)
      if (w[4] != 0) lossOut <- lossOut + w[4] * self$loss(pred[[4]], target)
      lossOut <- lossOut / wSum
    } else {
      l1 <- self$loss(pred[[1]], target)
      l2 <- self$loss(pred[[2]], target)
      l4 <- self$loss(pred[[3]], target)
      l8 <- self$loss(pred[[4]], target)
      lossOut <- ((self$wght1 * l1) + (self$wght2 * l2) +
                  (self$wght4 * l4) + (self$wght8 * l8)) / self$wghtSum
    }
    return(lossOut)
  }
)
