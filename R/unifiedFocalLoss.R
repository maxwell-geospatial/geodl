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
#' asymmetric methods implemented by the authors, (3) including an option to apply
#' a logcosh transform to the region-based loss, (4) optionally incorporating
#' distance-to-class-boundary weights computed by \code{makeBoundaryDistance}, (5) an
#' option to compute the distribution loss as a weighted average of per-class means rather
#' than as a single pixel-level weighted mean, and (6) optional label smoothing on the
#' distribution loss.
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
#' When \code{labelSmoothing > 0} the cross-entropy is computed against soft targets that
#' spread \code{labelSmoothing} probability mass uniformly across all classes, reducing the
#' model's incentive to become overconfident. The focal modifier is kept anchored to the
#' true-class softmax probability gathered directly from the network output, so the focal
#' weighting continues to reflect genuine model confidence rather than being distorted by
#' the smoothed targets. Label smoothing is applied only to the distribution loss;
#' the region-based (Tversky) loss is not affected.
#'
#' When \code{useEdgeDist = TRUE} a distance-to-class-boundary tensor must be passed as the third
#' argument to \code{forward(pred, target, dist)}. The tensor should have shape (N, 1, H, W) and
#' contain the per-pixel distances produced by \code{makeBoundaryDistance}, stacked into a batch.
#' Pixels at a class boundary have distance 0; distance increases away from boundaries.
#'
#' Three edge-weighting modes are available via \code{edgeMode}:
#' \itemize{
#'   \item \code{"threshold"}: pixels within \code{distThreshold} receive one weight; pixels beyond
#'         receive the other. When \code{prioritizeEdge = TRUE} the inside weight is \code{maxWght}
#'         and the outside weight is \code{minWght}; the relationship is reversed when
#'         \code{prioritizeEdge = FALSE}.
#'   \item \code{"linear"}: weight changes linearly from \code{maxWght} at distance 0 to
#'         \code{minWght} at \code{distThreshold} (when \code{prioritizeEdge = TRUE}), or from
#'         \code{minWght} to \code{maxWght} (when \code{prioritizeEdge = FALSE}). Beyond
#'         \code{distThreshold} the weight is clamped to the far-end value.
#'   \item \code{"gaussian"}: weight follows a Gaussian curve centred at distance 0. The curve
#'         uses sigma = distThreshold / 3 so that the weight is approximately \code{minWght}
#'         at \code{distThreshold} and falls to exactly \code{minWght} beyond it.
#'         Direction is controlled by \code{prioritizeEdge} as above.
#' }
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
#' @param useEdgeDist TRUE or FALSE. Whether to apply distance-to-class-boundary weights during loss
#' computation. When TRUE a \code{dist} tensor must be supplied to \code{forward}. Default is FALSE.
#' @param prioritizeEdge TRUE or FALSE. When \code{useEdgeDist = TRUE}, controls which pixels receive
#' the higher weight. If TRUE (default), pixels near class boundaries (small distance) receive
#' \code{maxWght} and pixels far from boundaries receive \code{minWght}. If FALSE, the weighting is
#' reversed so that non-edge pixels are emphasised.
#' @param edgeMode Character string specifying how the distance is converted to a per-pixel weight.
#' One of \code{"threshold"}, \code{"linear"}, or \code{"gaussian"}. Default is \code{"threshold"}.
#' See the Details section for a full description of each mode.
#' @param distThreshold Numeric. The distance value (in map units or pixel units, matching the
#' distance raster) that defines the boundary of the high-weight zone. Default is 10.
#' @param minWght Numeric. The minimum per-pixel weight applied to loss contributions.
#' Must be less than \code{maxWght}. Default is 0.5.
#' @param maxWght Numeric. The maximum per-pixel weight applied to loss contributions.
#' Must be greater than \code{minWght}. Default is 1.0.
#' @param applyDistWght Character string controlling which loss component the edge weights are
#' applied to. One of \code{"distribution"} (distribution-based/CE loss only),
#' \code{"region"} (region-based/Tversky loss only), or \code{"both"} (default).
#' @param labelSmoothing Numeric value in [0, 1). When greater than 0, this amount of
#' probability mass is spread uniformly across all classes when computing the cross-entropy,
#' preventing the model from becoming overconfident. The focal modifier is always derived
#' from the true-class softmax probability gathered directly from the network output, so
#' focal weighting is not distorted by the smoothed targets. Default is 0 (no smoothing).
#' @param classAvgDist TRUE or FALSE. When FALSE (default), the distribution loss is computed
#' as a pixel-level weighted mean of the focal cross-entropy, with \code{clsWghtsDist}
#' rescaling each pixel's contribution by its true class weight. When TRUE, the focal
#' cross-entropy is first averaged within each class and the resulting per-class means are
#' then combined as a weighted average using \code{clsWghtsDist}. The per-class approach
#' mirrors the macro-averaging philosophy already used by the region-based loss and can
#' give better gradient signal to under-represented classes under severe class imbalance.
#' Note that if a class has no pixels in a given batch its contribution is effectively
#' zero for that step, consistent with the behaviour of the region-based loss.
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
                        smooth=1e-8,
                        zeroStart=TRUE,
                        clsWghtsDist=1,
                        clsWghtsReg=1,
                        useLogCosH=FALSE,
                        useEdgeDist=FALSE,
                        prioritizeEdge=TRUE,
                        edgeMode="threshold",
                        distThreshold=10,
                        minWght=0.5,
                        maxWght=1.0,
                        applyDistWght="both",
                        labelSmoothing=0,
                        classAvgDist=FALSE,
                        device="cuda"){

    self$nCls            <- nCls
    self$cropFactorMsk   <- cropFactorMsk
    self$cropFactorPred  <- cropFactorPred
    self$lambda          <- lambda
    self$gamma           <- gamma
    self$delta           <- delta
    self$smooth          <- smooth
    self$zeroStart       <- zeroStart
    self$useLogCosH      <- useLogCosH
    self$device          <- device
    self$useEdgeDist     <- useEdgeDist
    self$prioritizeEdge  <- prioritizeEdge
    self$edgeMode        <- edgeMode
    self$applyDistWght   <- applyDistWght
    self$labelSmoothing  <- labelSmoothing
    self$classAvgDist    <- classAvgDist

    if (labelSmoothing < 0 || labelSmoothing >= 1)
      stop("labelSmoothing must be in [0, 1).")

    if (useEdgeDist) {
      if (!edgeMode %in% c("threshold", "linear", "gaussian"))
        stop("edgeMode must be 'threshold', 'linear', or 'gaussian'.")
      if (!applyDistWght %in% c("distribution", "region", "both"))
        stop("applyDistWght must be 'distribution', 'region', or 'both'.")
      if (minWght >= maxWght)
        stop("minWght must be less than maxWght.")
      if (distThreshold <= 0)
        stop("distThreshold must be greater than 0.")
    }

    # Expand scalar delta and class weights to per-class vectors once
    delta2        <- if(length(delta) == 1)         rep(delta,        nCls) else delta
    clsWghtsDist2 <- if(length(clsWghtsDist) == 1)  rep(clsWghtsDist, nCls) else clsWghtsDist
    clsWghtsReg2  <- if(length(clsWghtsReg) == 1)   rep(clsWghtsReg,  nCls) else clsWghtsReg

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

    # Edge distance tensors — only built when needed
    if (useEdgeDist) {
      self$distThresholdT <- torch::torch_tensor(as.numeric(distThreshold),
                                                 dtype=torch::torch_float32(),
                                                 device=device)
      self$minWghtT       <- torch::torch_tensor(as.numeric(minWght),
                                                 dtype=torch::torch_float32(),
                                                 device=device)
      self$maxWghtT       <- torch::torch_tensor(as.numeric(maxWght),
                                                 dtype=torch::torch_float32(),
                                                 device=device)
    }
  },

  forward = function(pred, target, dist=NULL){

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

    # Compute per-pixel edge weight map -------------------------------------------
    # edgeWghtMap: (N, 1, H, W) — broadcasts over the class dimension in region loss
    # edgeWghtFlat: (N, H, W)   — matches the shape of the per-pixel distribution loss
    useEdgeWgt <- self$useEdgeDist && !is.null(dist)
    if (useEdgeWgt) {
      dist <- cropTensor(dist, crpFactor=self$cropFactorMsk)
      dist <- dist$to(dtype=torch::torch_float32(), device=self$device)

      if (self$edgeMode == "threshold") {
        insideMask  <- (dist <= self$distThresholdT)$to(dtype=torch::torch_float32())
        outsideMask <- 1.0 - insideMask
        if (self$prioritizeEdge) {
          edgeWghtMap <- insideMask * self$maxWghtT + outsideMask * self$minWghtT
        } else {
          edgeWghtMap <- insideMask * self$minWghtT + outsideMask * self$maxWghtT
        }

      } else if (self$edgeMode == "linear") {
        # t = 0 at boundary, 1 at distThreshold (clamped beyond)
        t <- torch::torch_clamp(dist / self$distThresholdT, min=0.0, max=1.0)
        if (self$prioritizeEdge) {
          edgeWghtMap <- self$maxWghtT * (1.0 - t) + self$minWghtT * t
        } else {
          edgeWghtMap <- self$minWghtT * (1.0 - t) + self$maxWghtT * t
        }

      } else {
        # gaussian: sigma chosen so the curve reaches ~minWght at distThreshold
        sigma       <- self$distThresholdT / 3.0
        gaussFactor <- torch::torch_exp(-(dist * dist) / (2.0 * sigma * sigma))
        # Hard-clamp to zero beyond the threshold so weights don't creep upward
        beyondT     <- (dist > self$distThresholdT)$to(dtype=torch::torch_float32())
        gaussFactor <- gaussFactor * (1.0 - beyondT)
        # gaussFactor = 1 at dist=0, ~0 at distThreshold, exactly 0 beyond
        if (self$prioritizeEdge) {
          edgeWghtMap <- self$minWghtT + (self$maxWghtT - self$minWghtT) * gaussFactor
        } else {
          edgeWghtMap <- self$maxWghtT - (self$maxWghtT - self$minWghtT) * gaussFactor
        }
      }

      edgeWghtFlat <- edgeWghtMap$squeeze(2L)  # (N, H, W)
    }

    # Distribution-based loss ------------------------------------------------
    if(self$lambda > 0){
      # Remove channel dim for CE; squeeze(2) targets dim of size 1 only
      targetCE <- target1$squeeze(2L)

      # True-class softmax probability gathered directly from network output.
      # This is always based on the un-smoothed prediction so that the focal
      # modifier reflects genuine model confidence regardless of label smoothing.
      pt <- torch::torch_sum(pred_soft * target_one_hot, dim=2L)  # (N, H, W)

      # CE loss value — label smoothing applied here only, not to pt above.
      # Implemented manually for version compatibility: CE_smooth = (1-ε)*CE_hard + ε*mean_k(-log p_k)
      ceL <- torch::nnf_cross_entropy(pred, targetCE, reduction="none")
      if (self$labelSmoothing > 0) {
        eps     <- self$labelSmoothing
        log_p   <- torch::nnf_log_softmax(pred, dim=2L)   # (N, C, H, W)
        ce_unif <- -torch::torch_mean(log_p, dim=2L)       # (N, H, W)
        ceL     <- (1 - eps) * ceL + eps * ce_unif
      }

      # Shared per-pixel focal modifier — weighting strategy diverges below
      focalMod <- (1.0 - pt)**(1.0 - self$gammaT)  # (N, H, W)

      if (self$classAvgDist) {
        # Per-class mean focal CE, then weighted average across classes ----------
        # Class weights enter only through the final weighted average, not per-pixel,
        # giving each class a single equal-footing contribution before weighting.
        focalCE <- focalMod * ceL  # (N, H, W)

        if (useEdgeWgt && self$applyDistWght %in% c("distribution", "both")) {
          focalCE <- focalCE * edgeWghtFlat
          # Normalise each class mean by its total edge-weight mass
          normPerCls <- torch::torch_sum(
            edgeWghtFlat$unsqueeze(2L) * target_one_hot, dim=c(1L, 3L, 4L))  # (C,)
        } else {
          # Normalise each class mean by its pixel count
          normPerCls <- torch::torch_sum(target_one_hot, dim=c(1L, 3L, 4L))  # (C,)
        }

        # Sum focal CE over all pixels belonging to each class: (C,)
        perClsSum  <- torch::torch_sum(
          focalCE$unsqueeze(2L) * target_one_hot, dim=c(1L, 3L, 4L))
        normPerCls <- torch::torch_clamp(normPerCls, min=self$smoothT)
        perClsMean <- perClsSum / normPerCls  # (C,)

        # Weighted average of per-class means
        distMetric <- torch::torch_sum(perClsMean * self$wghtT) /
                      torch::torch_sum(self$wghtT)

      } else {
        # Traditional pixel-level weighted focal CE ----------------------------
        wghtsPerPixel <- torch::torch_sum(target_one_hot * self$clsWghtsDistMapT, dim=2L)
        mFL <- focalMod * ceL * wghtsPerPixel

        if (useEdgeWgt && self$applyDistWght %in% c("distribution", "both")) {
          mFL        <- mFL * edgeWghtFlat
          # Normalise by combined class-weight + edge-weight mass
          normFactor <- torch::torch_sum(wghtsPerPixel * edgeWghtFlat)
          distMetric <- torch::torch_sum(mFL) / normFactor
        } else {
          wghtSumT   <- torch::torch_sum(target_one_hot * self$clsWghtsDistMapT)
          distMetric <- torch::torch_sum(mFL) / wghtSumT
        }
      }
    }

    # Region-based loss -------------------------------------------------------
    if(self$lambda < 1){
      if (useEdgeWgt && self$applyDistWght %in% c("region", "both")) {
        # edgeWghtMap (N, 1, H, W) broadcasts over (N, C, H, W)
        tps <- torch::torch_sum(pred_soft * target_one_hot * edgeWghtMap,
                                dim=c(1L, 3L, 4L))
        fps <- torch::torch_sum(pred_soft * (1.0 - target_one_hot) * edgeWghtMap,
                                dim=c(1L, 3L, 4L))
        fns <- torch::torch_sum((1.0 - pred_soft) * target_one_hot * edgeWghtMap,
                                dim=c(1L, 3L, 4L))
      } else {
        tps <- torch::torch_sum(pred_soft * target_one_hot,          dim=c(1L, 3L, 4L))
        fps <- torch::torch_sum(pred_soft * (1.0 - target_one_hot),  dim=c(1L, 3L, 4L))
        fns <- torch::torch_sum((1.0 - pred_soft) * target_one_hot,  dim=c(1L, 3L, 4L))
      }

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
#' asymmetric methods implemented by the authors, (3) including an option to apply
#' a logcosh transform to the region-based loss, (4) optionally incorporating
#' distance-to-class-boundary weights computed by \code{makeBoundaryDistance}, (5) an
#' option to compute the distribution loss as a weighted average of per-class means rather
#' than as a single pixel-level weighted mean, and (6) optional label smoothing on the
#' distribution loss.
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
#' When \code{useEdgeDist = TRUE} a distance-to-class-boundary tensor must be passed as the third
#' argument to \code{forward(pred, target, dist)}. The same distance tensor is applied to all
#' decoder-output losses. See \code{defineUnifiedFocalLoss} for a full description of the
#' edge-distance weighting options.
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
#' @param useEdgeDist TRUE or FALSE. Whether to apply distance-to-class-boundary weights during loss
#' computation. When TRUE a \code{dist} tensor must be supplied to \code{forward}. Default is FALSE.
#' @param prioritizeEdge TRUE or FALSE. When \code{useEdgeDist = TRUE}, controls which pixels receive
#' the higher weight. If TRUE (default), pixels near class boundaries receive \code{maxWght} and
#' pixels far from boundaries receive \code{minWght}. If FALSE, the weighting is reversed.
#' @param edgeMode Character string specifying how the distance is converted to a per-pixel weight.
#' One of \code{"threshold"}, \code{"linear"}, or \code{"gaussian"}. Default is \code{"threshold"}.
#' @param distThreshold Numeric. The distance value that defines the boundary of the high-weight zone.
#' Default is 10.
#' @param minWght Numeric. The minimum per-pixel weight. Must be less than \code{maxWght}.
#' Default is 0.5.
#' @param maxWght Numeric. The maximum per-pixel weight. Must be greater than \code{minWght}.
#' Default is 1.0.
#' @param applyDistWght Character string controlling which loss component the edge weights are
#' applied to. One of \code{"distribution"}, \code{"region"}, or \code{"both"} (default).
#' @param labelSmoothing Numeric value in [0, 1). Amount of label smoothing applied to
#' the cross-entropy component of the distribution loss. The focal modifier is derived
#' from the true-class softmax probability and is not affected by smoothing.
#' Default is 0 (no smoothing). See \code{defineUnifiedFocalLoss} for full details.
#' @param classAvgDist TRUE or FALSE. When TRUE, the distribution loss is computed as a
#' weighted average of per-class focal cross-entropy means rather than a pixel-level
#' weighted mean. See \code{defineUnifiedFocalLoss} for a full description. Default is FALSE.
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
                        useEdgeDist=FALSE,
                        prioritizeEdge=TRUE,
                        edgeMode="threshold",
                        distThreshold=10,
                        minWght=0.5,
                        maxWght=1.0,
                        applyDistWght="both",
                        labelSmoothing=0,
                        classAvgDist=FALSE,
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
                                        useEdgeDist=useEdgeDist,
                                        prioritizeEdge=prioritizeEdge,
                                        edgeMode=edgeMode,
                                        distThreshold=distThreshold,
                                        minWght=minWght,
                                        maxWght=maxWght,
                                        applyDistWght=applyDistWght,
                                        labelSmoothing=labelSmoothing,
                                        classAvgDist=classAvgDist,
                                        device=device)
  },

  forward = function(pred, target, dist=NULL){
    if (!is.null(self$weight_env)) {
      w    <- self$weight_env$values
      wSum <- sum(w)
      lossOut <- w[1] * self$loss(pred[[1]], target, dist)
      if (w[2] != 0) lossOut <- lossOut + w[2] * self$loss(pred[[2]], target, dist)
      if (w[3] != 0) lossOut <- lossOut + w[3] * self$loss(pred[[3]], target, dist)
      if (w[4] != 0) lossOut <- lossOut + w[4] * self$loss(pred[[4]], target, dist)
      lossOut <- lossOut / wSum
    } else {
      l1 <- self$loss(pred[[1]], target, dist)
      l2 <- self$loss(pred[[2]], target, dist)
      l4 <- self$loss(pred[[3]], target, dist)
      l8 <- self$loss(pred[[4]], target, dist)
      lossOut <- ((self$wght1 * l1) + (self$wght2 * l2) +
                  (self$wght4 * l4) + (self$wght8 * l8)) / self$wghtSum
    }
    return(lossOut)
  }
)
