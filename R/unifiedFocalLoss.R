#' define_unified_focal_loss
#'
#' Define a loss for semantic segmentation using a modified unified focal loss framework (function).
#'
#' Implementation of modified version of the unified focal dice loss after:
#'
#' Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal loss:
#' Generalising dice and cross entropy-based losses to handle class imbalanced
#' medical image segmentation. Computerized Medical Imaging and Graphics, 95, p.102026.
#'
#' Modifications include (1) allowing users to define class weights for both the distribution-
#' based and region-based metrics, (2) using class weights as opposed to the symmetric and
#' asymmetric methods implemented by the authors, and (3) including an option to apply
#' a logcosh transform for the region-based loss.
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
#' classes (region-based losses). Larger gamma values put increased weight on difficult samples or classes.
#' Using a value of 0 equates to not using a focal adjustment.
#'
#' The delta term controls the relative weight of
#' false positive and false negative errors for each class. The default is 0.6 for each class, which results in
#' placing a higher weight on false positive as opposed to false negative errors relative to that class.
#'
#' By adjusting the lambda, gamma, delta, and class weight terms, the user can implement a variety of different loss metrics
#' including cross entropy loss, weighted cross entropy loss, focal cross entropy loss, focal weighted cross entropy loss,
#' Dice loss, focal Dice loss, Tversky loss, and focal Tversky loss.
#'
#' @param pred Tensor of predicted class logits. Should be of shape (mini-batch,
#' class, width, height) where the class dimension has a length equal to the number
#' of classes being differentiated. For a binary classification, output can be provided
#' as (mini-batch, class, width, height) or (mini-batch, width, height) if only the positive
#' case logit is returned.
#' @param target Tensor or predicted class indices from 0 to n-1 or 1 to n where n is the
#' number of classes. For a binary classification, only the positive case logit can be returned.
#' Shape can be (mini-batch, class, width, height) or (mini-batch, width, height).
#' @param nCls number of classes being differentiated. Should be 1 for a binary classification
#' where only the positive case logit is returned. Default is 3.
#' @param lambda Term used to control the relative weighting of the distribution- and region-based
#' losses. Default is 0.5, or equal weighting between the losses. If lambda = 1, only the distribution-
#' based loss is considered. If lambda = 0, only the region-based loss is considered. Values between 0.5
#' and 1 put more weight on the distribution-based loss while values between 0 and 0.5 put more
#' weight on the region-based loss.
#' @param gamma Parameter that controls increased weighting applied to difficult-to-predict pixels (for
#' distribution-based losses) or difficult-to-predict classes (region-based losses). Default is 0, or no focal
#' weighting is applied.
#' @param delta Parameter that controls the relative weightings of false positive and false negative errors for
#' each class. Different weightings can be provided for each class. The default is 0.6, which results in prioritizing
#' false positive errors relative to false negative errors.
#' @param smooth Smoothing factor to avoid divide-by-zero errors and provide numeric stability. Default is 1e-8.
#' Recommend using the default.
#' @param chnDim TRUE or FALSE. Whether the channel dimension is included in the target tensor:
#' (Batch, Channel, Height, Width) as opposed to (Batch, Channel, Height, Width). If the channel dimension
#' is included, this should be set to TRUE. If it is not, this should be set to FALSE. Default is TRUE.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param clsWghtsDist Vector of class weights for use in calculating a weighted version of the CE loss.
#' Default is for all classes to be equally weighted.
#' @param clsWghtsReg Vector of class weights for use in calculating a weighted version of the
#' region-based loss. Default is for all classes to be equally weighted.
#' @param useLogCosH TRUE or FALSE. Whether or not to apply a logCosH transformation to the region-based
#' loss. Default is FALSE.
#' @return Loss metric for use in training process.
#' @export
define_unified_focal_loss <- function(pred, #Input predictions as N,C,H,W
                                      target, #Input targets as N,C,H,W or N,H,W
                                      nCls=3,
                                      lambda=.5,
                                      gamma=0,
                                      delta=0.6,
                                      smooth = 1,
                                      chnDim=TRUE,
                                      zeroStart=TRUE,
                                      clsWghtsDist=1,
                                      clsWghtsReg=1,
                                      useLogCosH =FALSE
                          ){

  #============Prepare data======================

  gammaRep = rep(gamma, nCls)

  if(length(delta)==1){
    delta = rep(delta, nCls)
  }

  if(length(clsWghtsDist)==1){
    clsWghtsDist = rep(clsWghtsDist, nCls)
  }

  if(length(clsWghtsReg)==1){
    clsWghtsReg = rep(clsWghtsReg, nCls)
  }

  #Add channel/class dimension if missing

  target1 <- torch::torch_tensor(target, dtype=torch::torch_long())

  if(chnDim == FALSE){
    target1 <- target1$unsqueeze(dim=2)
  }

  #Add 1 if class codes start at 0
  if(zeroStart == TRUE){
    target1 <- torch::torch_tensor(target1+1, dtype=torch::torch_long())
  }


  #Apply softmax to logits along class dimension
  pred_soft <- pred |>
    torch::nnf_softmax(dim = 2)

  #One hot encode masks
  target_one_hot <- torch::nnf_one_hot(target1, num_classes = nCls)
  target_one_hot <- target_one_hot$squeeze()
  target_one_hot <- target_one_hot$permute(c(1,4,2,3))

  #===================Calculate distribution-based loss============================

  #Calculate focal CE loss with gamma and class weights
  #https://github.com/pytorch/vision/issues/3250
  targetCE <- target1$squeeze()
  wghtT <- torch::torch_tensor(clsWghtsDist)
  ceL = torch::nnf_cross_entropy(pred, targetCE, weight=wghtT, reduction="none")
  pt = torch::torch_exp(-ceL)
  mFL <- ((1-pt)**(1.0-gamma))*ceL

  #sum of all weights
  wghtMet <- pred
  wghtMet[] = 0.0
  for(i in 1:length(clsWghtsDist)){
    wghtMet[,i,,] = clsWghtsDist[i]
  }
  wghtSum <- torch::torch_sum(target_one_hot*wghtMet)

  #Get mean distribution-based loss for all pixels
  distMetric <- torch::torch_sum(mFL)/wghtSum

  #===================Calculate region-based loss============================

  #Get tps, fps, and fns
  tps <- torch::torch_sum(pred_soft * target_one_hot, dim=c(1,3,4))
  fps <- torch::torch_sum(pred_soft * (1.0 - target_one_hot), dim=c(1,3,4))
  fns <- torch::torch_sum((1.0 - pred_soft) * target_one_hot, dim=c(1,3,4))

  #Calculated modified Tversky Index using tps, fps, fns, and delta parameter
  mTI <- (tps)/(tps + ((1.0-delta) * fps) + (delta * fns))

  #Apply class-level focal correction using gamma
  regMetric <- (1-mTI)**gammaRep

  #Apply class-level weights
  clsWghtsRegT <- torch::torch_tensor(clsWghtsReg, dtype=torch::torch_float32())
  regMetric <- regMetric*clsWghtsRegT

  #Get macro-averaged focal tversky loss
  regMetric <- torch::torch_sum(regMetric)/torch::torch_sum(clsWghtsRegT)

  #Apply log-cosh correction if desired
  if(useLogCosH == TRUE){
    regMetric <- torch::torch_log(torch::torch_cosh(regMetric))
  }

  #Calculate combined metrics using relative weightings specified by lambda
  comboMetric <- (lambda*distMetric)+((1-lambda)*regMetric)

  return(comboMetric)
}

#' defineUnifiedFocalLoss
#'
#' Define a loss for semantic segmentation using a modified unified focal loss framework as a subclass of torch::nn_module()
#'
#' Implementation of modified version of the unified focal dice loss after:
#'
#' Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal loss:
#' Generalising dice and cross entropy-based losses to handle class imbalanced
#' medical image segmentation. Computerized Medical Imaging and Graphics, 95, p.102026.
#'
#' Modifications include (1) allowing users to define class weights for both the distribution-
#' based and region-based metrics, (2) using class weights as opposed to the symmetric and
#' asymmetric methods implemented by the authors, and (3) including an option to apply
#' a logcosh transform for the region-based loss.
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
#' classes (region-based losses). Larger gamma values put increased weight on difficult samples or classes.
#' Using a value of 0 equates to not using a focal adjustment.
#'
#' The delta term controls the relative weight of
#' false positive and false negative errors for each class. The default is 0.6 for each class, which results in
#' placing a higher weight on false positive as opposed to false negative errors relative to that class.
#'
#' By adjusting the lambda, gamma, delta, and class weight terms, the user can implement a variety of different loss metrics
#' including cross entropy loss, weighted cross entropy loss, focal cross entropy loss, focal weighted cross entropy loss,
#' Dice loss, focal Dice loss, Tversky loss, and focal Tversky loss. Please see the associated vignettes that discuss how
#' to parameterize the function to obtain different loss metrics.
#'
#' @param pred Tensor of predicted class logits. Should be of shape (mini-batch,
#' class, width, height) where the class dimension has a length equal to the number
#' of classes being differentiated. For a binary classification, output can be provided
#' as (mini-batch, class, width, height) or (mini-batch, width, height) if only the positive
#' case logit is returned.
#' @param target Tensor or predicted class indices from 0 to n-1 or 1 to n where n is the
#' number of classes. For a binary classification, only the positive case logit can be returned.
#' Shape can be (mini-batch, class, width, height) or (mini-batch, width, height)
#' @param nCls number of classes being differentiated. Should be 1 for a binary classification
#' where only the positive case logit is returned. Default is 3.
#' @param lambda Term used to control the relative weighting of the distribution- and region-based
#' losses. Default is 0.5, or equal weighting between the losses. If lambda = 1, only the distribution-
#' based loss is considered. If lambda = 0, only the region-based loss is considered. Values between 0.5
#' and 1 put more weight on the distribution-based loss while values between 0 and 0.5 put more
#' weight on the region-based loss.
#' @param gamma Parameter that controls increased weighting applied to difficult-to-predict pixels (for
#' distribution-based losses) or difficult-to-predict classes (region-based losses). Default is 0, or no focal
#' weighting is applied.
#' @param delta Parameter that controls the relative weightings of false positive and false negative errors for
#' each class. Different weightings can be provided for each class. The default is 0.6, which results in prioritizing
#' false positive errors relative to false negative errors.
#' @param smooth Smoothing factor to avoid divide-by-zero errors and provide numeric stability. Default is 1e-8.
#' Recommend using the default.
#' @param chnDim TRUE or FALSE. Whether the channel dimension is included in the target tensor:
#' (Batch, Channel, Height, Width) as opposed to (Batch, Channel, Height, Width). If the channel dimension
#' is included, this should be set to TRUE. If it is not, this should be set to FALSE. Default is TRUE.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param clsWghtsDist Vector of class weights for use in calculating a weighted version of the CE loss.
#' Default is for all classes to be equally weighted.
#' @param clsWghtsReg Vector of class weights for use in calculating a weighted version of the
#' region-based loss. Default is for all classes to be equally weighted.
#' @param useLogCosH TRUE or FALSE. Whether or not to apply a logCosH transformation to the region-based
#' loss. Default is FALSE.
#' @return Loss metric for use in training process.
#' @export
defineUnifiedFocalLoss <- torch::nn_module(#This is simply a class-based version of function defined above/implements function internally
  initialize = function(nCls=3,
                        lambda=.5,
                        gamma=rep(.5, nCls),
                        delta=rep(0.6, nCls),
                        smooth = 1e-8,
                        chnDim=TRUE,
                        zeroStart=TRUE,
                        clsWghtsDist=rep(1, nCls),
                        clsWghtsReg=rep(1,nCls),
                        useLogCosH =FALSE){

    self$nCls = nCls
    self$lambda = lambda
    self$gamma = gamma
    self$delta= delta
    self$smooth = smooth
    self$chnDim= chnDim
    self$zeroStart = zeroStart
    self$clsWghtsDist = clsWghtsDist
    self$clsWghtsReg = clsWghtsReg
    self$useLogCosH =useLogCosH

  },

  forward = function(pred, target){
    return(define_unified_focal_loss(self$nCls,
                                     self$lambda,
                                     self$gamma,
                                     self$delta,
                                     self$smooth,
                                     self$chnDim,
                                     self$zeroStart,
                                     self$clsWghtsDist,
                                     self$clsWghtsReg,
                                     self$useLogCosH)
    )
  }
)




#Multiclass example
#refC <- terra::rast("C:/Users/vidcg/Dropbox/code_dev/geodl/data/metricCheck/multiclass_reference.tif")
#predL <- terra::rast("C:/Users/vidcg/Dropbox/code_dev/geodl/data/metricCheck/multiclass_logits.tif")
#predC <- terra::rast("C:/Users/vidcg/Dropbox//code_dev/geodl/data/metricCheck/multiclass_prediction.tif")
#names(refC) <- "reference"
#names(predC) <- "prediction"
#cm <- terra::crosstab(c(predC, refC))
#yardstick::f_meas(cm, estimator="macro")
#yardstick::accuracy(cm, estimator="micro")
#yardstick::recall(cm, estimator="macro")
#yardstick::precision(cm, estimator="macro")


#predL <- terra::as.array(predL)
#refC <- terra::as.array(refC)

#target <- torch::torch_tensor(refC, dtype=torch::torch_long())
#pred <- torch::torch_tensor(predL, dtype=torch::torch_float32())
#target <- target$permute(c(3,1,2))
#pred <- pred$permute(c(3,1,2))

#target <- target$unsqueeze(1)
#pred <- pred$unsqueeze(1)

#target <- torch::torch_cat(list(target, target), dim=1)
#pred <- torch::torch_cat(list(pred, pred), dim=1)

#define_unified_focal_loss(pred=pred,
                          #target=target,
                          #nCls=5,
                          #lambda=0,
                          #gamma= 1,
                          #delta= 0.6,
                          #smooth = 1e-8,
                          #chnDim=TRUE,
                          #zeroStart=TRUE,
                          #clsWghtsDist=1,
                          #clsWghtsReg=1,
                          #useLogCosH =FALSE)


#' define_unified_focal_lossDS
#'
#' Define a loss for semantic segmentation using a modified unified focal loss framework (function) with deep supervision.
#'
#' Implementation of modified version of the unified focal dice loss after:
#'
#' Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal loss:
#' Generalising dice and cross entropy-based losses to handle class imbalanced
#' medical image segmentation. Computerized Medical Imaging and Graphics, 95, p.102026.
#'
#' Modifications include (1) allowing users to define class weights for both the distribution-
#' based and region-based metrics, (2) using class weights as opposed to the symmetric and
#' asymmetric methods implemented by the authors, and (3) including an option to apply
#' a logcosh transform for the region-based loss.
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
#' classes (region-based losses). Larger gamma values put increased weight on difficult samples or classes.
#' Using a value of 0 equates to not using a focal adjustment.
#'
#' The delta term controls the relative weight of
#' false positive and false negative errors for each class. The default is 0.6 for each class, which results in
#' placing a higher weight on false positive as opposed to false negative errors relative to that class.
#'
#' By adjusting the lambda, gamma, delta, and class weight terms, the user can implement a variety of different loss metrics
#' including cross entropy loss, weighted cross entropy loss, focal cross entropy loss, focal weighted cross entropy loss,
#' Dice loss, focal Dice loss, Tversky loss, and focal Tversky loss.
#'
#' @param pred Tensor of predicted class logits. Should be of shape (mini-batch,
#' class, width, height) where the class dimension has a length equal to the number
#' of classes being differentiated. For a binary classification, output can be provided
#' as (mini-batch, class, width, height) or (mini-batch, width, height) if only the positive
#' case logit is returned.
#' @param target Tensor or predicted class indices from 0 to n-1 or 1 to n where n is the
#' number of classes. For a binary classification, only the positive case logit can be returned.
#' Shape can be (mini-batch, class, width, height) or (mini-batch, width, height).
#' @param nCls number of classes being differentiated. Should be 1 for a binary classification
#' where only the positive case logit is returned. Default is 3.
#' @param dsWghts Vector of 4 weights. Weights to apply to the losses calculated at each spatial
#' resolution when using deep supervision. The default is c(.6, .2, .1, .1) where larger weights are
#' placed on the results at a higher spatial resolution.
#' @param lambda Term used to control the relative weighting of the distribution- and region-based
#' losses. Default is 0.5, or equal weighting between the losses. If lambda = 1, only the distribution-
#' based loss is considered. If lambda = 0, only the region-based loss is considered. Values between 0.5
#' and 1 put more weight on the distribution-based loss while values between 0 and 0.5 put more
#' weight on the region-based loss.
#' @param gamma Parameter that controls increased weighting applied to difficult-to-predict pixels (for
#' distribution-based losses) or difficult-to-predict classes (region-based losses). Default is 0, or no focal
#' weighting is applied.
#' @param delta Parameter that controls the relative weightings of false positive and false negative errors for
#' each class. Different weightings can be provided for each class. The default is 0.6, which results in prioritizing
#' false positive errors relative to false negative errors.
#' @param smooth Smoothing factor to avoid divide-by-zero errors and provide numeric stability. Default is 1e-8.
#' Recommend using the default.
#' @param chnDim TRUE or FALSE. Whether the channel dimension is included in the target tensor:
#' (Batch, Channel, Height, Width) as opposed to (Batch, Channel, Height, Width). If the channel dimension
#' is included, this should be set to TRUE. If it is not, this should be set to FALSE. Default is TRUE.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param clsWghtsDist Vector of class weights for use in calculating a weighted version of the CE loss.
#' Default is for all classes to be equally weighted.
#' @param clsWghtsReg Vector of class weights for use in calculating a weighted version of the
#' region-based loss. Default is for all classes to be equally weighted.
#' @param useLogCosH TRUE or FALSE. Whether or not to apply a logCosH transformation to the region-based
#' loss. Default is FALSE.
#' @return Loss metric for use in training process.
#' @export
define_unified_focal_lossDS <- function(pred, #Input predictions as N,C,H,W
                                      target, #Input targets as N,C,H,W or N,H,W
                                      nCls=3,
                                      dsWghts=c(.6,.2,.1, .1),
                                      lambda=.5,
                                      gamma=0,
                                      delta=0.6,
                                      smooth = 1,
                                      chnDim=TRUE,
                                      zeroStart=TRUE,
                                      clsWghtsDist=1,
                                      clsWghtsReg=1,
                                      useLogCosH =FALSE
){

  #============Prepare data======================

  pred1 <- pred[1]
  pred2 <- pred[2]
  pred3 <- pred[3]
  pred4 <- pred[4]

  target1 <- target[1]
  target2 <- target[2]
  target3 <- target[3]
  target4 <- target[4]


  gammaRep = rep(gamma, nCls)

  if(length(delta)==1){
    delta = rep(delta, nCls)
  }

  if(length(clsWghtsDist)==1){
    clsWghtsDist = rep(clsWghtsDist, nCls)
  }

  if(length(clsWghtsReg)==1){
    clsWghtsReg = rep(clsWghtsReg, nCls)
  }

  #Add channel/class dimension if missing

  target1x <- torch::torch_tensor(target, dtype=torch::torch_long())
  target2x <- torch::torch_tensor(target2, dtype=torch::torch_long())
  target3x <- torch::torch_tensor(target3, dtype=torch::torch_long())
  target4x <- torch::torch_tensor(target4, dtype=torch::torch_long())

  if(chnDim == FALSE){
    target1x <- target1x$unsqueeze(dim=2)
    target2x <- target2x$unsqueeze(dim=2)
    target3x <- target3x$unsqueeze(dim=2)
    target4x <- target4x$unsqueeze(dim=2)
  }

  #Add 1 if class codes start at 0
  if(zeroStart == TRUE){
    target1x <- torch::torch_tensor(target1x+1, dtype=torch::torch_long())
    target2x <- torch::torch_tensor(target2x+1, dtype=torch::torch_long())
    target3x <- torch::torch_tensor(target3x+1, dtype=torch::torch_long())
    target4x <- torch::torch_tensor(target4x+1, dtype=torch::torch_long())
  }


  #Apply softmax to logits along class dimension
  pred_soft1 <- pred1 |>
    torch::nnf_softmax(dim = 2)

  pred_soft2 <- pred2 |>
    torch::nnf_softmax(dim = 2)

  pred_soft3 <- pred3 |>
    torch::nnf_softmax(dim = 2)

  pred_soft4 <- pred4 |>
    torch::nnf_softmax(dim = 2)


  #One hot encode masks
  target_one_hot1 <- torch::nnf_one_hot(target1x, num_classes = nCls)
  target_one_hot1 <- target_one_hot1$squeeze()
  target_one_hot1 <- target_one_hot1$permute(c(1,4,2,3))

  target_one_hot2 <- torch::nnf_one_hot(target2x, num_classes = nCls)
  target_one_hot2 <- target_one_hot2$squeeze()
  target_one_hot2 <- target_one_hot2$permute(c(1,4,2,3))

  target_one_hot3 <- torch::nnf_one_hot(target3x, num_classes = nCls)
  target_one_hot3 <- target_one_hot3$squeeze()
  target_one_hot3 <- target_one_hot3$permute(c(1,4,2,3))

  target_one_hot4 <- torch::nnf_one_hot(target4x, num_classes = nCls)
  target_one_hot4 <- target_one_hot4$squeeze()
  target_one_hot4 <- target_one_hot4$permute(c(1,4,2,3))

  #===================Calculate distribution-based loss============================

  #Calculate focal CE loss with gamma and class weights
  #https://github.com/pytorch/vision/issues/3250
  targetCE1 <- target1$squeeze()
  targetCE2 <- target2$squeeze()
  targetCE3 <- target3$squeeze()
  targetCE4 <- target4$squeeze()


  wghtT <- torch::torch_tensor(clsWghtsDist)


  ceL1 = torch::nnf_cross_entropy(pred1, targetCE1, weight=wghtT, reduction="none")
  pt1 = torch::torch_exp(-ceL1)
  mFL1 <- ((1-pt1)**(1.0-gamma))*ceL1

  ceL2 = torch::nnf_cross_entropy(pred2, targetCE2, weight=wghtT, reduction="none")
  pt2 = torch::torch_exp(-ceL2)
  mFL2 <- ((1-pt2)**(1.0-gamma))*ceL2

  ceL3 = torch::nnf_cross_entropy(pred3, targetCE3, weight=wghtT, reduction="none")
  pt3 = torch::torch_exp(-ceL3)
  mFL3 <- ((1-pt3)**(1.0-gamma))*ceL3

  ceL4 = torch::nnf_cross_entropy(pred4, targetCE4, weight=wghtT, reduction="none")
  pt4 = torch::torch_exp(-ceL4)
  mFL4 <- ((1-pt4)**(1.0-gamma))*ceL4

  #sum of all weights
  wghtMet1 <- pred1
  wghtMet2 <- pred2
  wghtMet3 <- pred3
  wghtMet4 <- pred4

  wghtMet1[] = 0.0
  wghtMet2[] = 0.0
  wghtMet3[] = 0.0
  wghtMet4[] = 0.0


  for(i in 1:length(clsWghtsDist)){
    wghtMet1[,i,,] = clsWghtsDist[i]
  }

  for(i in 1:length(clsWghtsDist)){
    wghtMet2[,i,,] = clsWghtsDist[i]
  }

  for(i in 1:length(clsWghtsDist)){
    wghtMet3[,i,,] = clsWghtsDist[i]
  }

  for(i in 1:length(clsWghtsDist)){
    wghtMet4[,i,,] = clsWghtsDist[i]
  }

  wghtSum1 <- torch::torch_sum(target_one_hot*wghtMet1)
  wghtSum2 <- torch::torch_sum(target_one_hot*wghtMet2)
  wghtSum3 <- torch::torch_sum(target_one_hot*wghtMet3)
  wghtSum4 <- torch::torch_sum(target_one_hot*wghtMet4)

  #Get mean distribution-based loss for all pixels
  distMetric1 <- torch::torch_sum(mFL1)/wghtSum1
  distMetric2 <- torch::torch_sum(mFL2)/wghtSum2
  distMetric3 <- torch::torch_sum(mFL3)/wghtSum3
  distMetric4 <- torch::torch_sum(mFL4)/wghtSum4

  distMetric <- distMetric1*dsWghts[1]+distMetric2*dsWghts[2]+distMetric3*dsWghts[3]+distMetric4*dsWghts[4]

  #===================Calculate region-based loss============================

  #Get tps, fps, and fns
  tps1 <- torch::torch_sum(pred_soft1 * target_one_hot1, dim=c(1,3,4))
  fps1 <- torch::torch_sum(pred_soft1 * (1.0 - target_one_hot1), dim=c(1,3,4))
  fns1 <- torch::torch_sum((1.0 - pred_soft1) * target_one_hot1, dim=c(1,3,4))

  tps2 <- torch::torch_sum(pred_soft2 * target_one_hot2, dim=c(1,3,4))
  fps2 <- torch::torch_sum(pred_soft2 * (1.0 - target_one_hot2), dim=c(1,3,4))
  fns2 <- torch::torch_sum((1.0 - pred_soft2) * target_one_hot2, dim=c(1,3,4))

  tps3 <- torch::torch_sum(pred_soft3 * target_one_hot3, dim=c(1,3,4))
  fps3 <- torch::torch_sum(pred_soft3 * (1.0 - target_one_hot3), dim=c(1,3,4))
  fns3 <- torch::torch_sum((1.0 - pred_soft3) * target_one_hot3, dim=c(1,3,4))

  tps4 <- torch::torch_sum(pred_soft4 * target_one_hot4, dim=c(1,3,4))
  fps4 <- torch::torch_sum(pred_soft4 * (1.0 - target_one_hot4), dim=c(1,3,4))
  fns4 <- torch::torch_sum((1.0 - pred_soft4) * target_one_hot4, dim=c(1,3,4))

  #Calculated modified Tversky Index using tps, fps, fns, and delta parameter
  mTI1 <- (tps1)/(tps1 + ((1.0-delta) * fps1) + (delta * fns1))
  mTI2 <- (tps)/(tps2 + ((1.0-delta) * fps2) + (delta * fns2))
  mTI3 <- (tps)/(tps3 + ((1.0-delta) * fps3) + (delta * fns3))
  mTI4 <- (tps)/(tps4 + ((1.0-delta) * fps4) + (delta * fns4))

  #Apply class-level focal correction using gamma
  regMetric1 <- (1-mTI1)**gammaRep
  regMetric2 <- (1-mTI2)**gammaRep
  regMetric3 <- (1-mTI3)**gammaRep
  regMetric4 <- (1-mTI4)**gammaRep

  #Apply class-level weights
  clsWghtsRegT <- torch::torch_tensor(clsWghtsReg, dtype=torch::torch_float32())
  regMetric1 <- regMetric1*clsWghtsRegT
  regMetric2 <- regMetric2*clsWghtsRegT
  regMetric3 <- regMetric3*clsWghtsRegT
  regMetric4 <- regMetric4*clsWghtsRegT

  #Get macro-averaged focal tversky loss
  regMetric1 <- torch::torch_sum(regMetric1)/torch::torch_sum(clsWghtsRegT)
  regMetric2 <- torch::torch_sum(regMetric2)/torch::torch_sum(clsWghtsRegT)
  regMetric3 <- torch::torch_sum(regMetric3)/torch::torch_sum(clsWghtsRegT)
  regMetric4 <- torch::torch_sum(regMetric4)/torch::torch_sum(clsWghtsRegT)

  #Apply log-cosh correction if desired
  if(useLogCosH == TRUE){
    regMetric1 <- torch::torch_log(torch::torch_cosh(regMetric1))
    regMetric2 <- torch::torch_log(torch::torch_cosh(regMetric2))
    regMetric3 <- torch::torch_log(torch::torch_cosh(regMetric3))
    regMetric4 <- torch::torch_log(torch::torch_cosh(regMetric4))
  }

  regMetric <- regMetric1*dsWghts[1]+regMetric2*dsWghts[2]+regMetric3*dsWghts[3]+regMetric4*dsWghts[4]


  #Calculate combined metrics using relative weightings specified by lambda
  comboMetric <- (lambda*distMetric)+((1-lambda)*regMetric)

  return(comboMetric)
}


#' defineUnifiedFocalLossDS
#'
#' Define a loss for semantic segmentation using a modified unified focal loss framework as a subclass of torch::nn_module() when using deep supervision.
#'
#' Implementation of modified version of the unified focal dice loss after:
#'
#' Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal loss:
#' Generalising dice and cross entropy-based losses to handle class imbalanced
#' medical image segmentation. Computerized Medical Imaging and Graphics, 95, p.102026.
#'
#' Modifications include (1) allowing users to define class weights for both the distribution-
#' based and region-based metrics, (2) using class weights as opposed to the symmetric and
#' asymmetric methods implemented by the authors, and (3) including an option to apply
#' a logcosh transform for the region-based loss.
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
#' classes (region-based losses). Larger gamma values put increased weight on difficult samples or classes.
#' Using a value of 0 equates to not using a focal adjustment.
#'
#' The delta term controls the relative weight of
#' false positive and false negative errors for each class. The default is 0.6 for each class, which results in
#' placing a higher weight on false positive as opposed to false negative errors relative to that class.
#'
#' By adjusting the lambda, gamma, delta, and class weight terms, the user can implement a variety of different loss metrics
#' including cross entropy loss, weighted cross entropy loss, focal cross entropy loss, focal weighted cross entropy loss,
#' Dice loss, focal Dice loss, Tversky loss, and focal Tversky loss. Please see the associated vignettes that discuss how
#' to parameterize the function to obtain different loss metrics.
#'
#' @param pred Tensor of predicted class logits. Should be of shape (mini-batch,
#' class, width, height) where the class dimension has a length equal to the number
#' of classes being differentiated. For a binary classification, output can be provided
#' as (mini-batch, class, width, height) or (mini-batch, width, height) if only the positive
#' case logit is returned.
#' @param target Tensor or predicted class indices from 0 to n-1 or 1 to n where n is the
#' number of classes. For a binary classification, only the positive case logit can be returned.
#' Shape can be (mini-batch, class, width, height) or (mini-batch, width, height)
#' @param nCls number of classes being differentiated. Should be 1 for a binary classification
#' where only the positive case logit is returned. Default is 3.
#' @param dsWghts Vector of 4 weights. Weights to apply to the losses calculated at each spatial
#' resolution when using deep supervision. The default is c(.6, .2, .1, .1) where larger weights are
#' placed on the results at a higher spatial resolution.
#' @param lambda Term used to control the relative weighting of the distribution- and region-based
#' losses. Default is 0.5, or equal weighting between the losses. If lambda = 1, only the distribution-
#' based loss is considered. If lambda = 0, only the region-based loss is considered. Values between 0.5
#' and 1 put more weight on the distribution-based loss while values between 0 and 0.5 put more
#' weight on the region-based loss.
#' @param gamma Parameter that controls increased weighting applied to difficult-to-predict pixels (for
#' distribution-based losses) or difficult-to-predict classes (region-based losses). Default is 0, or no focal
#' weighting is applied.
#' @param delta Parameter that controls the relative weightings of false positive and false negative errors for
#' each class. Different weightings can be provided for each class. The default is 0.6, which results in prioritizing
#' false positive errors relative to false negative errors.
#' @param smooth Smoothing factor to avoid divide-by-zero errors and provide numeric stability. Default is 1e-8.
#' Recommend using the default.
#' @param chnDim TRUE or FALSE. Whether the channel dimension is included in the target tensor:
#' (Batch, Channel, Height, Width) as opposed to (Batch, Channel, Height, Width). If the channel dimension
#' is included, this should be set to TRUE. If it is not, this should be set to FALSE. Default is TRUE.
#' @param zeroStart TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
#' TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.
#' @param clsWghtsDist Vector of class weights for use in calculating a weighted version of the CE loss.
#' Default is for all classes to be equally weighted.
#' @param clsWghtsReg Vector of class weights for use in calculating a weighted version of the
#' region-based loss. Default is for all classes to be equally weighted.
#' @param useLogCosH TRUE or FALSE. Whether or not to apply a logCosH transformation to the region-based
#' loss. Default is FALSE.
#' @return Loss metric for use in training process.
#' @export
defineUnifiedFocalLossDS <- torch::nn_module(#This is simply a class-based version of function defined above/implements function internally
  initialize = function(nCls=3,
                        dsWghts=c(.6,.2,.1,.1),
                        lambda=.5,
                        gamma=rep(.5, nCls),
                        delta=rep(0.6, nCls),
                        smooth = 1e-8,
                        chnDim=TRUE,
                        zeroStart=TRUE,
                        clsWghtsDist=rep(1, nCls),
                        clsWghtsReg=rep(1,nCls),
                        useLogCosH =FALSE){

    self$nCls = nCls
    self$dsWghts=dsWghts
    self$lambda = lambda
    self$gamma = gamma
    self$delta= delta
    self$smooth = smooth
    self$chnDim= chnDim
    self$zeroStart = zeroStart
    self$clsWghtsDist = clsWghtsDist
    self$clsWghtsReg = clsWghtsReg
    self$useLogCosH =useLogCosH

  },

  forward = function(pred, target){
    return(define_unified_focal_lossDS(self$nCls,
                                     self$dsWghts,
                                     self$lambda,
                                     self$gamma,
                                     self$delta,
                                     self$smooth,
                                     self$chnDim,
                                     self$zeroStart,
                                     self$clsWghtsDist,
                                     self$clsWghtsReg,
                                     self$useLogCosH)
    )
  }
)

