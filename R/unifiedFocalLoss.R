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
#' By adjusting the lambda, gamma, delta, and class weight terms, the user can implement a variety of different loss metrics
#' including cross entropy loss, weighted cross entropy loss, focal cross entropy loss, focal weighted cross entropy loss,
#' Dice loss, focal Dice loss, Tversky loss, and focal Tversky loss.
#'
#' @param nCls Number of classes being differentiated.
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
#' #Instantiate loss metric
#' myDiceLoss <- defineUnifiedFocalLoss(nCls=3,
#'                                     lambda=0, #Only use region-based loss
#'                                     gamma= 1,
#'                                     delta= 0.5, #Equal weights for FP and FN
#'                                     smooth = 1e-8,
#'                                     zeroStart=FALSE,
#'                                     clsWghtsDist=1,
#'                                     clsWghtsReg=1,
#'                                     useLogCosH =FALSE,
#'                                     device='cpu')
#' #Calculate loss
#' myDiceLoss(pred, ref)
#' }
#' @export
defineUnifiedFocalLoss <- torch::nn_module(#This is simply a class-based version of function defined above/implements function internally
  initialize = function(nCls=3,
                        lambda=.5,
                        gamma=.5,
                        delta=0.6,
                        smooth = 1e-8,
                        zeroStart=TRUE,
                        clsWghtsDist=1,
                        clsWghtsReg=1,
                        useLogCosH =FALSE,
                        device="cuda"){

    self$nCls = nCls
    self$lambda = lambda
    self$gamma = gamma
    self$delta= delta
    self$smooth = smooth
    self$zeroStart = zeroStart
    self$clsWghtsDist = clsWghtsDist
    self$clsWghtsReg = clsWghtsReg
    self$useLogCosH =useLogCosH
    self$device=device

  },

  forward = function(pred, target){

    # Preparation ------------------------------------------------------------

    #Replicate gamma to vector with same length as class count
    gammaRep = rep(self$gamma, self$nCls)

    #If only one delta value is provided, replicated to a vector with same length as the class count
    if(length(self$delta)==1){
      delta2 = rep(self$delta, self$nCls)
    }else{
      delta2 = self$delta
    }

    #If only one class weight is provided, replicate to a vector with the same length as the class count

    #For distribution-based loss
    if(length(self$clsWghtsDist)==1){
      clsWghtsDist2 = rep(self$clsWghtsDist, self$nCls)
    }else{
      clsWghtsDist2 = self$clsWghtsDist
    }

    #For region-based loss
    if(length(self$clsWghtsReg)==1){
      clsWghtsReg2 = rep(self$clsWghtsReg, self$nCls)
    }else{
      clsWghtsReg2 = self$clsWghtsReg
    }

    #Convert smooth, lambda, delta, gamma, and class weight vectors/parameters to torch tensors and move to device
    smoothT <- torch::torch_tensor(self$smooth,
                                   dtype=torch::torch_float32(),
                                   device=self$device)
    lambdaT <- torch::torch_tensor(self$lambda,
                                   dtype=torch::torch_float32(),
                                   device=self$device)
    deltaT <- torch::torch_tensor(delta2,
                                  dtype=torch::torch_float32(),
                                  device=self$device)
    gammaT <- torch::torch_tensor(self$gamma,
                                  dtype=torch::torch_float32(),
                                  device=self$device)
    gammaRepT <- torch::torch_tensor(gammaRep,
                                     dtype=torch::torch_float32(),
                                     device=self$device)
    wghtT <- torch::torch_tensor(clsWghtsDist2,
                                 dtype=torch::torch_float32(),
                                 device=self$device)
    clsWghtsRegT <- torch::torch_tensor(clsWghtsReg2,
                                        dtype=torch::torch_float32(),
                                        device=self$device)

    #Convert target to long type
    target1 <- torch::torch_tensor(target,
                                   dtype=torch::torch_long(),
                                   device=self$device)

    #Add 1 if class codes start at 0 for one-hot encoding
    if(self$zeroStart == TRUE){
      target1 <- torch::torch_tensor(target1+1,
                                     dtype=torch::torch_long(),
                                     device=self$device)
    }

    #Apply softmax to logits along class dimension (all predictions at pixel will sum to 1)
    pred_soft <- pred |>
      torch::nnf_softmax(dim = 2)

    #One hot encode masks and reshape as needed
    target_one_hot <- torch::nnf_one_hot(target1,
                                         num_classes = self$nCls)
    target_one_hot <- target_one_hot$squeeze()
    target_one_hot <- target_one_hot$permute(c(1,4,2,3))

    # Distribution-based loss -------------------------------------------------

    #Calculate focal CE loss with gamma and class weights
    #https://github.com/pytorch/vision/issues/3250

    if(self$lambda > 0){
      #Remove class dimension (required for CE loss as implemented with torch)
      targetCE <- target1$squeeze()

      #Calculate CE loss with no reduction (use predicted logits with no softmax applied)
      ceL = torch::nnf_cross_entropy(pred,
                                     targetCE,
                                     weight=wghtT,
                                     reduction="none")

      #Calculate modified focal CE loss
      pt = torch::torch_exp(-ceL)
      mFL <- ((1.0-pt)**(1.0-gammaT))*ceL

      #sum of all weights
      wghtMet <- pred
      wghtMet[] = 0.0
      for(i in 1:length(clsWghtsDist2)){
        wghtMet[,i,,] = clsWghtsDist2[i]
      }

      wghtMetT <- torch::torch_tensor(wghtMet,
                                      dtype=torch::torch_float32(),
                                      device=self$device)
      wghtSumT <- torch::torch_sum(target_one_hot*wghtMetT)

      #Get mean distribution-based loss for all pixels
      distMetric <- torch::torch_sum(mFL)/wghtSumT
    }

    # Region-based loss -------------------------------------------------------

    if(self$lambda < 1){
      #Get tps, fps, and fns
      tps <- torch::torch_sum(pred_soft * target_one_hot, dim=c(1,3,4))
      fps <- torch::torch_sum(pred_soft * (1.0 - target_one_hot), dim=c(1,3,4))
      fns <- torch::torch_sum((1.0 - pred_soft) * target_one_hot, dim=c(1,3,4))

      #Calculated modified Tversky Index using tps, fps, fns, and delta parameter
      mTI <- (tps + smoothT)/(tps + ((1.0-deltaT) * fps) + (deltaT * fns) + smoothT)

      #Apply class-level focal correction using gamma
      regMetric <- (1.0-mTI)**gammaRepT

      #Apply class-level weights
      regMetric <- regMetric*clsWghtsRegT

      #Get macro-averaged focal tversky loss
      regMetric <- torch::torch_sum(regMetric)/torch::torch_sum(clsWghtsRegT)

      #Apply log-cosh correction if desired
      if(self$useLogCosH == TRUE){
        regMetric <- torch::torch_log(torch::torch_cosh(regMetric))
      }
    }

    if(self$lambda == 1){
      comboMetric <- distMetric
    }else if(self$lambda == 0){
      comboMetric <- regMetric
    }else{
      #Calculate combined metrics using relative weightings specified by lambda
      comboMetric <- (lambdaT*distMetric)+((1.0-lambdaT)*regMetric)
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
#' @param dsWghts Vector of 4 weights. Weights to apply to the losses calculated at each spatial
#' resolution when using deep supervision. The default is c(.6, .2, .1, .1) where larger weights are
#' placed on the results at a higher spatial resolution.
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
defineUnifiedFocalLossDS <- torch::nn_module(#This is simply a class-based version of function defined above/implements function internally
  initialize = function(nCls=3,
                        dsWghts = c(.6,.2,.1,.1),
                        lambda=.5,
                        gamma=.5,
                        delta=0.6,
                        smooth = 1e-8,
                        zeroStart=TRUE,
                        clsWghtsDist=1,
                        clsWghtsReg=1,
                        useLogCosH =FALSE,
                        device="cuda"){

    self$nCls = nCls
    self$dsWghts = dsWghts
    self$lambda = lambda
    self$gamma = gamma
    self$delta= delta
    self$smooth = smooth
    self$zeroStart = zeroStart
    self$clsWghtsDist = clsWghtsDist
    self$clsWghtsReg = clsWghtsReg
    self$useLogCosH =useLogCosH
    self$device=device

    self$wght1 <- torch::torch_tensor(dsWghts[1], dtype=torch::torch_float32(), device=device)
    self$wght2 <- torch::torch_tensor(dsWghts[2], dtype=torch::torch_float32(), device=device)
    self$wght4 <- torch::torch_tensor(dsWghts[3], dtype=torch::torch_float32(), device=device)
    self$wght8 <- torch::torch_tensor(dsWghts[4], dtype=torch::torch_float32(), device=device)

    self$loss1 <- defineUnifiedFocalLoss(self$nCls,
                                  self$lambda,
                                  self$gamma,
                                  self$delta,
                                  self$smooth,
                                  self$zeroStart,
                                  self$clsWghtsDist,
                                  self$clsWghtsReg,
                                  self$useLogCosH,
                                  self$device)

    self$loss2 <- defineUnifiedFocalLoss(self$nCls,
                                  self$lambda,
                                  self$gamma,
                                  self$delta,
                                  self$smooth,
                                  self$zeroStart,
                                  self$clsWghtsDist,
                                  self$clsWghtsReg,
                                  self$useLogCosH,
                                  self$device)

    self$loss4 <- defineUnifiedFocalLoss(self$nCls,
                                  self$lambda,
                                  self$gamma,
                                  self$delta,
                                  self$smooth,
                                  self$zeroStart,
                                  self$clsWghtsDist,
                                  self$clsWghtsReg,
                                  self$useLogCosH,
                                  self$device)

    self$loss8 <- defineUnifiedFocalLoss(self$nCls,
                                  self$lambda,
                                  self$gamma,
                                  self$delta,
                                  self$smooth,
                                  self$zeroStart,
                                  self$clsWghtsDist,
                                  self$clsWghtsReg,
                                  self$useLogCosH,
                                  self$device)
  },

  forward = function(pred, target){
    l1 <- self$loss1(pred[[1]], target)
    l2 <- self$loss2(pred[[2]], target)
    l4 <- self$loss4(pred[[3]], target)
    l8 <- self$loss8(pred[[4]], target)

    lossOut <- ((self$wght1*l1)+(self$wght2*l2)+(self$wght4*l4)+(self$wght8*l8))/(self$wght1+self$wght2+self$wght4+self$wght8)

    return(lossOut)
  }
)
