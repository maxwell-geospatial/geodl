#' viewBatchPreds
#'
#' Generate image grid of mini-batch of image chips, masks, and predictions for all samples in a DataLoader mini-batch.
#'
#' The goal of this function is to provide a visual check of predictions for a mini-batch of data.
#'
#' @param dataLoader Instantiated instance of a DataLoader created using torch::dataloader().
#' @param model Fitted model used to predict mini-batch.
#' @param mode "multiclass" or "binary". If the prediction returns the positive case logit
#' for a binary classification problem, use "binary". If 2 or more class logits are returned,
#' use "multiclass". This package treats all cases as multiclass.
#' @param nCols Number of columns in the image grid. Default is 3.
#' @param r Index of channel to assign to red channel. Default is 1 or the first channel.
#' For gray scale or single-band images, assign the same index to all three bands.
#' @param g Index of channel to assign to green channel. Default is 2 or the second channel.
#' For gray scale or single-band images, assign the same index to all three bands.
#' @param b Index of channel to assign to blue channel. Default is 3 or the third channel.
#' For gray scale or single-band images, assign the same index to all three bands.
#' @param cNames Vector of class names. Must be the same length as number of classes.
#' @param cColors Vector of color values to use to display the masks. Colors are applied based on the
#' order of class indices. Length of vector must be the same as the number of classes.
#' @param useCUDA TRUE or FALSE. Default is FALSE. If TRUE, GPU will be used to predict
#' the data mini-batch. If FALSE, predictions will be made on the CPU. We recommend using a GPU.
#' @param probs TRUE or FALSE. Default is FALSE. If TRUE, rescaled logits will be
#' shown as opposed to the hard classification. If FALSE, hard classification will be
#' shown. For a binary problem where only the positive case logit is returned, the logit
#' is transformed using a sigmoid function. When 2 or more classes are predicted, softmax
#' is used to rescale the logits.
#' @param usedDS TRUE or FALSE. Must be set to TRUE when using deep supervision. Default is FALSE,
#' or it is assumed that deep supervision is not used.
#' @return Image grids of example chips, reference masks, and predictions loaded from a mini-batch provided by the DataLoader.
#' @export
viewBatchPreds <- function(dataLoader,
                           model,
                           mode="multiclass",
                           nCols = 4,
                           r = 1,
                           g = 2,
                           b = 3,
                           cCodes,
                           cNames,
                           cColors,
                           useCUDA=TRUE,
                           probs=FALSE,
                           usedDS=FALSE){

  batch1 <- dataLoader$.iter()$.next()

  nSamps <- dataLoader$batch_size
  if(usedDS == TRUE){
    masks <- batch1$mask
    images <- batch1$image
    model2 <- model$model
  }else{
    masks <- batch1$mask
    images <- batch1$image
  }


  masks <- torch::torch_tensor(masks, dtype=torch_float32())

  if(useCUDA == TRUE){
    images <- images$to(device="cuda")
  }

  if(usedDS==TRUE){
    preds <- model2(images)
  }else{
    preds <- predict(model, batch1$image)
  }

  if(usedDS==TRUE){
    preds <- preds[[1]]
  }

  theImgGrid <- torchvision::vision_make_grid(batch1$image, num_rows=nCols)$permute(c(2,3,1))$to(device="cpu")
  theMskGrid <- torchvision::vision_make_grid(masks, scale=FALSE, num_rows=nCols)$permute(c(2,3,1))$cpu()$to(device="cpu")
  img1 <- terra::rast(as.array(theImgGrid)*255)
  msk1 <- terra::rast(as.array(theMskGrid))

  if(min(cCodes>0)){
    msk1[msk1 == 0] <- NA
  }

  terra::plotRGB(img1, r=r, g=g, b=b, scale=1, axes=FALSE, stretch="lin")
  terra::plot(msk1, type="classes", axes=FALSE, levels=cNames, col=cColors, main="Reference")

  if(probs == TRUE & mode=="multiclass"){
    preds <- torch::nnf_softmax(preds, dim=2)
    thePredsGrid <- torchvision::vision_make_grid(preds, num_rows=nCols)$permute(c(2,3,1))$cpu()$to(device="cpu")
    pred1 <- terra::rast(as.array(thePredsGrid))
    terra::plot(pred1, type="continuous", axes=FALSE, main="Predictions", col=terra::map.pal("grey"))
  }else if(probs == FALSE & mode=="multiclass"){
    predsC <- torch::torch_argmax(preds, dim=2)
    predsC2 <- predsC$unsqueeze(2)
    predsC3 <- torch::torch_tensor(predsC2, dtype=torch_float32())
    thePredsGrid <- torchvision::vision_make_grid(predsC3, scale=FALSE, num_rows=nCols)$permute(c(2,3,1))$cpu()$to(device="cpu")
    pred1 <- terra::rast(as.array(thePredsGrid))
    if(min(cCodes>0)){
      pred1[pred1 == 0] <- NA
    }
    terra::plot(pred1, type="classes", axes=FALSE, levels=cNames, col=cColors, main="Predictions")
  }else if(probs == TRUE & mode == "binary"){
    preds <- torch::nnf_sigmoid(preds)
    thePredsGrid <- torchvision::vision_make_grid(preds, num_rows=nCols)$permute(c(2,3,1))$cpu()$to(device="cpu")
    pred1 <- terra::rast(as.array(thePredsGrid))
    terra::plot(pred1, type="continuous", axes=FALSE, main="Predictions", col=map.pal("grey"))
  }else{
    preds <- torch::nnf_sigmoid(preds)
    preds <- torch::torch_round(preds)
    thePredsGrid <- torchvision::vision_make_grid(preds, num_rows=nCols)$permute(c(2,3,1))$cpu()$to(device="cpu")
    pred1 <- terra::rast(as.array(thePredsGrid))
    if(min(cCodes>0)){
      pred1[pred1 == 0] <- NA
    }
    terra::plot(pred1, type="classes", axes=FALSE, levels=cNames, col=cColors, main="Predictions")
  }
}
