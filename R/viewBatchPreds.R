#' viewBatchPreds
#'
#' Generate image grid of batch of image chips, masks, and predictions for all samples in a DataLoader batch.
#'
#' The goal of this function is to provide a visual check of predictions for a batch of data.
#'
#' @param dataLoader Instantiated instance of a DataLoader created using torch::dataloader().
#' @param model Fitted model used to predict batch.
#' @param mode "multiclass" or "binary". If the prediction returns the postive case logit
#' for a binary classification problem, use "binary". If 2 or more class logits are returned,
#' use "multiclass".
#' @param chnDim TRUE or FALSE. Default is TRUE. If TRUE, assumes the target tensor includes the
#' channel dimension: (Batch, Channel, Height, Width). If FALSE, assumes the target tensor does not include the
#' channel dimension: (Channel, Height, Width). The script is written such that it expects the channel dimension.
#' So, if FALSE, the script will add the channel dimension as needed.
#' @param mskLong TRUE or FALSE. Default is TRUE. Data type of target or mask. If the provided target
#' has a data type other than torch_long, this parameter should be set to FALSE. This will cause the
#' script to convert the target to the tensor_long data type as required for the calculations.
#' @param nRows Number of rows in the image grid. Default is 3.
#' @param r Index of channel to assign to red channel. Default is 1.
#' For gray scale or single-band images, assign the same index to all three bands.
#' @param g Index of channel to assign to green channel. Default is 2.
#' For gray scale or single-band images, assign the same index to all three bands.
#' @param b Index of channel to assign to blue channel. Default is 3.
#' For gray scale or single-band images, assign the same index to all three bands.
#' @param cNames Vector of class names. Must be the same length as number of classes.
#' @param cColors Vector of color values to use to display the masks. Colors are applied based on the
#' order of class indices. Length of vector must be the same as the number of classes.
#' @param useCUDA TRUE or FALSE. Default is FALSE. If TRUE, GPU will be used to predict
#' the data batch. If FALSE, predictions will be made on the CPU.
#' @param probs TRUE or FALSE. Default is FALSE. If TRUE, class probabilities will be
#' shown as opposed to the hard classification. If FALSE, hard classification will be
#' shown. For a binary problem, the positive class logit is transformed using a sigmoid
#' function. For multiclass, softmax is used to transform the class logits to probabilities
#' that sum to 1.
#' @param usedDS TRUE or FALSE. Must be set to TRUE when using defineSegDataSetDS(). Default is FALSE,
#' or it is assumed that deep supervision is not used.
#' @return Image grids of example chips and masks loaded from a batch produced by the DataLoader.
#' @export
viewBatchPreds <- function(dataLoader=testDL,
                           model=model2,
                           mode="multiclass",
                           chnDim=FALSE,
                           mskLong = TRUE,
                           nRows = 4,
                           r = 1,
                           g = 2,
                           b = 3,
                           cNames,
                           cColors,
                           useCUDA=TRUE,
                           probs=FALSE,
                           usedDS=FALSE){

  batch1 <- dataLoader$.iter()$.next()

  nSamps <- dataLoader$batch_size
  if(usedDS == TRUE){
    masks <- batch1$mask[1]
    images <- batch1$image
  }else{
    masks <- batch1$mask
    images <- batch1$image
  }

  if(chnDim == FALSE){
    masks <- masks$unsqueeze(2)
  }

  if(mskLong == TRUE){
    masks <- torch::torch_tensor(masks, dtype=torch_float32())
  }

  if(useCUDA == TRUE){
    imgs <- batch1$image$to(device="cuda")
  }

  preds <- predict(model, batch1$image)

  if(usedDS==TRUE){
    preds <- preds[1]
  }

  theImgGrid <- torchvision::vision_make_grid(batch1$image, num_rows=nRows)$permute(c(2,3,1))$to(device="cpu")
  theMskGrid <- torchvision::vision_make_grid(masks, num_rows=nRows)$permute(c(2,3,1))$cpu()$to(device="cpu")
  img1 <- terra::rast(as.array(theImgGrid)*255)
  msk1 <- terra::rast(as.array(theMskGrid))

  terra::plotRGB(img1, r=r, g=g, b=b, scale=1, axes=FALSE, stretch="lin")
  terra::plot(msk1, type="classes", axes=FALSE, levels=cNames, col=cColors, main="Reference")

  if(probs == TRUE & mode=="multiclass"){
    preds <- torch::nnf_softmax(preds, dim=2)
    thePredsGrid <- torchvision::vision_make_grid(preds, num_rows=nRows)$permute(c(2,3,1))$cpu()$to(device="cpu")
    pred1 <- terra::rast(as.array(thePredsGrid))
    terra::plot(pred1, type="continuous", axes=FALSE, main="Predictions", col=terra::map.pal("grey"))
  }else if(probs == FALSE & mode=="multiclass"){
    predsC <- torch::torch_argmax(preds, dim=2)
    predsC2 <- predsC$unsqueeze(2)
    predsC3 <- torch::torch_tensor(predsC2, dtype=torch_float32())
    thePredsGrid <- torchvision::vision_make_grid(predsC3, scale=FALSE, num_rows=nRows)$permute(c(2,3,1))$cpu()$to(device="cpu")
    pred1 <- terra::rast(as.array(thePredsGrid))
    terra::plot(pred1, type="classes", axes=FALSE, levels=cNames, col=cColors, main="Predictions")
  }else if(probs == TRUE & mode == "binary"){
    preds <- torch::nnf_sigmoid(preds)
    thePredsGrid <- torchvision::vision_make_grid(preds, num_rows=nRows)$permute(c(2,3,1))$cpu()$to(device="cpu")
    pred1 <- terra::rast(as.array(thePredsGrid))
    terra::plot(pred1, type="continuous", axes=FALSE, main="Predictions", col=map.pal("grey"))
  }else{
    preds <- torch::nnf_sigmoid(preds)
    preds <- torch::torch_round(preds)
    thePredsGrid <- torchvision::vision_make_grid(preds, num_rows=nRows)$permute(c(2,3,1))$cpu()$to(device="cpu")
    pred1 <- terra::rast(as.array(thePredsGrid))
    terra::plot(pred1, type="classes", axes=FALSE, levels=cNames, col=cColors, main="Predictions")
  }
}
