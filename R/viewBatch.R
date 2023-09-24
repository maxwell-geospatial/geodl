#' viewBatch
#'
#' Generate image grid of batch of image chips and associated masks created by a DataLoader.
#'
#' The goal of this function is to provide a visual check of a batch of image chips and associated masks
#' generated from a DataLoader.
#'
#' @param dataLoader Instantiated instance of a DataLoader created using torch::dataloader().
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
#' @return Image grids of example chips and masks loaded from a batch produced by the DataLoader.
#' @export
viewBatch <- function(dataLoader,
                      chnDim=TRUE,
                      mskLong = TRUE,
                      nRows = 3,
                      r = 1,
                      g = 2,
                      b = 3,
                      cNames,
                      cColors){

  batch1 <- dataLoader$.iter()$.next()

  nSamps <- dataLoader$batch_size
  masks <- batch1$mask

  if(chnDim == FALSE){
    masks <- mask$unsqueeze(2)
  }

  if(mskLong == TRUE){
    masks <- torch::torch_tensor(masks, dtype=torch_float32())
  }

  theImgGrid <- torchvision::vision_make_grid(batch1$image, num_rows=nRows)$permute(c(2,3,1))
  theMskGrid <- torchvision::vision_make_grid(masks, num_rows=nRows)$permute(c(2,3,1))

  img1 <- terra::rast(as.array(theImgGrid)*255)
  msk1 <- terra::rast(as.array(theMskGrid))

  terra::plotRGB(img1, r=r, g=g, b=b, scale=1, axes=FALSE, stretch="lin")
  terra::plot(msk1, type="classes", axes=FALSE, levels=cNames, col=cColors)
}
