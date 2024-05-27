#' viewBatch
#'
#' Generate image grid of mini-batch of image chips and associated masks created by a DataLoader.
#'
#' The goal of this function is to provide a visual check of a mini-batch of image chips and associated masks
#' generated from a DataLoader.
#'
#' @param dataLoader Instantiated instance of a DataLoader created using torch::dataloader().
#' @param nCols Number of columns in the image grid. Default is 3.
#' @param r Index of channel to assign to red channel. Default is 1 or first channel.
#' For gray scale or single-band images, assign the same index to all three bands.
#' @param g Index of channel to assign to green channel. Default is 2 or the second channel.
#' For gray scale or single-band images, assign the same index to all three bands.
#' @param b Index of channel to assign to blue channel. Default is 3 or the third channel.
#' For gray scale or single-band images, assign the same index to all three bands.
#' @param cNames Vector of class names. Must be the same length as number of classes.
#' @param cColors Vector of color values to use to display the masks. Colors are applied based on the
#' order of class indices. Length of vector must be the same as the number of classes.
#' @return Image grids of example chips and masks loaded from a mini-batch produced by the DataLoader.
#' @examples
#' /dontrun{
#' viewBatch(dataLoader=trainDL,
#'           nCols = 5,
#'           r = 1,
#'           g = 2,
#'           b = 3,
#'           cNames=c("Background", "Mine"),
#'           cColors=c("gray", "darksalmon"))
#' }
#' @export
viewBatch <- function(dataLoader,
                      nCols = 3,
                      r = 1,
                      g = 2,
                      b = 3,
                      cNames,
                      cColors){

  batch1 <- dataLoader$.iter()$.next()


  masks <- batch1$mask
  images <- batch1$image

  masks <- torch::torch_tensor(masks, dtype=torch::torch_float32())

  theImgGrid <- torchvision::vision_make_grid(images, num_rows=nCols)$permute(c(2,3,1))
  theMskGrid <- torchvision::vision_make_grid(masks, num_rows=nCols)$permute(c(2,3,1))

  img1 <- terra::rast(as.array(theImgGrid)*255)
  msk1 <- terra::rast(as.array(theMskGrid))

  terra::plotRGB(img1, r=r, g=g, b=b, scale=1, axes=FALSE, stretch="lin")
  terra::plot(msk1, type="classes", axes=FALSE, levels=cNames, col=cColors)
}
