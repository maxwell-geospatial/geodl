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
#' @param cCodes class indices as a vector of integer values equal in length to the number of
#' classes.
#' @param cNames Vector of class names. Must be the same length as number of classes.
#' @param cColors Vector of color values to use to display the masks. Colors are applied based on the
#' order of class indices. Length of vector must be the same as the number of classes.
#' @param padding to place between chips in plot. Default is 10.
#' @return Image grids of example chips and masks loaded from a mini-batch produced by the DataLoader.
#' @examples
#' \dontrun{
#' viewBatch(dataLoader=trainDL,
#'           nCols = 5,
#'           r = 1,
#'           g = 2,
#'           b = 3,
#'           cCodes=c(1,2),
#'           cNames=c("Background", "Mine"),
#'           cColors=c("gray", "darksalmon"))
#' }
#' @export
viewBatch <- function(dataLoader,
                      nCols = 3,
                      r = 1,
                      g = 2,
                      b = 3,
                      cCodes,
                      cNames,
                      cColors,
                      padding=10){

  batch1 <- dataLoader$.iter()$.next()


  masks <- batch1$mask
  images <- batch1$image

  masks <- torch::torch_tensor(masks, dtype=torch::torch_float32())

  theImgGrid <- torchvision::vision_make_grid(images, num_rows=nCols, padding=padding,)$permute(c(2,3,1))
  theMskGrid <- torchvision::vision_make_grid(masks, num_rows=nCols, padding=padding, pad_value=-1, scale=FALSE)$permute(c(2,3,1))

  img1 <- terra::rast(as.array(theImgGrid)*255)
  msk1 <- terra::rast(as.array(theMskGrid))

  msk1 <- terra::subst(msk1, -1, NA)

  catData <- data.frame(cCodes=cCodes,
                        cNames=cNames,
                        cColors=cColors)

  used <- usedCodes <- terra::unique(msk1) |> as.vector() |> unlist()

  catData <- catData |> dplyr::filter(cCodes %in% used)

  layout(matrix(1:2, nrow = 2), heights = c(1, 1))
  par(mar = c(1, 1, 1, 1))

  terra::plotRGB(img1, r=r, g=g, b=b, scale=1, axes=FALSE, stretch="lin")
  terra::plot(msk1, type="classes", axes=FALSE, levels=catData$cNames, col=catData$cColors)

  layout(1)
}
