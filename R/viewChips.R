#' viewChips
#'
#' Plot a grid of image and/or mask chips
#'
#' This function generates a plot of image chips and/or image masks. It serves
#' as a means to visualize chips generated with the makeChips() or makeChipsMultiClass()
#' function. It can be used as a check to make sure chips were generated as expected.
#'
#' @param chpDF Data frame of chip paths created with the makeChipsDF() function.
#' @param folder Full path or path relative to the working directory to the
#' folder containing the image chips and associated masks. You must include the
#' final forward slash in the path (e.g., "C:/data/chips/").
#' @param nSamps Number of samples to include in the grid. The default is 16.
#' @param mode Either "image", "mask" or "both". If "image", a grid is produced
#' for the image chips only. If "mask", a grid is produced for just the masks.
#' If "both", grids are produced for both the image chips and masks. Default is
#' "both".
#' @param justPostitive TRUE or FALSE. If makeChips() was executed using the mode "Divided", you can
#' choose to only show chips that contained some pixels mapped to the positive class.
#' The default is FALSE. This should be left to the default or set to FALSE if chips
#' were generated using a method other than "Divided".
#' @param cCnt Number of columns in the grid. Row X Column count must sum to the number
#' of samples being displayed (nSamps). Default is 4.
#' @param rCnt Number of rows in the grid. Row X Column count must sum to the number
#' of samples being displayed (nSamps). Default is 4.
#' @param r Band number to map to the red channel. Default is 1 or the first channel.
#' @param g Band number to map to the green channel. Default is 2 or the second channel.
#' @param b Band number to map to the red channel. Default is 3 or the third channel.
#' @param cNames Vector of class names. Class names must be provided.
#' @param cColors Vector of colors (named colors, hex codes, or rgb()).
#' Color used to visualize each class is matched based on position
#' in the vector. Colors must be provided.
#' @param useSeed TRUE or FALSE. Whether or not to set a random seed to make result
#' reproducible. If FALSE, seed is ignored. Default is FALSE.
#' @param seed Random seed value. Default is 42. This is ignored if useSeed is FALSE.
#' @return Plot of image chip grid (if mode = "image"); plot of mask chip grid
#' (if mode ="mask"); plot of image and mask chip grids (if model = "both").
#' @export
viewChips <- function(chpDF,
                      folder,
                      nSamps = 16,
                      mode = "both",
                      justPositive = FALSE,
                      cCnt = 4,
                      rCnt = 4,
                      r = 1,
                      g = 2,
                      b = 3,
                      rescale = FALSE,
                      rescaleVal = 1,
                      cNames,
                      cColors,
                      useSeed = FALSE,
                      seed = 42){

  if(justPositive == TRUE){
    chpDF <- chpDF |> dplyr::filter(division == "Positive")
  }

  if(useSeed == TRUE){
      set.seed(seed)
  }
  subset1 <- chpDF |> dplyr::sample_n(nSamps, replace=FALSE)
  testImg <- terra::rast(paste0(folder,subset1[1,"chpPth"]))
  w <- terra::nrow(testImg)
  h <- terra::ncol(testImg)
  l <- terra::nlyr(testImg)
  blankR <- terra::rast(ncols=cCnt*h,
                        nrows=rCnt*w,
                        nlyrs=l,
                        extent=c(xmin=1, xmax=cCnt*w, ymin=1, ymax=rCnt*h))
  blankR[] <- 1

  rSeq <- seq(1,terra::nrow(blankR),w)
  cSeq <- seq(1,terra::ncol(blankR), h)

  theGrid <- expand.grid(rSeq, cSeq)
  names(theGrid) <- c("rI", "cI")
  subset1 <- dplyr::bind_cols(subset1, theGrid)
  if(mode == "both"){
    blankM <- terra::subset(blankR, 1)
    blankM[] <- 0
    for(i in 1:nrow(subset1)){
      img1 <- terra::rast(paste0(folder,subset1[i,"chpPth"]))
      msk1 <- terra::rast(paste0(folder,subset1[i,"mskPth"]))
      if(rescale == TRUE){
        img1 <- img1/rescaleVal
      }
      currentR <-subset1[i,"rI"]
      currentC <-subset1[i, "cI"]
      blankR[currentR:(currentR+w-1), currentC:(currentC+h-1), 1:l] <- img1[]
      blankM[currentR:(currentR+w-1), currentC:(currentC+h-1), 1] <- msk1[]
    }
    imgPlot = terra::plotRGB(blankR, r=1, g=2, b=3, scale=255, axes=FALSE, stretch="lin", maxcell=1000000)
    mskPlot = terra::plot(blankM, type="classes", axes=FALSE, levels=cNames, col=cColors, maxcell=1000000)
  }else if(mode == "image"){
    for(i in 1:nrow(subset1)){
      img1 <- terra::rast(paste0(folder,subset1[i,"chpPth"]))
      if(rescale == TRUE){
        img1 <- img1/rescaleVal
      }
      currentR <-subset1[i,"rI"]
      currentC <-subset1[i, "cI"]
      blankR[currentR:(currentR+w-1), currentC:(currentC+h-1), 1:l] <- img1[]
    }
    terra::plotRGB(blankR, r=r, g=g, b=b, scale=255, axes=FALSE, stretch="lin", maxcell=1000000)
  }else{
    blankM <- terra::subset(blankR, 1)
    blankM[] <- 0
    for(i in 1:nrow(subset1)){
      msk1 <- terra::rast(paste0(folder,subset1[i,"mskPth"]))
      currentR <-subset1[i,"rI"]
      currentC <-subset1[i, "cI"]
      blankM[currentR:(currentR+w-1), currentC:(currentC+h-1), 1] <- msk1[]
    }
    terra::plot(blankM, type="classes", axes=FALSE, levels=cNames, col=cColors, maxcell=1000000)
  }
}
