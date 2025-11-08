makeDynamicChip <- function(chipIn,
                            chipSize,
                            cellSize,
                            doJitter,
                            jitterSD,
                            useSeed,
                            seed){

  chipIn2 <- chipIn |> sf::st_drop_geometry()

  inFeat <- sf::st_read(paste0(chipIn2[1, "featPth"], chipIn2[1, "featName"]), quiet=TRUE)
  inRaster <- terra::rast(paste0(chipIn2[1, "imgPth"], chipIn2[1, "imgName"]))

  if(doJitter == TRUE){
    noise_sd <- jitterSD
    if(useSeed==TRUE){
      set.seed(seed)
      noise <- matrix(rnorm(n = 2, mean = 0, sd = noise_sd), ncol = 2)
    }else{
      noise <- matrix(rnorm(n = 2, mean = 0, sd = noise_sd), ncol = 2)
    }

    origCoords <- sf::st_coordinates(chipIn)
    newCoords <- origCoords+noise
    newCoords <- lapply(1:nrow(newCoords), function(i) sf::st_point(newCoords[i, ]))
    newCoords <- sf::st_sfc(newCoords, crs = sf::st_crs(chipIn))
    chipIn <- sf::st_set_geometry(chipIn, newCoords)
  }

  vB <- sf::st_buffer(chipIn, ((chipSize*cellSize)/2))
  vBB <- sf::st_make_grid(vB, n=1)

  f1V <- sf::st_intersection(inFeat, vBB)

  r1 <- terra::crop(inRaster, vBB)
  m1 <- terra::rasterize(f1V, r1, field="code")
  m1<- terra::ifel(is.na(m1), 0, m1)

  return(list(image=r1, mask=m1))

}





#' defineDynamicSegDataSet
#'
#' Instantiate a subclass of torch::dataset() for geospatial semantic segmentation using dynamically generated chips
#'
#' This function instantiates a subclass of torch::dataset() that dynmaically generates chips
#' using tghe output from makeDynamicChips.R Can also define random augmentations to combat
#' overfitting. Note that horizontal and vertical flips will affect the alignment of the
#' image and associated mask chips. As a result, the same augmentation will be applied
#' to both the image and the mask. Changes in brightness, contrast, gamma, hue, and
#' saturation will not be applied to the masks since alignment is not impacted by these
#' transformations. Predictor variables are generated with three dimensions
#' (channel/variable, width, height) regardless of the number of channels/variables.
#' Masks are generated as three dimensional tensors (class index, width, height).
#'
#' @param chipsSF sf object created by makeDynamicChipsSF().
#' @param chipSize Size of desired image chips. Default is 512 (512x512 cells)
#' @param cellSize Cells size of input data. Default is 1 m.
#' @param doJitter Whether or not to add random noise to chip center coordinates. Default is FALSE.
#' @param jitterSD If doJitter is TRUE, standard deviation of random positional noise to add in
#' both the x and y directions. Default is 15 (15 meters).
#' @param useSeed Whether or not to use a random seed for added jitter noise. Default is FALSE.
#' @param seed Random seed value.
#' @param normalize TRUE or FALSE. Whether to apply normalization. If FALSE,
#' bMns and bSDs are ignored. Default is FALSE. If TRUE, you must provide bMns
#' and bSDs.
#' @param rescaleFactor A rescaling factor to rescale the bands to 0 to 1. For
#' example, this could be set to 255 to rescale 8-bit data. Default is 1 or no
#' rescaling.
#' @param mskRescale Can be used to rescale binary masks that are not scaled from
#' 0 to 1. For example, if masks are scaled from 0 and 255, you can divide by 255 to
#' obtain a 0 to 1 scale. Default is 1 or no rescaling.
#' @param mskAdd Value to add to mask class numeric codes. For example, if class indices
#' start are 0, 1 can be added so that indices start at 1. Default is 0 (return
#' original class codes). Note that several other functions in this package have a zeroStart
#' parameter. If class codes start at 0, this argument should be set to TRUE. If they start at 1,
#' this argument should be set to FALSE. The importance of this arises from the use of one-hot
#' encoding internally, which requires that class indices start at 1.
#' @param bands Vector of bands to include. The default is to only include the
#' first 3 bands. If you want to use a different subset of bands, you must provide
#' a vector of band indices here to override the default.
#' @param bMns Vector of band means. Length should be the same as the number of bands.
#' Normalization is applied before any rescaling within the function.
#' @param bSDs Vector of band standard deviations. Length should be the same
#' as the number of bands. Normalization is applied before any rescaling.
#' @param doAugs TRUE or FALSE. Whether or not to apply data augmentations to combat
#' overfitting. If FALSE, all augmentations parameters are ignored. Data augmentations
#' are generally only applied to the training set. Default is FALSE.
#' @param maxAugs 0 to 7. Maximum number of random augmentations to apply. Default is 0
#' or no augmentations. Must be changed if augmentations are desired.
#' @param probVFlip 0 to 1. Probability of applying vertical flips. Default is 0
#' or no augmentations. Must be changed if augmentations are desired.
#' @param probHFlip 0 to 1. Probability of applying horizontal flips. Default is 0
#' or no augmentations. Must be changed if augmentations are desired.
#' @param probBrightness 0 to 1. Probability of applying brightness augmentation.
#' Default is 0 or no augmentations. Must be changed if augmentations are desired.
#' @param probContrast 0 to 1. Probability of applying contrast augmentations.
#' Default is 0 or no augmentations. Must be changed if augmentations are desired.
#' @param probGamma 0 to 1. Probability of applying gamma augmentations. Default is 0
#' or no augmentations. Must be changed if augmentations are desired.
#' @param probHue 0 to 1. Probability of applying hue augmentations. Default is 0
#' or no augmentations. Must be changed if augmentations are desired.
#' This is only applicable to RGB data.
#' @param probRotate  0 to 1. Probability of applying rotation by 0-, 90-, 180- or 270-degrees.
#' Default is 0 or no augmentations. Must be changed if augmentations are desired.
#' @param probSaturation 0 to 1. Probability of applying saturation augmentations.
#' Default is 0 or no augmentations. Must be changed if augmentations are desired.
#' This is only applicable to RGB data.
#' @param brightFactor Vector of smallest and largest brightness adjustment factors.
#' Random value will be selected between these extremes. The default is 0.8 to 1.2.
#' Can be any non negative number. For example, 0 gives a black image, 1 gives the original
#' image, and 2 increases the brightness by a factor of 2.
#' @param contrastFactor Vector of smallest and largest contrast adjustment factors.
#' Random value will be selected between these extremes. The default is 0.8 to 1.2.
#' Can be any non negative number. For example, 0 gives a solid gray image, 1 gives the original
#' image, and 2 increases the contrast by a factor of 2.
#' @param gammaFactor Vector of smallest and largest gamma values and gain value
#' for a total of 3 values. Random value will be selected between these extremes.
#' The default gamma value range is 0.8 to 1.2 and the default gain is 1. The gain
#' is not randomly altered, only the gamma. Non negative real number. A gamma larger
#' than 1 makes the shadows darker while a gamma smaller than 1 makes dark regions
#' lighter.
#' @param hueFactor Vector of smallest and largest hue adjustment factors.
#' Random value will be selected between these extremes. The default is -0.2 to 0.2.
#' Should be in range -0.5 to 0.5. 0.5 and -0.5 give complete reversal of hue channel
#' in HSV space in positive and negative direction, respectively. 0 means no shift.
#' Therefore, both -0.5 and 0.5 will give an image with complementary colors while
#' 0 gives the original image.
#' @param saturationFactor Vector of smallest and largest saturation adjustment factors.
#' Random value will be selected between these extremes. The default is 0.8 to 1.2.
#' For example, 0 will give a black-and-white image, 1 will give the original image, and 2 will
#' enhance the saturation by a factor of 2.
#' @return A dataset object that can be provided to torch::dataloader().
#' @export
defineDynmamicSegDataSet <- torch::dataset(

  name = "defineDynamicSegDataSet",

  initialize = function(chipsSF,
                        chipSize=512,
                        cellSize=2,
                        doJitter=TRUE,
                        jitterSD=15,
                        useSeed=TRUE,
                        seed=42,
                        normalize = FALSE,
                        rescaleFactor = 1,
                        mskRescale=1,
                        mskAdd=0,
                        bands = c(1,2,3),
                        bMns=1,
                        bSDs=1,
                        doAugs = FALSE,
                        maxAugs = 0,
                        probVFlip = 0,
                        probHFlip = 0,
                        probRotate = 0,
                        probBrightness = 0,
                        probContrast = 0,
                        probGamma = 0,
                        probHue = 0,
                        probSaturation = 0,
                        brightFactor = c(.8,1.2),
                        contrastFactor = c(.8,1.2),
                        gammaFactor = c(.8, 1.2, 1),
                        hueFactor = c(-.2, .2),
                        saturationFactor = c(.8, 1.2)){
    self$chipsSF <- chipsSF
    self$chipSize <- chipSize
    self$cellSize <- cellSize
    self$doJitter <- doJitter
    self$jitterSD <- jitterSD
    self$useSeed <- useSeed
    self$seed <- seed
    self$normalize <- normalize
    self$rescaleFactor <- rescaleFactor
    self$mskRescale <- mskRescale
    self$mskAdd <- mskAdd
    self$bands <- bands
    self$bMns <- bMns
    self$bSDs <- bSDs
    self$doAugs <- doAugs
    self$maxAugs <- maxAugs
    self$probVFlip <- probVFlip
    self$probHFlip <- probHFlip
    self$probRotate <- probRotate
    self$probBrightness <- probBrightness
    self$probContrast <- probContrast
    self$probGamma <- probGamma
    self$probHue <- probHue
    self$probSaturation <- probSaturation
    self$brightFactor <- brightFactor
    self$contrastFactor <- contrastFactor
    self$gammaFactor <- gammaFactor
    self$hueFactor <- hueFactor
    self$saturationFactor <- saturationFactor
  },

  .getitem = function(i){

    chipData <- makeDynamicChip(chipIn = self$chipsSF[i,],
                                chipSize = self$chipSize,
                                cellSize = self$cellSize,
                                doJitter = self$doJitter,
                                jitterSD = self$jitterSD,
                                useSeed = self$useSeed,
                                seed = self$seed)

    image <- chipData$image
    mask <- chipData$mask

    image <- terra::subset(image, self$bands)

    image <- terra::as.array(image)
    mask <- terra::as.array(mask)/self$mskRescale
    mask <- mask+self$mskAdd

    image <- torch::torch_tensor(image, dtype=torch::torch_float32())
    image <- image$permute(c(3,1,2))

    mask <- torch::torch_tensor(mask, dtype=torch::torch_long())

    mask <- mask$permute(c(3,1,2))

    if(self$normalize == TRUE){
      image <- torchvision::transform_normalize(image, self$bMns, self$bSDs, inplace = FALSE)
    }

    image <- torch::torch_div(image,self$rescaleFactor)

    if(self$doAugs == TRUE){

      probVFlipX <- runif(1) < self$probVFlip
      probHFlipX <- runif(1) < self$probHFlip
      probRotateX <- runif(1) < self$probRotate
      probBrightnessX <- runif(1) < self$probBrightness
      probContrastX <- runif(1) < self$probContrast
      probGammaX <- runif(1) < self$probGamma
      probHueX <- runif(1) < self$probHue
      probSaturationX <- runif(1) < self$probSaturation


      augIndex <- sample(c(1:8), self$maxAugs, replace=FALSE)


      if(probVFlipX == TRUE & 1 %in% augIndex){
        image <- torchvision::transform_vflip(image)
        mask <- torchvision::transform_vflip(mask)
      }

      if(probHFlipX == TRUE & 2 %in% augIndex){
        image <- torchvision::transform_hflip(image)
        mask <- torchvision::transform_hflip(mask)
      }

      if(probBrightnessX == TRUE & 3 %in% augIndex){
        brightFactor = runif(1, self$brightFactor[1], self$brightFactor[2])
        image <- torchvision::transform_adjust_brightness(image, brightness_factor=brightFactor)
      }

      if(probContrastX == TRUE & 4 %in% augIndex){
        contrastFactor = runif(1, self$contrastFactor[1], self$contrastFactor[2])
        image <- torchvision::transform_adjust_contrast(image, contrast_factor=contrastFactor)
      }

      if(probGammaX == TRUE & 5 %in% augIndex){
        gainIn = self$gammaFactor[3]
        gammaFactor = runif(1, self$gammaFactor[1], self$gammaFactor[2])
        image <- torchvision::transform_adjust_gamma(image, gamma=gammaFactor, gain=gainIn)
      }

      if(probHueX == TRUE & 6 %in% augIndex){
        hueFactor = runif(1, self$hueFactor[1], self$hueFactor[2])
        image <-torchvision::transform_adjust_hue(image, hue_factor=hueFactor)
      }

      if(probSaturationX == TRUE & 7 %in% augIndex){
        saturationFactor = runif(1, self$saturationFactor[1], self$saturationFactor[2])
        image <- torchvision::transform_adjust_saturation(image, saturation_factor=saturationFactor)
      }

      if(probRotateX == TRUE & 8 %in% augIndex){
        selectedAngle <- dplyr::sample_n(data.frame(angles = c(0, 90, 180, 270)), 1, replace=FALSE)[1,1]
        image <- torchvision::transform_rotate(image, angle=selectedAngle)
        mask <- torchvision::transform_rotate(mask, angle=selectedAngle)
      }
    }
    return(list(image = image, mask = mask))
  },

  .length = function(){
    return(nrow(self$chipsSF))
  }
)
