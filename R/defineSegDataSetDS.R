#' defineSegDataSet
#'
#' Instantiate a subclass of torch dataset() function for semantic segmentation with rescaled masks for deep supervision
#'
#' This function instantiates a subclass of the torch dataset() function that loads
#' data generated using the makeChips() or makeChipsMultiClass() functions. Can also
#' define random augmentations to combat overfitting. Note that horizontal and vertical
#' flips will effect the alignment of the image and associated mask chips. As a result,
#' the same augmentation will be applied to both the image and the mask. Changes in
#' brightness, contrast, gamma, hue, and saturation will not be applied to the masks
#' since alignment is not impacted by these transformations. This version of the function
#' generates masks at the original, original/2, original/4, and original/8 spatial scales
#' to support deep supervision.
#'
#' @param chpDF Data frame of image chip paths created using makeChipsDF().
#' @param folder Full path or path relative to the working directory to the
#' folder containing the image chips and associated masks. You must include the
#' final forward slash in the path (e.g., "C:/data/chips/").
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
#' start are zero, 1 can be added so that indices start at 1. Default is 0 (return
#' original class codes).
#' @param bands Vector of bands to include. The default is to only include the
#' first 3 bands. If you want to use a different subset of bands, you must provide
#' a vector of band indices here to override the default.
#' @param bMns Vector of band means. Length should be the same as the number of bands.
#' Normalization is applied before any rescaling within the function.
#' @param bSDs Vector of band standard deviations. Length should be the same
#' as the number of bands. Normalization is applied before any rescaling within
#' the function.
#' @param chnDim TRUE or FALSE. Default is TRUE. If TRUE, will produce target tensors that include the
#' channel dimension: (N, C, H, W). If FALSE, will produce target tensors that do not include the
#' channel dimension: (C, H, W). We recommend including the target dimension since other functions
#' in this package expect this by default.
#' @param mskLong TRUE or FALSE. Default is TRUE. If TRUE, target tensors will be produced with a
#' tensor_long data type. If FALSE, target tensors will be produced with a tensor_float32 data type.
#' We recommend using TRUE or producing targets with a torch_long data type.
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
#' @param probSaturation 0 to 1. Probability of applying saturation augmentations.
#' Default is 0 or no augmentations. Must be changed if augmentations are desired.
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
#' is not randomly altered, only the gamma. Non negative real number. gamma larger
#' than 1 make the shadows darker, while gamma smaller than 1 make dark regions
#' lighter.
#' @param hueFactor Vector of smallest and largest hue adjustment factors.
#' Random value will be selected between these extremes. The default is -0.2 to 0.2.
#' Should be in (-0.5, 0.5). 0.5 and -0.5 give complete reversal of hue channel
#' in HSV space in positive and negative direction, respectively. 0 means no shift.
#' Therefore, both -0.5 and 0.5 will give an image with complementary colors while
#' 0 gives the original image.
#' @param saturationFactor Vector of smallest and largest saturation adjustment factors.
#' Random value will be selected between these extremes. The default is 0.8 to 1.2.
#' For example, 0 will give a black and white image, 1`will give the original image while 2 will
#' enhance the saturation by a factor of 2.
#' @return A dataset object that can be provided to torch::dataloader().
#' @export
defineSegDataSet <- torch::dataset(

  name = "segDataSet",

  initialize = function(chpDF,
                        folder,
                        normalize = FALSE,
                        rescaleFactor = 1,
                        mskRescale=1,
                        mskAdd=0,
                        bands = c(1,2,3),
                        bMns=1,
                        bSDs=1,
                        mskLong = TRUE,
                        chnDim = TRUE,
                        doAugs = FALSE,
                        maxAugs = 0,
                        probVFlip = 0,
                        probHFlip = 0,
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
    self$chpDF <- chpDF
    self$folder <- folder
    self$normalize <- normalize
    self$rescaleFactor <- rescaleFactor
    self$mskRescale <- mskRescale
    self$mskAdd <- mskAdd
    self$bands <- bands
    self$bMns <- bMns
    self$bSDs <- bSDs
    self$mskLong <- mskLong
    self$chnDim <- chnDim
    self$doAugs <- doAugs
    self$maxAugs <- maxAugs
    self$probVFlip <- probVFlip
    self$probHFlip <- probHFlip
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
    image_name <- unlist(self$chpDF[i, "chpPth"], use.names=FALSE)
    mask_name <- unlist(self$chpDF[i, "mskPth"], use.names=FALSE)
    image <- terra::rast(paste0(self$folder, image_name))
    mask <- terra::rast(paste0(self$folder, mask_name))

    image <- terra::subset(image, self$bands)

    image <- terra::as.array(image)
    mask <- terra::as.array(mask)/self$mskRescale
    mask <- mask+self$mskAdd

    image <- torch::torch_tensor(image, dtype=torch_float32())
    image <- image$permute(c(3,1,2))
    if(self$mskLong == TRUE){
      mask <- torch::torch_tensor(mask, dtype=torch_long())
    }else{
      mask <- torch::torch_tensor(mask, dtype=torch_float32())
    }
    mask <- mask$permute(c(3,1,2))

    if(self$normalize == TRUE){
      image <- torchvision::transform_normalize(image, self$bMns, self$bSDs, inplace = FALSE)
    }

    image <- torch::torch_div(image,self$rescaleFactor)

    if(self$chnDim == FALSE){
      mask <- mask$squeeze()
    }


    if(self$doAugs == TRUE){

      probVFlipX <- runif(1) < self$probVFlip
      probHFlipX <- runif(1) < self$probHFlip
      probBrightnessX <- runif(1) < self$probBrightness
      probContrastX <- runif(1) < self$probContrast
      probGammaX <- runif(1) < self$probGamma
      probHueX <- runif(1) < self$probHue
      probSaturationX <- runif(1) < self$probSaturation

      augIndex <- sample(c(1:7), self$maxAugs, replace=FALSE)


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
    }
    mask2 <- terra::aggregate(mask, fact=2, fun="modal")
    mask4 <- terra::aggregate(mask, fact=4, fun="modal")
    mask8 <- terra::aggregate(mask, fact=8, fun="modal")
    return(list(image = image, list(mask1 = mask, mask2=mask2, mask4=mask4, mask8=mask8)))
  },

  .length = function(){
    return(nrow(self$chpDF))
  }
)
