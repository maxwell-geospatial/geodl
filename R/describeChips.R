#' describeChips
#'
#' Generate data frame of band summary statistics and class pixel counts
#'
#' This function generates a set of summary metrics from image chips and
#' associated masks stored in a directory. For each band, the minimum, median, mean,
#' maximum, and standard deviation are returned (along with some other metrics). For
#' mask data, the count of pixels in each class are returned. These summarizations
#' can be useful for data normalization and determining class weightings in loss
#' calculations.
#'
#' @param folder Full folder path or folder path relative to the current working
#' directory that holds the image chips and associated masks. You must include
#' the final forward slash in the folder path (e.g., "C:/data/chips/").
#' @param extension Raster file extension (e.g., ".tif", ".png", ".jpeg", or ".img").
#' The utilities in this package generate files in ".tif" format, so this is the default.
#' This option is provided if chips are generated using another method.
#' @param mode Either "All", "Positive", or "Divided". This should match the settings
#' used in the makeChips() function or be set to "All" if makeChipsMultiClass() is
#' used. Default is "All".
#' @param subSample TRUE or FALSE. Whether or not to subsample the image chips to
#' calculate the summary metrics. We recommend using a subset if a large set of
#' chips are being summarized to reduce computational load. The default is TRUE.
#' @param numChips If subSample is set to TRUE, this parameter defines the
#' number of chips to subset. The default is 200. This parameter will be ignored
#' if subSample is set to FALSE.
#' @param numChipsBack If subSample is set to TRUE and the mode is "Divided", this
#' parameter indicates the number of chips to sample from the background-only
#' samples. The default is 200. This parameter will be ignored if subSample is
#' set to FALSE and/or mode is not "Divided".
#' @param subSamplePix TRUE or FALSE. Whether or not to calculate statistics using
#' a subsample of pixels from each image chip as opposed to all pixels. If a large
#' number of chips are available and/or each chip is large, we suggest setting this
#' argument to TRUE to reduce the computational load. The default is TRUE.
#' @param sampsPerChip If subSamplePix is TRUE, this parameters specifies the
#' number of random pixels to sample per chip. The default is 100. If
#' subSamplePix is set to FALSE, this parameter is ignored.
#' @return List object containing the summary metrics for each band in the
#' $ImageStats object and the count of pixels by class in the $maskStats object.
#' @examples
#' \dontrun{
#' chpDescript <- describeChips(folder= "PATH TO CHIPS FOLDER",
#'                              extension = ".tif",
#'                              mode = "Positive",
#'                              subSample = TRUE,
#'                              numChips = 100,
#'                              numChipsBack = 100,
#'                              subSamplePix = TRUE,
#'                              sampsPerChip = 400)
#' }
#' @export
describeChips <- function(folder,
                          extension = ".tif",
                          mode = "All",
                          subSample = TRUE,
                          numChips = 200,
                          numChipsBack = 200,
                          subSamplePix = TRUE,
                          sampsPerChip = 100){
  chipDF <- data.frame()
  mskStats <- data.frame()
  if(subSample == FALSE){
    if(subSamplePix == FALSE){
      if(mode == "All" | mode == "Positive"){
        lstChips <- list.files(paste0(folder, "images/"), pattern=paste0("\\", extension, "$"))
        lstMsk <- list.files(paste0(folder, "masks/"), pattern=paste0("\\", extension, "$"))
        for(chip in lstChips){
          chipIn <- terra::rast(paste0(folder, "images/", chip))
          chipInDF <- data.frame(chipIn)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }
        imgStats <- psych::describe(chipDF)
        for(msk in lstMsk){
          mskIn <- terra::rast(paste0(folder, "masks/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        mskStats2 <- mskStats |> dplyr::group_by(value) |> dplyr::summarize(cnt = sum(count))

      }else{
        lstChipsB <- list.files(paste0(folder, "images/background/"), pattern=paste0("\\", extension, "$"))
        lstChipsP <- list.files(paste0(folder, "images/positive/"), pattern=paste0("\\", extension, "$"))
        lstMskB <- list.files(paste0(folder, "masks/background/"), pattern=paste0("\\", extension, "$"))
        lstMskP <- list.files(paste0(folder, "masks/positive/"), pattern=paste0("\\", extension, "$"))
        for(chip in lstChipsB){
          chipIn <- terra::rast(paste0(folder, "images/background/", chip))
          chipInDF <- data.frame(chipIn)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }
        for(chip in lstChipsP){
          chipIn <- terra::rast(paste0(folder, "images/positive/", chip))
          chipInDF <- data.frame(chipIn)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }

        imgStats <- psych::describe(chipDF)
        for(msk in lstMskB){
          mskIn <- terra::rast(paste0(folder, "masks/background/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        for(msk in lstMskP){
          mskIn <- terra::rast(paste0(folder, "masks/positive/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        mskStats2 <- mskStats |> dplyr::group_by(value) |> dplyr::summarize(cnt = sum(count))
      }
    }else{
      if(mode == "All" | mode == "Positive"){
        lstChips <- list.files(paste0(folder, "images/"), pattern=paste0("\\", extension, "$"))
        lstMsk <- list.files(paste0(folder, "masks/"), pattern=paste0("\\", extension, "$"))
        for(chip in lstChips){
          chipIn <- terra::rast(paste0(folder, "images/", chip))
          chipInDF <- data.frame(chipIn)
          chipInDF <- chipInDF |> dplyr::sample_n(sampsPerChip)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }
        imgStats <- psych::describe(chipDF)
        for(msk in lstMsk){
          mskIn <- terra::rast(paste0(folder, "masks/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        mskStats2 <- mskStats |> dplyr::group_by(value) |> dplyr::summarize(cnt = sum(count))}
      else{
        lstChipsB <- list.files(paste0(folder, "images/background/"), pattern=paste0("\\", extension, "$"))
        lstChipsP <- list.files(paste0(folder, "images/positive/"), pattern=paste0("\\", extension, "$"))
        lstMskB <- list.files(paste0(folder, "masks/background/"), pattern=paste0("\\", extension, "$"))
        lstMskP <- list.files(paste0(folder, "masks/positive/"), pattern=paste0("\\", extension, "$"))
        for(chip in lstChipsB){
          chipIn <- terra::rast(paste0(folder, "images/background/", chip))
          chipInDF <- data.frame(chipIn)
          chipInDF <- chipInDF |> dplyr::sample_n(sampsPerChip)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }
        for(chip in lstChipsP){
          chipIn <- terra::rast(paste0(folder, "images/positive/", chip))
          chipInDF <- data.frame(chipIn)
          chipInDF <- chipInDF |> dplyr::sample_n(sampsPerChip)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }

        imgStats <- psych::describe(chipDF)
        for(msk in lstMskB){
          mskIn <- terra::rast(paste0(folder, "masks/background/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        for(msk in lstMskP){
          mskIn <- terra::rast(paste0(folder, "masks/positive/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        mskStats2 <- mskStats |> dplyr::group_by(value) |> dplyr::summarize(cnt = sum(count))
      }
    }
  }else{
    if(subSamplePix == FALSE){
      if(mode == "All" | mode == "Positive"){
        lstChips <- list.files(paste0(folder, "images/"), pattern=paste0("\\", extension, "$"))
        lstMsk <- list.files(paste0(folder, "masks/"), pattern=paste0("\\", extension, "$"))
        samps <- sample(seq(1, length(lstChips), 1), numChips)
        lstChips <- lstChips[c(samps)]
        lstMsk <- lstMsk[c(samps)]
        for(chip in lstChips){
          chipIn <- terra::rast(paste0(folder, "images/", chip))
          chipInDF <- data.frame(chipIn)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }
        imgStats <- psych::describe(chipDF)
        for(msk in lstMsk){
          mskIn <- terra::rast(paste0(folder, "masks/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        mskStats2 <- mskStats |> dplyr::group_by(value) |> dplyr::summarize(cnt = sum(count))
      }else{
        lstChipsB <- list.files(paste0(folder, "images/background/"), pattern=paste0("\\", extension, "$"))
        lstChipsP <- list.files(paste0(folder, "images/positive/"), pattern=paste0("\\", extension, "$"))
        lstMskB <- list.files(paste0(folder, "masks/background/"), pattern=paste0("\\", extension, "$"))
        lstMskP <- list.files(paste0(folder, "masks/positive/"), pattern=paste0("\\", extension, "$"))
        sampsB <- sample(seq(1, length(lstChipsB), 1), numChipsBack)
        sampsP <- sample(seq(1, length(lstChipsP), 1), numChips)
        lstChipsB <- lstChipsB[c(sampsB)]
        lstMskB <- lstMskB[c(sampsB)]
        lstChipsP <- lstChipsP[c(sampsP)]
        lstMskP <- lstMskP[c(sampsP)]
        for(chip in lstChipsB){
          chipIn <- terra::rast(paste0(folder, "images/background/", chip))
          chipInDF <- data.frame(chipIn)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }
        for(chip in lstChipsP){
          chipIn <- terra::rast(paste0(folder, "images/positive/", chip))
          chipInDF <- data.frame(chipIn)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }

        imgStats <- psych::describe(chipDF)
        for(msk in lstMskB){
          mskIn <- terra::rast(paste0(folder, "masks/background/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        for(msk in lstMskP){
          mskIn <- terra::rast(paste0(folder, "masks/positive/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        mskStats2 <- mskStats |> dplyr::group_by(value) |> dplyr::summarize(cnt = sum(count))
      }
    }else{
      if(mode == "All" | mode == "Positive"){
        lstChips <- list.files(paste0(folder, "images/"), pattern=paste0("\\", extension, "$"))
        lstMsk <- list.files(paste0(folder, "masks/"), pattern=paste0("\\", extension, "$"))
        samps <- sample(seq(1, length(lstChips), 1), numChips)
        lstChips <- lstChips[c(samps)]
        lstMsk <- lstMsk[c(samps)]
        for(chip in lstChips){
          chipIn <- terra::rast(paste0(folder, "images/", chip))
          chipInDF <- data.frame(chipIn)
          chipInDF <- chipInDF |> dplyr::sample_n(sampsPerChip)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }
        imgStats <- psych::describe(chipDF)
        for(msk in lstMsk){
          mskIn <- terra::rast(paste0(folder, "masks/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        mskStats2 <- mskStats |> dplyr::group_by(value) |> dplyr::summarize(cnt = sum(count))
      }else{
        lstChipsB <- list.files(paste0(folder, "images/background/"), pattern=paste0("\\", extension, "$"))
        lstChipsP <- list.files(paste0(folder, "images/positive/"), pattern=paste0("\\", extension, "$"))
        lstMskB <- list.files(paste0(folder, "masks/background/"), pattern=paste0("\\", extension, "$"))
        lstMskP <- list.files(paste0(folder, "masks/positive/"), pattern=paste0("\\", extension, "$"))
        sampsB <- sample(seq(1, length(lstChipsB), 1), numChipsBack)
        sampsP <- sample(seq(1, length(lstChipsP), 1), numChips)
        lstChipsB <- lstChipsB[c(sampsB)]
        lstMskB <- lstMskB[c(sampsB)]
        lstChipsP <- lstChipsP[c(sampsP)]
        lstMskP <- lstMskP[c(sampsP)]
        for(chip in lstChipsB){
          chipIn <- terra::rast(paste0(folder, "images/background/", chip))
          chipInDF <- data.frame(chipIn)
          chipInDF <- chipInDF |> dplyr::sample_n(sampsPerChip)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }
        for(chip in lstChipsP){
          chipIn <- terra::rast(paste0(folder, "images/positive/", chip))
          chipInDF <- data.frame(chipIn)
          chipInDF <- chipInDF |> dplyr::sample_n(sampsPerChip)
          nCols <- ncol(chipInDF)
          colNames <- paste0("B", seq(1,nCols))
          names(chipInDF) <- colNames
          chipDF <- dplyr::bind_rows(chipDF, chipInDF)
        }

        imgStats <- psych::describe(chipDF)
        for(msk in lstMskB){
          mskIn <- terra::rast(paste0(folder, "masks/background/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        for(msk in lstMskP){
          mskIn <- terra::rast(paste0(folder, "masks/positive/", msk))
          mskInDF <- terra::freq(mskIn)
          mskStats <- dplyr::bind_rows(mskStats, mskInDF)
        }
        mskStats2 <- mskStats |> dplyr::group_by(value) |> dplyr::summarize(cnt = sum(count))
      }
    }
  }
  outStats <- list(ImageStats = imgStats,
                   mskStats = mskStats2)
  return(outStats)
}
