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

  chipDF   <- data.frame()
  mskStats <- data.frame()

  # Read one chip's pixel values into a band-named dataframe
  readChipPixels <- function(path) {
    chipIn <- terra::rast(path)
    df     <- data.frame(chipIn)
    if(subSamplePix) df <- df |> dplyr::sample_n(min(sampsPerChip, nrow(df)))
    names(df) <- paste0("B", seq_len(ncol(df)))
    df
  }

  # Subsample a file list, keeping mask list aligned
  subsamplePair <- function(chips, msks, n) {
    idx  <- sample(seq_len(length(chips)), min(n, length(chips)))
    list(chips=chips[idx], msks=msks[idx])
  }

  # Accumulate pixel stats from a list of chip paths
  addChipStats <- function(chipPaths) {
    for(chip in chipPaths){
      chipDF <<- dplyr::bind_rows(chipDF, readChipPixels(chip))
    }
  }

  # Accumulate frequency counts from a list of mask paths
  addMskStats <- function(mskPaths) {
    for(msk in mskPaths){
      mskStats <<- dplyr::bind_rows(mskStats, terra::freq(terra::rast(msk)))
    }
  }

  if(mode == "All" | mode == "Positive"){

    lstChips <- list.files(paste0(folder, "images/"), pattern=paste0("\\", extension, "$"), full.names=TRUE)
    lstMsk   <- list.files(paste0(folder, "masks/"),  pattern=paste0("\\", extension, "$"), full.names=TRUE)

    if(subSample){
      pair     <- subsamplePair(lstChips, lstMsk, numChips)
      lstChips <- pair$chips
      lstMsk   <- pair$msks
    }

    addChipStats(lstChips)
    addMskStats(lstMsk)

  }else{

    lstChipsB <- list.files(paste0(folder, "images/background/"), pattern=paste0("\\", extension, "$"), full.names=TRUE)
    lstChipsP <- list.files(paste0(folder, "images/positive/"),   pattern=paste0("\\", extension, "$"), full.names=TRUE)
    lstMskB   <- list.files(paste0(folder, "masks/background/"),  pattern=paste0("\\", extension, "$"), full.names=TRUE)
    lstMskP   <- list.files(paste0(folder, "masks/positive/"),    pattern=paste0("\\", extension, "$"), full.names=TRUE)

    if(subSample){
      pairB     <- subsamplePair(lstChipsB, lstMskB, numChipsBack)
      pairP     <- subsamplePair(lstChipsP, lstMskP, numChips)
      lstChipsB <- pairB$chips; lstMskB <- pairB$msks
      lstChipsP <- pairP$chips; lstMskP <- pairP$msks
    }

    addChipStats(lstChipsB)
    addChipStats(lstChipsP)
    addMskStats(lstMskB)
    addMskStats(lstMskP)
  }

  imgStats  <- psych::describe(chipDF)
  mskStats2 <- mskStats |> dplyr::group_by(value) |> dplyr::summarize(cnt=sum(count))

  return(list(ImageStats=imgStats, mskStats=mskStats2))
}
