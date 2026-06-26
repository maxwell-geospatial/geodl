#' makeDynamicChipsSF
#'
#' Generate a vector point sf object of chip center locations for use with defineDynamicSegDataSet().
#'
#' Requires an input vector point object defining chip center locations, an input vector polygon object
#' of reference features used to generate segmentation masks, a boundary extent, and the path and name
#' of the associated raster predictor variables. Note that the raster predictor variable must cover the
#' full spatial extent defined by the extent object. If not, a smaller extent should be defined for
#' which predictor variables are available.
#'
#' The input point layer must contain a "code" column of numeric class codes and a "class" column of
#' text class names. The input polygon feature layer is used only as the mask source during training
#' and is not required to share attributes with the point layer, but must also contain "code" and
#' "class" columns if class-specific masking is expected downstream.
#'
#' The extent can be cropped to avoid generating chips without a full set of cells available. We recommend
#' cropping by a factor larger than 50% the width/height of the desired chip size.
#'
#' The tool returns a sf dataframe with class name ("class"), class numeric code ("code"), path to the
#' image or predictor variables ("imgPth"), name of the image or predictor variables ("imgName"), the
#' path to the input point layer ("pntPth"), the file name of the input point layer ("pntName"), the
#' path to the input polygon features ("featPth"), the file name of the input polygon features
#' ("featName"), the file path to the extent ("extentPth"), the file name for the extent ("extentName"),
#' and the point geometry ("geometry").
#' @param pntPth Full folder path or relative path to vector point layer defining chip centers. Must
#' include final forward slash.
#' @param pntName File name of point layer including file extension. Must contain "code" (numeric class
#' code) and "class" (text class name) columns.
#' @param featPth Full folder path or relative path to vector polygon features used as the mask source.
#' Must include final forward slash.
#' @param featName File name of polygon feature layer including file extension.
#' @param extentPth Full folder path or relative path to vector extent. Must include final forward slash.
#' @param extentName File name of extent including file extension.
#' @param extentCrop Amount to crop extent to avoid edge effects. We recommend cropping
#' the extent more than the size of the chips you will use to avoid chips with NA cells. Default is 50 cells.
#' @param imgPth Path to image or predictor variable raster data. Must include the final forward slash.
#' @param imgName File name of image or predictor variable raster data including file extension.
#' @param doBackground Whether or not to randomly select background cells. Default is FALSE.
#' @param backgroundCnt Number of background cells to select.
#' @param backgroundDist Minimum allowed distance between randomly selected background locations and
#' boundaries of reference features.
#' @param useSeed Whether or not to set a random seed for replicability. Default is FALSE.
#' @param seed Random seed value.
#' @param doShuffle Whether or not to shuffle the rows in the resulting table. Default is FALSE.
#' @importFrom graphics layout par
#' @importFrom stats rnorm
#' @return sf dataframe object of center locations for chip creation with associated information as attributes.
#' @export
makeDynamicChipsSF <- function(pntPth,
                               pntName,
                               featPth,
                               featName,
                               extentPth,
                               extentName,
                               extentCrop = 50,
                               imgPth,
                               imgName,
                               doBackground = FALSE,
                               backgroundCnt,
                               backgroundDist,
                               useSeed = FALSE,
                               seed,
                               doShuffle = FALSE){

  inPnt  <- sf::st_read(paste0(pntPth, pntName), quiet=TRUE) |>
    dplyr::select(code, class) |>
    sf::st_cast("POINT")

  extent <- sf::st_read(paste0(extentPth, extentName), quiet=TRUE)

  if(extentCrop != 0){
    bndB <- sf::st_buffer(extent, extentCrop)
  }else{
    bndB <- extent
  }

  # Clip input points to the (optionally cropped) study extent
  vC <- sf::st_intersection(inPnt, bndB)

  if(doBackground == TRUE){
    if(useSeed == TRUE){
      set.seed(seed)
    }
    rndPts <- sf::st_sample(bndB, size = backgroundCnt*4) |> sf::st_cast("POINT") |> sf::st_sf()
    sf::st_geometry(rndPts) <- "geometry"
    rndPts <- sf::st_filter(rndPts, sf::st_buffer(bndB, backgroundDist), .predicate=sf::st_within, inverse=TRUE)
    rndPts <- rndPts |> dplyr::sample_n(backgroundCnt)
    rndPts <- rndPts |> dplyr::mutate(class = "Background", code = 0)
    vC <- dplyr::bind_rows(vC, rndPts)
  }

  vC <- vC |>
    dplyr::mutate(imgPth     = imgPth,
                  imgName    = imgName,
                  pntPth     = pntPth,
                  pntName    = pntName,
                  featPth    = featPth,
                  featName   = featName,
                  extentPth  = extentPth,
                  extentName = extentName) |>
    dplyr::select(class,
                  code,
                  imgPth,
                  imgName,
                  pntPth,
                  pntName,
                  featPth,
                  featName,
                  extentPth,
                  extentName,
                  geometry)

  if(doShuffle == TRUE){
    vC <- vC |> dplyr::sample_frac(1, replace=FALSE)
  }

  vC <- vC |>
    dplyr::mutate(chipID = dplyr::row_number()) |>
    dplyr::select(chipID, dplyr::everything())

  return(vC)
}
