#' makeTerrainDerivatives
#'
#' Make three band terrain stack from input digital terrain model
#'
#' This function creates a three-band raster stack from an input digital terrain
#' model (DTM) of bare earth surface elevations. The first band is a topographic
#' position index (TPI) calculated using a moving window with a 50 m circular radius.
#' The second band is the square root of slope calculated in degrees. The third band
#' is a TPI calculated using an annulus moving window with an inner radius of 2
#' and outer radius of 5 meters. The TPI values are clamped to a range of -10 to 10
#' then linearly rescaled from 0 and 1. The square root of slope is clamped to a
#' range of 0 to 10 then linearly rescaled from 0 to 1. Values are provided in
#' floating point.
#'
#' The stack is described in the following publication and was originally proposed by
#' William Odom of the United States Geological Survey (USGS):
#'
#' Maxwell, A.E., W.E. Odom, C.M. Shobe, D.H. Doctor, M.S. Bester, and T. Ore,
#' 2023. Exploring the influence of input feature space on CNN-based geomorphic
#' feature extraction from digital terrain data, Earth and Space Science,
#' 10: e2023EA002845. https://doi.org/10.1029/2023EA002845.
#'
#' @param dtm Input SpatRaster object representing bare earth surface elevations.
#' @param res Resolution of the grid relative to coordinate reference system
#' units (e.g., meters).
#' @param filename Name and full path or path relative to working directory for
#' output terrain stack. We recommend saving the data to either TIFF (".tif") or
#' Image (".img") format.
#' @return Three-band raster grid written to disk in TIFF format and spatRaster object.
#' @export
makeTerrainDerivatives <- function(dtm,
                                   res,
                                   filename){
  slp <- terra::terrain(dtm, v = "slope", neighbors = 8, unit = "degrees")
  slp1 <- sqrt(slp)
  slp2 <- terra::clamp(slp1, 0, 10, values = TRUE)
  slp3 <- slp2/(10)
  print("Completed Slope.")
  fAnnulus <- MultiscaleDTM::annulus_window(radius = c(2,5), unit = "map", resolution = res)
  fCircle <- MultiscaleDTM::circle_window(radius = c(50), unit = "map", resolution = res)
  tpiC <- dtm - terra::focal(dtm, fCircle, "mean", na.rm = TRUE)
  tpiA <- dtm - terra::focal(dtm, fAnnulus, "mean", na.rm = TRUE)
  tpiC2 <- terra::clamp(tpiC, -10, 10, values = TRUE)
  tpiA2 <- terra::clamp(tpiA, -10, 10, values = TRUE)
  tpiC3 <- (tpiC2 - (-10))/(10-(-10))
  tpiA3 <- (tpiA2 - (-10))/(10-(-10))
  print("Completed TPIs.")
  stackOut <- c(tpiC3, slp3, tpiA3)
  names(stackOut) <- c("tpi1", "sqrtslp", "tpi2")
  terra::writeRaster(stackOut, filename, overwrite = TRUE)
  return(stackOut)
  print("Results stacked and written to disk.")
}
