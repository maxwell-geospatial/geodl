#' makeBoundaryDistance
#'
#' Generate distance-to-class-boundary grids from mask chips
#'
#' This function reads raster mask chips from the "masks" subfolder of a chips
#' directory created by \code{makeChips} and generates a corresponding distance
#' surface for each mask. Each output raster encodes, for every pixel, the
#' distance to the nearest class boundary. A class boundary is any pixel that
#' is adjacent (in an 8-connected neighborhood) to at least one pixel belonging
#' to a different class. Boundary pixels themselves receive a value of 0; all
#' other pixels receive the Euclidean distance (in pixel units, unless the
#' raster has a projected coordinate system, in which case map units are used)
#' to the nearest boundary pixel. Output grids are written to a "dist"
#' subfolder created inside the provided chips directory. Output files are
#' named identically to their corresponding masks and are saved in TIFF format.
#'
#' The function automatically detects whether the chips directory uses a flat
#' layout (\code{masks/}) produced by \code{makeChips} with \code{mode = "All"}
#' or \code{mode = "Positive"}, or a divided layout (\code{masks/positive/} and
#' \code{masks/background/}) produced with \code{mode = "Divided"}. For the
#' divided layout, distance grids are written to \code{dist/positive/} and
#' \code{dist/background/} respectively.
#'
#' @param chipsDir Full or relative path to the chips directory that contains
#' a "masks" subfolder. This is the same directory that was used as the
#' \code{outDir} argument to \code{makeChips}. Do not include a trailing slash.
#' @param noDataValue Numeric scalar. The value written to every pixel of a
#' distance grid when the corresponding mask chip contains no class boundaries
#' (i.e. the chip belongs to a single class throughout). Default is \code{0}.
#' @return Distance-to-boundary raster files written to disk in TIFF format
#' inside a "dist" subfolder (flat layout) or "dist/positive/" and
#' "dist/background/" subfolders (divided layout). No R object is returned.
#' @examples
#' \dontrun{
#' makeBoundaryDistance(chipsDir = "C:/data/chips")
#'
#' # Fill single-class chips with a large sentinel instead of 0
#' makeBoundaryDistance(chipsDir = "C:/data/chips", noDataValue = 999)
#' }
#' @export
makeBoundaryDistance <- function(chipsDir, noDataValue = 0) {
  masksDir <- file.path(chipsDir, "masks")

  if (!dir.exists(masksDir)) {
    stop("No 'masks' subfolder found in the provided directory.")
  }

  # Detect layout: divided mode has positive/ and/or background/ subfolders
  dividedSubs <- c("positive", "background")
  foundSubs   <- dividedSubs[dir.exists(file.path(masksDir, dividedSubs))]
  isDivided   <- length(foundSubs) > 0

  if (isDivided) {
    for (sub in foundSubs) {
      maskSubDir <- file.path(masksDir, sub)
      distSubDir <- file.path(chipsDir, "dist", sub)
      dir.create(distSubDir, recursive = TRUE, showWarnings = FALSE)

      maskFiles <- list.files(maskSubDir, pattern = "\\.tif$",
                              full.names = TRUE, ignore.case = TRUE)
      if (length(maskFiles) == 0) next

      for (mFile in maskFiles) {
        distRast <- .calcBoundaryDist(mFile, noDataValue)
        terra::writeRaster(distRast,
                           file.path(distSubDir, basename(mFile)),
                           overwrite = TRUE)
      }
    }
  } else {
    maskFiles <- list.files(masksDir, pattern = "\\.tif$",
                            full.names = TRUE, ignore.case = TRUE)

    if (length(maskFiles) == 0) {
      message("No TIF files found in the masks folder.")
      return(invisible(NULL))
    }

    distDir <- file.path(chipsDir, "dist")
    dir.create(distDir, showWarnings = FALSE)

    for (mFile in maskFiles) {
      distRast <- .calcBoundaryDist(mFile, noDataValue)
      terra::writeRaster(distRast,
                         file.path(distDir, basename(mFile)),
                         overwrite = TRUE)
    }
  }

  invisible(NULL)
}

# Internal helper: compute distance-to-boundary raster from a single mask file.
.calcBoundaryDist <- function(mFile, noDataValue = 0) {
  msk <- terra::rast(mFile)
  w <- matrix(1, 3, 3)
  neighborMin <- terra::focal(msk, w = w, fun = "min", na.rm = TRUE)
  neighborMax <- terra::focal(msk, w = w, fun = "max", na.rm = TRUE)
  isBoundary  <- terra::ifel(neighborMin != neighborMax, 1, NA)

  # If no boundary pixels exist (single-class chip), terra::distance() would
  # return an all-NaN raster. Return a constant fill instead.
  if (all(is.na(terra::values(isBoundary)))) {
    out <- msk
    terra::values(out) <- noDataValue
    return(out)
  }

  terra::distance(isBoundary)
}
