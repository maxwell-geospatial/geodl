#' makeChipsMultiClass
#'
#' Deprecated. Use \code{\link{makeChips}} with \code{mode = "Positive"} instead.
#'
#' This function is a deprecated wrapper around \code{\link{makeChips}}. It
#' will be removed in a future version. Replace calls of the form:
#'
#' \code{makeChipsMultiClass(image, mask, ...)}
#'
#' with:
#'
#' \code{makeChips(image, mask, ..., mode = "Positive")}
#'
#' @param image SpatRaster object or path to input image.
#' @param mask SpatRaster object or path to single-band mask.
#' @param n_channels Number of channels in the input image. Default is 3.
#' @param size Size of image chips in pixels. Default is 256.
#' @param stride_x Stride in the x (columns) direction. Default is 256.
#' @param stride_y Stride in the y (rows) direction. Default is 256.
#' @param outDir Full or relative path to the output directory. You must include
#' the final forward slash (e.g., "C:/data/chips/").
#' @param useExistingDir TRUE or FALSE. Write into an existing directory.
#' Default is FALSE.
#' @return Image and mask files written to disk in TIFF format. No R object is returned.
#' @examples
#' \dontrun{
#' # Deprecated — use makeChips() instead:
#' makeChips(image = "INPUT IMAGE NAME AND PATH",
#'           mask = "INPUT RASTER MASK NAME AND PATH",
#'           n_channels = 3,
#'           size = 512,
#'           stride_x = 512,
#'           stride_y = 512,
#'           outDir = "DIRECTORY IN WHICH TO SAVE CHIPS",
#'           mode = "Positive",
#'           useExistingDir = FALSE)
#' }
#' @export
makeChipsMultiClass <- function(image,
                                mask,
                                n_channels = 3,
                                size = 256,
                                stride_x = 256,
                                stride_y = 256,
                                outDir,
                                useExistingDir = FALSE){
  .Deprecated("makeChips",
              msg = paste0("makeChipsMultiClass() is deprecated and will be removed ",
                           "in a future version. Use makeChips(..., mode = \"Positive\") instead."))
  makeChips(image        = image,
            mask         = mask,
            n_channels   = n_channels,
            size         = size,
            stride_x     = stride_x,
            stride_y     = stride_y,
            outDir       = outDir,
            mode         = "Positive",
            useExistingDir = useExistingDir)
}
