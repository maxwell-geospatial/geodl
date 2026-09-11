#' viewDist
#'
#' Visualise a mini-batch of image chips, masks, distance-to-boundary grids,
#' and the resulting edge-weight surface for a DataLoader that supplies distance grids.
#'
#' Plots four panels stacked vertically:
#' \enumerate{
#'   \item RGB image chips arranged in a grid.
#'   \item Corresponding class masks.
#'   \item Distance-to-class-boundary grids produced by \code{makeBoundaryDistance}.
#'   \item The per-pixel edge-weight surface derived from those distances, computed
#'         with the same formula used by \code{defineUnifiedFocalLoss} when
#'         \code{useEdgeDist = TRUE}.
#' }
#'
#' The weight surface mirrors the three modes available in \code{defineUnifiedFocalLoss}:
#' \describe{
#'   \item{\code{"threshold"}}{Pixels within \code{distThreshold} receive one weight
#'         (\code{maxWght} when \code{prioritizeEdge = TRUE}); all others receive the
#'         opposite weight.}
#'   \item{\code{"linear"}}{Weight decreases linearly from \code{maxWght} at distance 0
#'         to \code{minWght} at \code{distThreshold} (when \code{prioritizeEdge = TRUE}),
#'         clamped beyond the threshold.}
#'   \item{\code{"gaussian"}}{Weight follows a Gaussian curve centred at distance 0 with
#'         sigma = distThreshold / 3, so the weight equals approximately \code{minWght}
#'         at the threshold and is clamped to \code{minWght} beyond it (when
#'         \code{prioritizeEdge = TRUE}).}
#' }
#'
#' The DataLoader must return batches with three named elements: \code{image},
#' \code{mask}, and \code{dist}. The \code{dist} element should be a tensor of
#' shape (N, 1, H, W) containing the per-pixel distances produced by
#' \code{makeBoundaryDistance}.
#'
#' @param dataLoader Instantiated DataLoader created with \code{torch::dataloader()}.
#'   The batch must contain named elements \code{image}, \code{mask}, and \code{dist}.
#' @param nCols Number of chip columns in each image grid. Default is 3.
#' @param r Index of the image channel mapped to the red display channel. Default is 1.
#'   For single-band imagery, assign the same index to \code{r}, \code{g}, and \code{b}.
#' @param g Index of the image channel mapped to the green display channel. Default is 2.
#' @param b Index of the image channel mapped to the blue display channel. Default is 3.
#' @param cCodes Integer class codes as a vector, one per class.
#' @param cNames Character vector of class names, in the same order as \code{cCodes}.
#' @param cColors Character vector of display colours for the mask panel, in the same
#'   order as \code{cCodes}.
#' @param edgeMode Character string specifying how distances are converted to weights.
#'   One of \code{"threshold"}, \code{"linear"}, or \code{"gaussian"} (default).
#'   Must match the \code{edgeMode} used in \code{defineUnifiedFocalLoss}.
#' @param distThreshold Positive numeric. The distance value (in the same units as the
#'   distance raster) that defines the boundary of the high-weight zone. Default is 10.
#' @param minWght Numeric. Minimum per-pixel weight. Must be less than \code{maxWght}.
#'   Default is 0.5.
#' @param maxWght Numeric. Maximum per-pixel weight. Must be greater than \code{minWght}.
#'   Default is 1.0.
#' @param prioritizeEdge Logical. When \code{TRUE} (default), pixels near class boundaries
#'   (small distance) receive \code{maxWght} and pixels far from boundaries receive
#'   \code{minWght}. When \code{FALSE}, the relationship is reversed.
#' @param padding Integer. Number of pixels of padding between chips in the grid.
#'   Default is 10.
#' @return Called for its side effect (plots to the active graphics device).
#'   Returns \code{invisible(NULL)}.
#' @examples
#' \dontrun{
#' viewDist(
#'   dataLoader    = trainDL,
#'   nCols         = 4,
#'   r = 1, g = 2, b = 3,
#'   cCodes        = c(1, 2, 3, 4, 5),
#'   cNames        = c("Background", "Forest", "Water", "Urban", "Agriculture"),
#'   cColors       = c("gray", "darkgreen", "blue", "red", "yellow"),
#'   edgeMode      = "gaussian",
#'   distThreshold = 10,
#'   minWght       = 0.4,
#'   maxWght       = 1.0,
#'   prioritizeEdge = TRUE
#' )
#' }
#' @export
viewDist <- function(dataLoader,
                     nCols          = 3,
                     r              = 1,
                     g              = 2,
                     b              = 3,
                     cCodes,
                     cNames,
                     cColors,
                     edgeMode       = "gaussian",
                     distThreshold  = 10,
                     minWght        = 0.5,
                     maxWght        = 1.0,
                     prioritizeEdge = TRUE,
                     padding        = 10) {

  edgeMode <- match.arg(edgeMode, c("threshold", "linear", "gaussian"))

  if (minWght >= maxWght)
    stop("minWght must be less than maxWght.")
  if (distThreshold <= 0)
    stop("distThreshold must be greater than 0.")

  batch1 <- dataLoader$.iter()$.next()

  images <- batch1$image
  masks  <- batch1$mask
  dists  <- batch1$dist

  if (is.null(dists)) {
    stop(
      "The dataloader batch does not contain a 'dist' element. ",
      "Update your dataset class to return distance grids (produced by ",
      "makeBoundaryDistance) as a 'dist' element alongside 'image' and 'mask'."
    )
  }

  masks <- torch::torch_tensor(masks, dtype = torch::torch_float32())
  dists <- torch::torch_tensor(dists, dtype = torch::torch_float32())

  # Build padded chip grids (pad_value = -1 marks inter-chip padding)
  theImgGrid  <- torchvision::vision_make_grid(
    images, num_rows = nCols, padding = padding
  )$permute(c(2L, 3L, 1L))$to(device = "cpu")

  theMskGrid  <- torchvision::vision_make_grid(
    masks, num_rows = nCols, padding = padding, pad_value = -1, scale = FALSE
  )$permute(c(2L, 3L, 1L))$to(device = "cpu")

  theDistGrid <- torchvision::vision_make_grid(
    dists, num_rows = nCols, padding = padding, pad_value = -1, scale = FALSE
  )$permute(c(2L, 3L, 1L))$to(device = "cpu")

  img1  <- terra::rast(as.array(theImgGrid) * 255)
  msk1  <- terra::rast(as.array(theMskGrid))
  dist1 <- terra::rast(as.array(theDistGrid))

  msk1  <- terra::subst(msk1,  -1, NA)
  dist1 <- terra::subst(dist1, -1, NA)

  # ------------------------------------------------------------------
  # Compute the edge-weight surface in R, mirroring defineUnifiedFocalLoss
  # distVals: (n_pixels x 1) matrix; NA where padding was applied
  # ------------------------------------------------------------------
  distVals <- terra::values(dist1)

  if (edgeMode == "threshold") {
    inside <- distVals <= distThreshold
    if (prioritizeEdge) {
      wVals <- ifelse(inside, maxWght, minWght)
    } else {
      wVals <- ifelse(inside, minWght, maxWght)
    }

  } else if (edgeMode == "linear") {
    t <- pmin(pmax(distVals / distThreshold, 0), 1)
    if (prioritizeEdge) {
      wVals <- maxWght * (1 - t) + minWght * t
    } else {
      wVals <- minWght * (1 - t) + maxWght * t
    }

  } else {
    # gaussian: sigma = distThreshold / 3, clamped to minWght beyond threshold
    # (matches the loss: beyondT pixels are hard-clamped to gaussFactor = 0)
    sigma       <- distThreshold / 3
    gaussFactor <- exp(-(distVals^2) / (2 * sigma^2))
    gaussFactor[!is.na(distVals) & distVals > distThreshold] <- 0
    if (prioritizeEdge) {
      wVals <- minWght + (maxWght - minWght) * gaussFactor
    } else {
      wVals <- maxWght - (maxWght - minWght) * gaussFactor
    }
  }

  # Restore NA at padding pixels (overrides any accidental fill above)
  wVals[is.na(distVals)] <- NA

  wgt1 <- dist1
  terra::values(wgt1) <- wVals

  # Class colour table — filter to codes present in this batch
  catData <- data.frame(cCodes = cCodes, cNames = cNames, cColors = cColors)
  used    <- terra::unique(msk1) |> as.vector() |> unlist()
  catData <- catData |> dplyr::filter(cCodes %in% used)

  # Colour palettes:
  #   distance: dark blue at dist=0 (boundary), fading to white at far pixels
  #   weight:   yellow at low weight, orange/red at high weight
  distCols <- rev(grDevices::hcl.colors(100, "Blues"))
  wgtCols  <- grDevices::hcl.colors(100, "YlOrRd")

  wgtTitle <- paste0(
    "Edge weights  [", edgeMode,
    " | threshold = ", distThreshold,
    " | min = ", minWght,
    " | max = ", maxWght,
    " | prioritize = ", ifelse(prioritizeEdge, "edge", "far"), "]"
  )

  layout(matrix(1:4, nrow = 4), heights = c(1, 1, 1, 1))
  par(mar = c(1, 1, 2, 5))

  terra::plotRGB(img1, r = r, g = g, b = b, scale = 255,
                 axes = FALSE, stretch = "lin", main = "Images")

  terra::plot(msk1, type = "classes", axes = FALSE,
              levels = catData$cNames, col = catData$cColors,
              main = "Masks")

  terra::plot(dist1, axes = FALSE, col = distCols,
              main = "Distance to class boundary")

  terra::plot(wgt1, axes = FALSE, col = wgtCols, main = wgtTitle)

  layout(1)
  invisible(NULL)
}
