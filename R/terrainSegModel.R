gaussPyramids <- torch::nn_module(
  classname = "gaussPyramids",

  # Define the constructor
  initialize = function(inChn, spatDims) {
    self$inChn <- inChn
    self$spatDims <- spatDims

    # Define the custom kernel as a non-trainable tensor
    gauss <- torch::torch_tensor(c(1, 4, 6, 4, 1,
                                   4, 16, 24, 16, 4,
                                   6, 24, 36, 24, 6,
                                   4, 16, 24, 16, 4,
                                   1, 4, 6, 4, 1), device="cuda")$view(c(5, 5))$float() / 256

    gaussKernel <- torch::torch_stack(lapply(1:1, function(i) {
      torch::torch_stack(lapply(1:1, function(j) gauss), dim = 1)
    }), dim = 1)

    oneRow <- rep(c(1, 0), spatDims / 2)
    gridRow <- matrix(rep(oneRow, spatDims), ncol = spatDims, nrow = spatDims, byrow = TRUE)
    oneColO <- rep(1, spatDims)
    oneColE <- rep(0, spatDims)
    gridCol <- matrix(rep(c(oneColO, oneColE), spatDims / 2), nrow = spatDims, ncol = spatDims, byrow = TRUE)
    maskGrid <- gridCol * gridRow

    self$maskGridT <- torch::torch_tensor(maskGrid, dtype = torch::torch_float32(), requires_grad = FALSE, device = "cuda")

    # Register the custom kernel as a buffer so it won't be updated during training
    self$gauss_kernel <- gaussKernel
    self$gauss_kernel$requires_grad_(FALSE)
  },

  # Define the forward pass
  forward = function(x) {
    # `x` should have shape [batch, channels, height, width]
    batch_size <- x$size(1)
    channels <- x$size(2)

    process_layer <- function(layer) {
      l1_1 <- torch::nnf_conv2d(layer, self$gauss_kernel, stride = 1, padding = 2)
      l1_2 <- l1_1 * self$maskGridT
      l1_2 <- torch::nnf_conv2d(l1_2, self$gauss_kernel, stride = 1, padding = 2) * 4.0

      l1_3 <- l1_2 * self$maskGridT
      l1_3 <- torch::nnf_conv2d(l1_3, self$gauss_kernel, stride = 1, padding = 2) * 4.0

      l1_4 <- l1_3 * self$maskGridT
      l1_4 <- torch::nnf_conv2d(l1_4, self$gauss_kernel, stride = 1, padding = 2) * 4.0

      l1_5 <- l1_4 * self$maskGridT
      l1_5 <- torch::nnf_conv2d(l1_5, self$gauss_kernel, stride = 1, padding = 2) * 4.0

      return(torch::torch_cat(list(l1_1, l1_2, l1_3, l1_4, l1_5), dim = 2))

    }

    thePyramids <- process_layer(x)

    return(thePyramids)
  }
)


lspModule <- torch::nn_module(
  "lspModule",

  initialize = function(cellSize=1,
                        innerRadius=2,
                        outerRadius=5,
                        hsRadius=50,
                        smoothRadius=11,
                        doTPIHS = TRUE) {
    #
    # 0. Store user-defined parameters
    #
    self$cellSize      <- cellSize
    self$innerRadius   <- innerRadius
    self$outerRadius   <- outerRadius
    self$hsRadius      <- hsRadius
    self$smoothRadius  <- smoothRadius
    self$doTPIHS       <- doTPIHS

    self$register_buffer("sunAltitudeT", torch::torch_tensor((90.0 - 45) * (pi / 180.0)))

    self$register_buffer("sunAzimuthNT", torch::torch_tensor(((360.0 - 360.0) + 90.0) * (pi / 180.0)))
    self$register_buffer("sunAzimuthWT", torch::torch_tensor(((360.0 - 270.00) + 90.0) * (pi / 180.0)))
    self$register_buffer("sunAzimuthET", torch::torch_tensor(((360.0 - 90) + 90.0) * (pi / 180.0)))
    self$register_buffer("sunAzimuthST", torch::torch_tensor(((360.0 - 180) + 90.0) * (pi / 180.0)))

    #
    # 1. Create Slope / Curvature Kernels (kx, ky, kxx, kyy, kxy)
    #    We do this once and store as buffers.
    #
    kx_init <- torch::torch_tensor(
      array(c(-1,  0,  1,
              -2,  0,  2,
              -1,  0,  1),
            dim = c(1,1,3,3)),
      dtype = torch::torch_float()
    )

    ky_init <- torch::torch_tensor(
      array(c(-1, -2, -1,
              0,  0,  0,
              1,  2,  1),
            dim = c(1,1,3,3)),
      dtype = torch::torch_float()
    )

    # For curvature (normalized versions):
    kx_curv <- kx_init / 8.0
    ky_curv <- ky_init / 8.0

    kxx_curv <- torch::torch_tensor(
      array(c( 1, -2,  1,
               1, -2,  1,
               1, -2,  1),
            dim = c(1,1,3,3)),
      dtype = torch::torch_float()
    ) / 3.0

    kyy_curv <- torch::torch_tensor(
      array(c( 1,  1,  1,
               -2, -2, -2,
               1,  1,  1),
            dim = c(1,1,3,3)),
      dtype = torch::torch_float()
    ) / 3.0

    kxy_curv <- torch::torch_tensor(
      array(c( 1,  0, -1,
               0,  0,  0,
               -1,  0,  1),
            dim = c(1,1,3,3)),
      dtype = torch::torch_float()
    ) / 4.0

    #
    # Register them as buffers (NOT as parameters)
    #
    self$kx_slope <- torch::nn_buffer(kx_init$view(c(1,1,3,3)))  # original slope kernel
    self$ky_slope <- torch::nn_buffer(ky_init$view(c(1,1,3,3)))

    self$kx_curv <- torch::nn_buffer(kx_curv)
    self$ky_curv <- torch::nn_buffer(ky_curv)
    self$kxx_curv <- torch::nn_buffer(kxx_curv)
    self$kyy_curv <- torch::nn_buffer(kyy_curv)
    self$kxy_curv <- torch::nn_buffer(kxy_curv)

    #
    # 2. Create Annulus Kernel
    #
    annulus_size <- 2 * outerRadius + 1
    annulus_ker <- torch::torch_zeros(c(1, 1, annulus_size, annulus_size), dtype = torch::torch_float())
    centerA <- outerRadius
    for(i in seq_len(annulus_size)) {
      for(j in seq_len(annulus_size)) {
        dist <- sqrt(((i - 1) - centerA)^2 + ((j - 1) - centerA)^2)
        if(dist >= innerRadius && dist <= outerRadius) {
          annulus_ker[1, 1, i, j] <- 1
        }
      }
    }
    self$annulus_kernel <- torch::nn_buffer(annulus_ker)
    self$annulus_area   <- torch::nn_buffer(annulus_ker$sum())  # store as buffer

    #
    # 3. Hillslope Kernel
    #
    hs_size <- 2 * hsRadius + 1
    hs_ker <- torch::torch_zeros(c(1, 1, hs_size, hs_size), dtype = torch::torch_float())
    centerHS <- hsRadius
    for(i in seq_len(hs_size)) {
      for(j in seq_len(hs_size)) {
        dist <- sqrt(((i - 1) - centerHS)^2 + ((j - 1) - centerHS)^2)
        if(dist <= hsRadius) {
          hs_ker[1, 1, i, j] <- 1
        }
      }
    }
    self$hs_kernel <- torch::nn_buffer(hs_ker)
    self$hs_area   <- torch::nn_buffer(hs_ker$sum())

    #
    # 4. Smoothness Kernel
    #

    smth_size <- 2 * smoothRadius + 1
    smth_ker <- torch::torch_zeros(c(1, 1, smth_size, smth_size), dtype = torch::torch_float())
    centerR <- smoothRadius
    for(i in seq_len(smth_size)) {
      for(j in seq_len(smth_size)) {
        dist <- sqrt(((i - 1) - centerR)^2 + ((j - 1) - centerR)^2)
        if(dist <= smoothRadius) {
          smth_ker[1, 1, i, j] <- 1
        }
      }
    }
    self$smth_kernel <- torch::nn_buffer(smth_ker)
    self$smth_area   <- torch::nn_buffer(smth_ker$sum())

  },

  forward = function(inDTM) {

    # 1. Slope calculation

    dx <- torch::nnf_conv2d(inDTM, self$kx_slope, padding = 1)
    dy <- torch::nnf_conv2d(inDTM, self$ky_slope, padding = 1)
    dx <- dx/(8*self$cellSize)
    dy <- dy/(8*self$cellSize)
    gradMag <- torch::torch_sqrt((dx*dx)+(dy*dy))
    slpR <- torch::torch_atan(gradMag)
    slp <- slpR*57.2958
    slp <- torch::torch_sqrt(slp)
    slp <- torch::torch_clamp(slp, 0, 10)/(10.0)

    aspect <- pi/2.0 - torch::torch_atan2(-dy, dx)

    # 2. Hillshade

    hillshadeN <-  (torch::torch_cos(self$sunAltitudeT) * torch::torch_cos(slpR) +
                      torch::torch_sin(self$sunAltitudeT) * torch::torch_sin(slpR) *
                      torch::torch_cos(self$sunAzimuthNT - aspect)) * 255.0

    hillshadeE <- (torch::torch_cos(self$sunAltitudeT) * torch::torch_cos(slpR) +
                      torch::torch_sin(self$sunAltitudeT) * torch::torch_sin(slpR) *
                      torch::torch_cos(self$sunAzimuthET - aspect)) * 255.0

    hillshadeW <- (torch::torch_cos(self$sunAltitudeT) * torch::torch_cos(slpR) +
                     torch::torch_sin(self$sunAltitudeT) * torch::torch_sin(slpR) *
                     torch::torch_cos(self$sunAzimuthWT - aspect)) * 255.0

    hillshadeS <- (torch::torch_cos(self$sunAltitudeT) * torch::torch_cos(slpR) +
                      torch::torch_sin(self$sunAltitudeT) * torch::torch_sin(slpR) *
                      torch::torch_cos(self$sunAzimuthST - aspect)) * 255.0

    hillshade <- (hillshadeN + hillshadeE + hillshadeW + hillshadeS)/4.0

    hillshade <- torch::torch_clamp(hillshade, min = 0.0, max = 255.0)/255


    # 3. Local TPI

    neighborhood_sum <- torch::nnf_conv2d(inDTM, self$annulus_kernel,
                                          padding = self$outerRadius)
    neighborhood_mean <- neighborhood_sum$div(self$annulus_area)
    tpiL <- inDTM - neighborhood_mean
    tpiL <- torch::torch_clamp(tpiL, -10, 10)
    tpiL <- (tpiL + 10.0) / 20.0


    # 4. Hillslope TPI (conditional)

    if (self$doTPIHS) {
      hs_sum <- torch::nnf_conv2d(inDTM, self$hs_kernel, padding = self$hsRadius)
      hs_mean <- hs_sum$div(self$hs_area)
      tpiHS <- inDTM - hs_mean
      tpiHS <- torch::torch_clamp(tpiHS, -10, 10)
      tpiHS <- (tpiHS + 10.0) / 20.0
    }


    # 5. Curvatures

    sum_elev <- torch::nnf_conv2d(inDTM, self$smth_kernel, padding = self$smoothRadius)
    mean_elev <- sum_elev$div(self$smth_area)

    p <- torch::nnf_conv2d(mean_elev, self$kx_curv,  padding = 1)
    q <- torch::nnf_conv2d(mean_elev, self$ky_curv,  padding = 1)
    r_ <- torch::nnf_conv2d(mean_elev, self$kxx_curv, padding = 1)
    t_ <- torch::nnf_conv2d(mean_elev, self$kyy_curv, padding = 1)
    s_ <- torch::nnf_conv2d(mean_elev, self$kxy_curv, padding = 1)

    # Remove the singleton channel dimension (dimension 2) while keeping the batch dimension.
    p_ <- p$squeeze(2)
    q_ <- q$squeeze(2)
    r_ <- r_$squeeze(2)
    s_ <- s_$squeeze(2)
    t_ <- t_$squeeze(2)

    slope_sq <- p_$pow(2) + q_$pow(2)

    crvPln <- (q_$pow(2) * r_ - 2.0 * p_ * q_ * s_ + p_$pow(2) * t_) /
      (slope_sq$pow(1.5) + 1e-12)
    crvPro <- (p_$pow(2) * r_ + 2.0 * p_ * q_ * s_ + q_$pow(2) * t_) /
      (slope_sq$pow(1.5) + 1e-12)

    crvPln <- torch::torch_clamp(crvPln, -0.1, 0.1)
    crvPln <- (crvPln + 0.1) / 0.2

    crvPro <- torch::torch_clamp(crvPro, -0.1, 0.1)
    crvPro <- (crvPro + 0.1) / 0.2

    # Add back the channel dimension (as dimension 2) so that each curvature tensor is of shape (N, 1, H, W)
    crvPln <- crvPln$unsqueeze(2)
    crvPro <- crvPro$unsqueeze(2)

    # 6. Concatenate outputs



    if(self$doTPIHS){
      outLSPs <- torch::torch_cat(
        list(tpiHS, slp, tpiL, hillshade, crvPro, crvPln),
        dim = 2  # channel dimension
      )
    } else {
      outLSPs <- torch::torch_cat(
        list(slp, tpiL, hillshade, crvPro, crvPln),
        dim = 2  # channel dimension
      )
    }

    return(outLSPs)
  }
)

#' defineTerrainSeg
#'
#' CNN-based semantic segmentation wrapper that generates LSPs from a DTM and
#' passes them to a user-supplied trainable segmentation model.
#'
#' Define a CNN-based semantic segmentation model for landform extraction or
#' classification. The module generates land surface parameters (LSPs) from the
#' input DTM, crops the LSP tensor to remove edge artefacts, and then passes the
#' result to an externally instantiated trainable segmentation model. The model
#' assumes a single-band DTM of elevation measurements as input.
#'
#' When \code{doGP = FALSE} the LSP module produces 6 channels; when
#' \code{doGP = TRUE} it produces 31 channels. The segmentation model supplied
#' via \code{segMod} must be instantiated with a matching \code{inChn} value and
#' with an \code{inChn} that accounts for the spatial reduction caused by
#' \code{tCrop} (input chip size minus 2 * tCrop).
#'
#' @param segMod An already-instantiated trainable segmentation model
#' (torch \code{nn_module}). Must accept the number of input channels produced
#' by the LSP module: 6 when \code{doGP = FALSE}, 31 when \code{doGP = TRUE}.
#' @param cellSize Input resolution of DTM data in map units. Default is 1.
#' @param spatDim Spatial dimension (height = width) of the input chip in cells.
#' Default is 640.
#' @param tCrop Number of rows and columns to crop from each side of the LSP
#' output before passing to the segmentation model. This removes edge artefacts
#' introduced by the convolution-based LSP calculations. Default is 64.
#' @param doGP Whether to compute Gaussian Pyramids of the DTM and derive LSPs
#' at multiple scales. Default is FALSE. If FALSE, 6 LSPs are passed to the
#' model. If TRUE, 31 LSPs are passed.
#' @param innerRadius Inner radius (cells) of the annulus window used for local
#' TPI calculation. Default is 2.
#' @param outerRadius Outer radius (cells) of the annulus window used for local
#' TPI calculation. Default is 10.
#' @param hsRadius Radius (cells) of the circular moving window for hillslope
#' TPI calculation. Default is 50.
#' @param smoothRadius Radius (cells) of the circular moving window used to
#' smooth the DTM before curvature calculations. Default is 11.
#' @return A \code{terrainSeg} \code{nn_module} wrapping the LSP pipeline and
#' the supplied segmentation model.
#' @export
defineTerrainSeg <- torch::nn_module(
  classname = "terrainSeg",

  initialize = function(segMod,
                        cellSize     = 1,
                        spatDim      = 640,
                        tCrop        = 64,
                        doGP         = FALSE,
                        innerRadius  = 2,
                        outerRadius  = 10,
                        hsRadius     = 50,
                        smoothRadius = 11) {

    self$tCrop   <- tCrop
    self$doGP    <- doGP
    self$spatDim <- spatDim

    self$segMod <- segMod

    self$gaussPyramid <- gaussPyramids(1, spatDim)

    self$lspOrig <- lspModule(cellSize     = cellSize,
                              innerRadius  = innerRadius,
                              outerRadius  = outerRadius,
                              hsRadius     = hsRadius,
                              smoothRadius = smoothRadius,
                              doTPIHS      = TRUE)

    self$lspGP <- lspModule(cellSize     = cellSize,
                            innerRadius  = innerRadius,
                            outerRadius  = outerRadius,
                            hsRadius     = hsRadius,
                            smoothRadius = smoothRadius,
                            doTPIHS      = FALSE)
  },

  forward = function(x) {

    if (self$doGP) {
      xGP <- self$gaussPyramid(x)

      xLSP    <- self$lspOrig(x)
      xGPLSP1 <- self$lspGP(xGP[, 1, , ]$unsqueeze(dim = 2))
      xGPLSP2 <- self$lspGP(xGP[, 2, , ]$unsqueeze(dim = 2))
      xGPLSP3 <- self$lspGP(xGP[, 3, , ]$unsqueeze(dim = 2))
      xGPLSP4 <- self$lspGP(xGP[, 4, , ]$unsqueeze(dim = 2))
      xGPLSP5 <- self$lspGP(xGP[, 5, , ]$unsqueeze(dim = 2))

      tIn <- torch::torch_cat(list(xLSP, xGPLSP1, xGPLSP2, xGPLSP3, xGPLSP4, xGPLSP5), dim = 2)
    } else {
      tIn <- self$lspOrig(x)
    }

    tIn    <- cropTensor(tIn, self$tCrop)
    modOut <- self$segMod(tIn)

    return(modOut)
  }
)
