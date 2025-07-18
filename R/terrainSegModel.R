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
#' CNN-based semantic segmentation architecture of landform extraction or
#' classification from a DTM.
#'
#' Define a CNN-based semantic segmentation model for landform extraction or
#' classification that includes a module that generates land surface parameters
#' (LSPs) from the input DTM that are then passed to a semantic segmentation model.
#' A UNet, UNet with a MobileNetv2 encoder, UNet3+, or HRNet architecture can be used.
#' Gaussian pyramids can be calculated from the DTM to calculate LSPs at different
#' scales. If Gaussian pyramids are not use, 6 LSPs are passed to the segmentation
#' model. If Gaussian pyramids are used, 31 LSPs are passed to UNet3+. Model assumes
#' a single band DTM of elevation measurements as input.
#'
#' @param segModel Segmentation architecture to use. Either UNet ("UNet"), UNet3+
#' ("UNet3p"), UNet with a MobileNetv2 encoder ("MobileUNet") or HRNet ("HRNet")
#' @param cellSize Input resolution of DTM data. Default in 1 m.
#' @param nCls Number of classes being differentiated. For a binary classification,
#' this can be either 1 or 2. If 2, the problem is treated as a multiclass problem,
#' and a multiclass loss metric should be used. Default is 3.
#' @param spatDim Input chip size. Default is 512 (512x512 cells)
#' @param tCrop Number of rows and columns to crop prior to passing LSPs to UNet3+.
#' @param doGP Whether or not to include Gaussian Pyramids of DTM and calulate LSPs at
#' different scales. Default is FALSE. If FALSE, 6 LSPs are passed to model. If TRUE,
#' 31 LSPs are passed to model.
#' @param negative_slope Negative slope term for leaky ReLU
#' @param innterRadius Inner radius for annulus window for local TPI calculation. Default is 2 cells.
#' @param outerRadius Outer radius for annulus window for local TPI calculation. Default is 10 cells.
#' @param hsRadius Radius for circular moving window for hillslope TPI calculation. Defaults is 50 cells.
#' @param smoothRadius Radius of circular moving window to smooth DTM prior to curvature calculations.
#' Default is 11 cells.
#' @param actFunc Defines activation function to use throughout the network. "relu" =
#' rectified linear unit (ReLU); "lrelu" = leaky ReLU; "swish" = swish. Default is "relu".
#' @param useAttn TRUE or FALSE. Whether to add attention gates along the skip connections.
#' Default is FALSE or no attention gates are added.
#' @param useSE TRUE or FALSE. Whether or not to include squeeze and excitation modules in
#' the encoder. Default is FALSE or no squeeze and excitation modules are used.
#' @param useRes TRUE or FALSE. Whether to include residual connections in the encoder, decoder,
#' and bottleneck/ASPP module blocks. Default is FALSE or no residual connections are included.
#' @param useASPP TRUE or FALSE. Whether to use an ASPP module as the bottleneck as opposed to a
#' double convolution operation. Default is FALSE or the ASPP module is not used as the bottleneck.
#' @param useDS TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four predictions are
#' made, one at each decoder block resolution, and the predictions are returned as a list object
#' containing the 4 predictions. If FALSE, only the final prediction at the original resolution is
#' returned. Default is FALSE or deep supervision is not implemented.
#' @param enChn Vector of 4 integers defining the number of output
#' feature maps for each of the four encoder blocks. Default is 16, 32, 64, and 128.
#' @param dcChn Vector of 4 integers defining the number of output feature
#' maps for each of the 4 decoder blocks. Default is 128, 64, 32, and 16.
#' @param btnChn Number of output feature maps from the bottleneck block. Default
#' is 256.
#' @param dilRates Vector of 3 values specifying the dilation rates used in the ASPP module.
#' Default is 6, 12, and 18.
#' @param dilChn Vector of 4 values specifying the number of channels to produce at each dilation
#' rate within the ASPP module. Default is 256 for each dilation rate.
#' @param negative_slope If actFunc = "lrelu", specifies the negative slope term
#' to use. Default is 0.01.
#' @param seRatio Ratio to use in squeeze and excitation module. The default is 8.
#' @param pretrainedEncoder TRUE or FALSE. Whether or not to initialized using pre-trained ImageNet
#' weights for the
#' MobileNet-v2 encoder. Default is TRUE.
#' @param freezeEncoder TRUE or FALSE. Whether or not to freeze the encoder during training.
#' The default is TRUE. If TRUE, only the decoder component is trained.
#' @param avgImNetWeights TRUE or FALSE. If three predictor variables are provided and
#' ImageNet weights are used, whether or not to use the original weights or average them.
#' Default is FALSE.
#' @param l1FMs Number of feature maps to produce throughout the Level 1 layers. Default is 32.
#' @param l2FMs Number of feature maps to produce throughout the Level 2 layers. Default is 64.
#' @param l3FMs Number of feature maps to produce throughout the Level 3 layers. Default is 128.
#' @param l4FMs Number of feature maps to produce throughout the Level 4 layers. Default is 256.
#' @param dcFMs Number of feature maps to produce throughout the decoder blocks. Default is 256.
#' @return terrainSeg model consisting of LSP generation of UNet3+ model.
#' @export
defineTerrainSeg <- torch::nn_module(
  classname = "terrainSeg",

  # Define the constructor
  initialize = function(segModel = "UNet",
                        nCls = 3,
                        cellSize = 1,
                        spatDim=512,
                        tCrop = 256,
                        doGP = FALSE,
                        innerRadius = 2,
                        outerRadius  =10,
                        hsRadius = 50,
                        smoothRadius = 11,
                        actFunc="lrelu",
                        useAttn = FALSE,
                        useSE = FALSE,
                        useRes = TRUE,
                        useASPP = TRUE,
                        useDS = FALSE,
                        pretrainedEncoder = TRUE,
                        freezeEncoder = TRUE,
                        avgImNetWeights = FALSE,
                        enChn = c(16,32,64,128),
                        dcChn = c(128,64,32,16),
                        btnChn = 256,
                        dilChn = c(256,256,256),
                        dilRates = c(6, 12, 18),
                        l1FMs = 32,
                        l2FMs = 64,
                        l3FMs = 128,
                        l4FMs = 256,
                        dcFMs = 256,
                        negative_slope = 0.01,
                        seRatio=8){

    self$segModel = segModel
    self$nCls = nCls
    self$cellSize = cellSize
    self$spatDim=spatDim
    self$tCrop = tCrop
    self$doGP = doGP
    self$innerRadius = innerRadius
    self$outerRadius  = outerRadius
    self$hsRadius = hsRadius
    self$smoothRadius = smoothRadius
    self$actFunc= actFunc
    self$useAttn = useAttn
    self$useSE = useSE
    self$useRes = useRes
    self$useASPP = useASPP
    self$useDS = useDS
    self$pretrainedEncoder = pretrainedEncoder
    self$freezeEncoder = freezeEncoder
    self$avgImNetWeights = avgImNetWeights
    self$enChn = enChn
    self$dcChn = dcChn
    self$btnChn = btnChn
    self$dilChn = dilChn
    self$dilRates = dilRates
    self$l1FMs = l1FMs
    self$l2FMs = l2FMs
    self$l3FMs = l3FMs
    self$l4FMs = l4FMs
    self$dcFMs = dcFMs
    self$negative_slope = negative_slope
    self$seRatio = seRatio

    if(self$doGP == TRUE){
      self$inChn <- 31
    }else{
      self$inChn <- 6
    }

    self$gaussPyramid <- gaussPyramids(1, self$spatDim)

    self$lspOrig <- lspModule(cellSize=self$cellSize,
                              innerRadius=self$innerRadius,
                              outerRadius=self$outerRadius,
                              hsRadius=self$hsRadius,
                              smoothRadius=self$smoothRadius,
                              doTPIHS = TRUE)

    self$lspGP <- lspModule(cellSize= self$cellSize,
                            innerRadius=self$innerRadius,
                            outerRadius=self$outerRadius,
                            hsRadius=self$hsRadius,
                            smoothRadius=self$smoothRadius,
                            doTPIHS = FALSE)

    if(segModel == "UNet3p"){
      self$segMod <- defineUNet3p(inChn=self$inChn,
                              nCls=self$nCls,
                              useDS = self$useDS,
                              enChn = self$enChn,
                              dcChn = self$dcChn,
                              btnChn = self$btnChn,
                              negative_slope=self$negative_slope)
    }else if(segModel == "MobileUNet"){
      self$segMod <- defineMobileUNet(inChn = self$inChn,
                              nCls = self$nCls,
                              pretrainedEncoder = self$pretrainedEncoder,
                              freezeEncoder = self$freezeEncoder,
                              avgImNetWeights = self$avgImNetWeights,
                              actFunc = self$actFunc,
                              useAttn = self$useAttn,
                              useDS = self$useDS,
                              dcChn = self$dcChn,
                              negative_slope = self$negative_slope)
    }else if(segModel == "HRNet"){
      self$segMod <- defineHRNet(inChn=self$inChn,
                               nCls = self$nCls,
                               l1FMs = self$l1FMs,
                               l2FMs = self$l2FMs,
                               l3FMs = self$l3FMs,
                               l4FMs = self$l4FMs,
                               dcFMs = self$dcFMs,
                               dilChn = self$dilChn,
                               dilRates = self$dilRates,
                               actFunc = self$actFunc,
                               negative_slope = self$negative_slope)

    }else if(segModel == "UNet"){
      self$segMod <- defineUNet(inChn = self$inChn,
                               nCls = self$nCls,
                               actFunc = self$actFunc,
                               useAttn = self$actFunc,
                               useSE = self$actFunc,
                               useRes = self$actFunc,
                               useASPP = self$actFunc,
                               useDS = self$actFunc0,
                               enChn = self$actFunc,
                               dcChn = self$actFunc,
                               btnChn = self$actFunc,
                               dilRates= self$actFunc,
                               dilChn= self$actFunc,
                               negative_slope= self$actFunc,
                               seRatio=8)

    }else{
      message("Invalid Segmentation Model.")
    }
  },

  # Define the forward pass
  forward = function(x) {

    if(self$doGP == TRUE){
      xGP <- self$gaussPyramid(x)

      #LSPs

      xLSP <- self$lspOrig(x)
      xGP1 <- xGP[,1,,]$unsqueeze(dim=2)
      xGP2 <- xGP[,2,,]$unsqueeze(dim=2)
      xGP3 <- xGP[,3,,]$unsqueeze(dim=2)
      xGP4 <- xGP[,4,,]$unsqueeze(dim=2)
      xGP5 <- xGP[,5,,]$unsqueeze(dim=2)

      xGPLSP1 <- self$lspGP(xGP1)
      xGPLSP2 <- self$lspGP(xGP2)
      xGPLSP3 <- self$lspGP(xGP3)
      xGPLSP4 <- self$lspGP(xGP4)
      xGPLSP5 <- self$lspGP(xGP5)

      tIn <- torch::torch_cat(list(xLSP, xGPLSP1, xGPLSP2,xGPLSP3,xGPLSP4,xGPLSP5), dim = 2)
    }else{
      tIn <- self$lspOrig(x)
    }

    tIn <- cropTensor(tIn, self$tCrop)

    modOut <- self$segMod(tIn)

    return(modOut)
  }
)
