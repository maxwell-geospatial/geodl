compute_aspect <- function(dx, dy) {
  # Case 1: When dx is non-zero
  aspect_rad_nonzero <- torch::torch_atan2(dy, -dx)
  aspect_rad_nonzero <- torch::torch_where(aspect_rad_nonzero < 0,
                                    aspect_rad_nonzero + (2 * pi),
                                    aspect_rad_nonzero)

  # Case 2: When dx is zero:
  #   If dy > 0 then aspect = pi/2,
  #   else if dy < 0 then aspect = 2*pi - pi/2,
  #   else (if dy == 0) then we set aspect = 0.
  aspect_rad_zero <- torch::torch_where(dy > 0,
                                 torch::torch_tensor(pi / 2, dtype = torch::torch_float()),
                                 torch::torch_where(dy < 0,
                                             torch::torch_tensor(2 * pi - pi / 2, dtype = torch::torch_float()),
                                             torch::torch_tensor(0, dtype = torch::torch_float())))

  # Combine the two cases:
  # When dx is non-zero, use aspect_rad_nonzero;
  # when dx equals zero, use aspect_rad_zero.
  aspect_rad <- torch::torch_where(dx != 0, aspect_rad_nonzero, aspect_rad_zero)

  return(aspect_rad)
}


# Slope -------------------------------------------------------------------

makeSlopeModule <- torch::nn_module(
  "TopographicSlope",

  initialize = function(cellSize=1) {
    #
    # 0. Store user-defined parameters
    #
    self$cellSize      <- cellSize

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

    #
    # Register them as buffers (NOT as parameters)
    #
    self$kx_slope <- torch::nn_buffer(kx_init$view(c(1,1,3,3)))  # original slope kernel
    self$ky_slope <- torch::nn_buffer(ky_init$view(c(1,1,3,3)))

  },

  forward = function(inDTM) {

    # 1. Slope calculation

    dx <- torch::nnf_conv2d(inDTM, self$kx_slope, padding = 1)
    dy <- torch::nnf_conv2d(inDTM, self$ky_slope, padding = 1)
    dx <- dx/(8*self$cellSize)
    dy <- dy/(8*self$cellSize)
    gradMag <- torch::torch_sqrt((dx*dx)+(dy*dy))
    slp <- torch::torch_atan(gradMag)*57.2958

    return(slp)
  }
)


#' makeSlope
#'
#' Calculate slope from a digital terrain model (DTM) using torch
#'
#' Calculate topographic slope using torch in degree units. Processing on the GPU can
#' be much faster than using base R and non-tensor-based calculations.
#'
#' @param dtm Input SpatRaster object representing bare earth surface elevations.
#' @param cellSize Resolution of raster grid. Default is 1 m.
#' @param writeRaster TRUE or FALSE. Save output to disk. Default is TRUE.
#' @param outName Name of output raster with full file path and extension.
#' @param device "cpu" or "cuda". Use "cuda" for GPU computation. Without using the GPU,
#' implementation will not be significantly faster than using non-tensor-based computation.
#' Defaults is "cpu".
#' @examples
#' \dontrun{
#' pth <- "OUTPUT PATH"
#' dtm <- rast(paste0(pth, "dtm.tif"))
#' slpR <- makeSlope(dtm, cellSize=1, writeRaster=TRUE, outName=paste0(pth, "slp.tif"), device="cuda")
#' }
#' @export
makeSlope <- function(dtm, cellSize=1, writeRaster=FALSE, outName, device="cpu"){
  dtmA <- terra::as.array(dtm)
  dtmT <- torch::torch_tensor(dtmA)$permute(c(3,1,2))$to(device=device)

  doSlope <- makeSlopeModule(cellSize=cellSize)$to(device=device)

  slp <- doSlope(dtmT)

  slpA <- terra::as.array(slp$permute(c(2,3,1))$cpu())
  slpR <- terra::rast(slpA)

  terra::ext(slpR) <- terra::ext(dtm)
  terra::crs(slpR) <- terra::crs(dtm)

  if(writeRaster ==TRUE){
    terra::writeRaster(slpR, outName, overwrite=TRUE)
  }

  return(slpR)
}


# Aspect ------------------------------------------------------------------

makeAspectModule <- torch::nn_module(
  "TopographicAspect",

  initialize = function(cellSize = 1, flatThreshold=1, mode="aspect") {
    # Store user-defined parameters.
    self$cellSize <- cellSize
    self$flatThreshold <- flatThreshold
    self$mode <- mode

    # Define the Sobel kernel for the x-derivative (east–west).
    # This kernel estimates change in elevation in the x-direction.
    kx_init <- torch::torch_tensor(
      array(c(-1,  0,  1,
              -2,  0,  2,
              -1,  0,  1),
            dim = c(1, 1, 3, 3)),
      dtype = torch::torch_float()
    )

    # Define the Sobel kernel for the y-derivative (north–south).
    # This kernel estimates change in elevation in the y-direction.
    ky_init <- torch::torch_tensor(
      array(c(-1, -2, -1,
              0,  0,  0,
              1,  2,  1),
            dim = c(1, 1, 3, 3)),
      dtype = torch::torch_float()
    )

    # Register these kernels as buffers (non-learnable parameters).
    self$kx_aspect <- torch::nn_buffer(kx_init$view(c(1, 1, 3, 3)))
    self$ky_aspect <- torch::nn_buffer(ky_init$view(c(1, 1, 3, 3)))
  },

  forward = function(inDTM) {
    # 1. Compute gradients using the Sobel kernels.
    #    dx estimates the east–west gradient and dy the north–south gradient.
    dx <- torch::nnf_conv2d(inDTM, self$kx_aspect, padding = 1)
    dy <- torch::nnf_conv2d(inDTM, self$ky_aspect, padding = 1)
    dx <- dx / (8 * self$cellSize)
    dy <- dy / (8 * self$cellSize)

    # 2. Compute upslope aspect (the direction the slope faces) in compass degrees.
    #    Using the formula:
    #       aspect = 90 - (atan2(dy, -dx) converted to degrees)
    #    This adjustment ensures:
    #       - North = 0°,
    #       - East  = 90°,
    #       - South = 180°,
    #       - West  = 270°.
    aspect <- 180 + torch::torch_atan2(dy, -dx) * 57.2958

    gradMag <- torch::torch_sqrt((dx*dx)+(dy*dy))
    slp <- torch::torch_atan(gradMag)*57.2958

    aspect <- torch::torch_where(slp > self$flatThreshold, aspect, -1)

    if(self$mode == "northness"){
      northness <- torch::torch_cos((aspect*(pi/180)))
      return(northness)
    }else if(self$mode == "eastness"){
      eastness <- torch::torch_sin((aspect*(pi/180)))
      return(eastness)
    }else if(self$mode == "trasp"){
      trasp <- (1.00-torch::torch_cos((pi/180)*(aspect-30.0)))/2.00
      return(trasp)
    }else if(self$mode == "sei"){
      sei <- slp*(torch::torch_cos(pi*((aspect-180)/180)))
      return(sei)
    }else{
      return(aspect)
    }
  }
)




#' makeAspect
#'
#' Calculate aspect or slope orientation  or a related metric from a digital terrain model (DTM) using torch
#'
#' Calculate topographic aspect or slope orientation or a related metric using torch
#' in degree units. Processing on the GPU can be much faster than using base R and non-tensor-based calculations.
#' "aspect" = topographic aspect in degree units where flat slopes are coded to -1; "northness" = cosine of aspect
#' in radians; "eastness" = sine of aspect in radians; "trasp" = topographic radiation aspect index; "sei" = site exposure
#' index.
#'
#' @param dtm Input SpatRaster object representing bare earth surface elevations.
#' @param cellSize Resolution of raster grid. Default is 1 m.
#' @param flatThreshold Maximum slope to be re-coded to flat and assigned an aspect value of -1.
#' Default is 1-degree.
#' @param mode "aspect", "northness", "eastness", "trasp", or "sei". Default is "aspect".
#' @param writeRaster TRUE or FALSE. Save output to disk. Default is TRUE.
#' @param outName Name of output raster with full file path and extension.
#' @param device "cpu" or "cuda". Use "cuda" for GPU computation. Without using the GPU,
#' implementation will not be significantly faster than using non-tensor-based computation.
#' Defaults is "cpu".
#' @examples
#' \dontrun{
#' pth <- "OUTPUT PATH"
#' dtm <- rast(paste0(pth, "dtm.tif"))
#' aspR <- makeAspect(dtm, cellSize=1,
#' flatThreshold=1,
#' mode= "aspect",
#' writeRaster=TRUE,
#' outName=paste0(pth, "asp.tif"),
#' device="cuda")
#' }
#' @export
makeAspect <- function(dtm, cellSize=1, flatThreshold = 1, mode="aspect", writeRaster=FALSE, outName, device="cpu"){
  dtmA <- terra::as.array(dtm)
  dtmT <- torch::torch_tensor(dtmA)$permute(c(3,1,2))$to(device=device)

  doAspect <- makeAspectModule(cellSize=cellSize, flatThreshold=flatThreshold, mode=mode)$to(device=device)

  asp <- doAspect(dtmT)

  aspA <- terra::as.array(asp$permute(c(2,3,1))$cpu())
  aspR <- terra::rast(aspA)

  terra::ext(aspR) <- terra::ext(dtm)
  terra::crs(aspR) <- terra::crs(dtm)

  if(writeRaster ==TRUE){
    writeRaster(aspR, outName, overwrite=TRUE)
  }

  return(aspR)
}



# Hillshade ---------------------------------------------------------------

makeHillshadeModule <- torch::nn_module(
  "Hillshade",

  initialize = function(cellSize = 1, sunAzimuth = 315, sunAltitude = 45, doMD =TRUE) {
    # Store user-defined parameters and convert sun position angles to radians
    self$cellSize   <- cellSize
    self$doMD <- doMD

    self$register_buffer("sunAzimuthT", torch::torch_tensor(((360.0 - sunAzimuth) + 90.0) * (pi / 180.0)))
    self$register_buffer("sunAltitudeT", torch::torch_tensor((90.0 - sunAltitude) * (pi / 180.0)))

    self$register_buffer("sunAzimuthNT", torch::torch_tensor(((360.0 - 360.0) + 90.0) * (pi / 180.0)))
    self$register_buffer("sunAzimuthNWT", torch::torch_tensor(((360.0 - 315.0) + 90.0) * (pi / 180.0)))
    self$register_buffer("sunAzimuthWT", torch::torch_tensor(((360.0 - 270.0) + 90.0) * (pi / 180.0)))
    self$register_buffer("sunAzimuthSET", torch::torch_tensor(((360.0 - 135.0) + 90.0) * (pi / 180.0)))

    # Create Sobel kernels for gradient estimation
    kx_init <- torch::torch_tensor(
      array(c(-1,  0,  1,
              -2,  0,  2,
              -1,  0,  1),
            dim = c(1, 1, 3, 3)),
      dtype = torch::torch_float()
    )
    ky_init <- torch::torch_tensor(
      array(c(-1, -2, -1,
              0,  0,  0,
              1,  2,  1),
            dim = c(1, 1, 3, 3)),
      dtype = torch::torch_float()
    )

    # Register the kernels as buffers (not learnable parameters)
    self$kx <- torch::nn_buffer(kx_init$view(c(1, 1, 3, 3)))
    self$ky <- torch::nn_buffer(ky_init$view(c(1, 1, 3, 3)))
  },

  forward = function(inDTM) {
    # 1. Compute gradients using Sobel kernels
    dx <- torch::nnf_conv2d(inDTM, self$kx, padding = 1)
    dy <- torch::nnf_conv2d(inDTM, self$ky, padding = 1)
    dx <- (dx / (8.0 * self$cellSize))
    dy <- (dy / (8.0 * self$cellSize))

    # 2. Calculate the slope in radians
    slope <- torch::torch_atan(torch::torch_sqrt(dx * dx + dy * dy))

    # 3. Calculate the aspect in radians.
    # The conversion (pi/2 - atan2(-dy, dx)) aligns the result with the typical DEM coordinate system.
    aspect <- pi/2.0 - torch::torch_atan2(-dy, dx)

    # 4. Compute hillshade using the illumination model

    if(self$doMD == TRUE){
      hillshadeN <-  (torch::torch_cos(self$sunAltitudeT) * torch::torch_cos(slope) +
                            torch::torch_sin(self$sunAltitudeT) * torch::torch_sin(slope) *
                            torch::torch_cos(self$sunAzimuthNT - aspect)) * 255.0

      hillshadeNW <- (torch::torch_cos(self$sunAltitudeT) * torch::torch_cos(slope) +
                            torch::torch_sin(self$sunAltitudeT) * torch::torch_sin(slope) *
                            torch::torch_cos(self$sunAzimuthNWT - aspect)) * 255.0

      hillshadeW <- (torch::torch_cos(self$sunAltitudeT) * torch::torch_cos(slope) +
                            torch::torch_sin(self$sunAltitudeT) * torch::torch_sin(slope) *
                            torch::torch_cos(self$sunAzimuthWT - aspect)) * 255.0

      hillshadeSE <- (torch::torch_cos(self$sunAltitudeT) * torch::torch_cos(slope) +
                            torch::torch_sin(self$sunAltitudeT) * torch::torch_sin(slope) *
                            torch::torch_cos(self$sunAzimuthSET - aspect)) * 255.0

      hillshade <- (hillshadeN + (2.00*hillshadeNW) + hillshadeW + hillshadeSE)/5.0

      hillshade <- torch::torch_clamp(hillshade, min = 0.0, max = 255.0)
    }else{
      hillshade <- (torch::torch_cos(self$sunAltitudeT) * torch::torch_cos(slope) +
                            torch::torch_sin(self$sunAltitudeT) * torch::torch_sin(slope) *
                            torch::torch_cos(self$sunAzimuthT - aspect)) * 255.0

      hillshade <- torch::torch_clamp(hillshade, min = 0.0, max = 255.0)
    }

    return(hillshade)
  }
)


#' makeAspect
#'
#' Calculate a hillshde from a digital terrain model (DTM) using torch
#'
#' Calculate a hillshade from a digital terrain model (DTM) using torch. User can specify a
#' illuminating position using an aspect and altitude. A multidirectiontal hillshade is calculated by averaging
#' hillshades with different sun positions ((north X 2Xnorthwest X west X southeast)/5.
#'
#' @param dtm Input SpatRaster object representing bare earth surface elevations.
#' @param cellSize Resolution of raster grid. Default is 1 m.
#' @param sunAzimuth Direction of illuminating source as a compass direction. Default is 315-degrees or northwest.
#' @param sunAltitude Angle of illuminating source above the horizon from 0-degrees (horizon) to 90-degrees (zenith).
#' Default is 45-degrees
#' @param doMD TRUE or FALSE. Whether or not to generate a multidirectional hillshade. Default is FALSE.
#' @param writeRaster TRUE or FALSE. Save output to disk. Default is TRUE.
#' @param outName Name of output raster with full file path and extension.
#' @param device "cpu" or "cuda". Use "cuda" for GPU computation. Without using the GPU,
#' implementation will not be significantly faster than using non-tensor-based computation.
#' Defaults is "cpu".
#' @examples
#' \dontrun{
#' pth <- "OUTPUT PATH"
#' dtm <- rast(paste0(pth, "dtm.tif"))
#' hsR <- makeHillshade(dtm,
#' cellSize=1,
#' sunAzimuth=315,
#' sunAltitude=45,
#' doMD = FALSE,
#' writeRaster=TRUE,
#' outName=paste0(pth, "hs.tif"), device="cuda")
#' }
#' @export
makeHillshade <- function(dtm,
                          cellSize=1,
                          sunAzimuth=315,
                          sunAltitude=45,
                          doMD = FALSE,
                          writeRaster=FALSE,
                          outName,
                          device="cpu"){
  dtmA <- terra::as.array(dtm)
  dtmT <- torch::torch_tensor(dtmA)$permute(c(3,1,2))$to(device=device)

  if(doMD == TRUE){
    doHS <- makeHillshadeModule(cellSize = 1,
                                sunAzimuth = sunAzimuth,
                                sunAltitude = sunAltitude,
                                doMD=TRUE)$to(device=device)
  }else{
    doHS <- makeHillshadeModule(cellSize = 1,
                                sunAzimuth = sunAzimuth,
                                sunAltitude = sunAltitude,
                                doMD=FALSE)$to(device=device)
  }

  hs <- doHS(dtmT)

  hsA <- terra::as.array(hs$permute(c(2,3,1))$cpu())
  hsR <- terra::rast(hsA)

  terra::ext(hsR) <- terra::ext(dtm)
  terra::crs(hsR) <- terra::crs(dtm)

  if(writeRaster ==TRUE){
    terra::writeRaster(hsR, outName, overwrite=TRUE)
  }

  return(hsR)
}


# Topographic Position Index ----------------------------------------------

makeTPIModule <- torch::nn_module(
  "TPI",

  initialize = function(cellSize=1,
                        mode="circle",
                        innerRadius=2,
                        outerRadius=5){


    self$cellSize      <- cellSize
    self$mode          <- mode
    self$innerRadius   <- innerRadius
    self$outerRadius   <- outerRadius

    if(mode == "annulus"){
      k_size <- 2 * outerRadius + 1
      k_ker <- torch::torch_zeros(c(1, 1, k_size, k_size), dtype = torch::torch_float())
      centerK <- outerRadius
      for(i in seq_len(k_size)) {
        for(j in seq_len(k_size)) {
          dist <- sqrt(((i - 1) - centerK)^2 + ((j - 1) - centerK)^2)
          if(dist >= innerRadius && dist <= outerRadius) {
            k_ker[1, 1, i, j] <- 1
          }
        }
      }
      self$tpi_kernel <- torch::nn_buffer(k_ker)
      self$tpi_area   <- torch::nn_buffer(k_ker$sum())  # store as buffer
    }else{
      k_size <- 2 * outerRadius + 1
      k_ker <- torch::torch_zeros(c(1, 1, k_size, k_size), dtype = torch::torch_float())
      centerK <- outerRadius
      for(i in seq_len(k_size)) {
        for(j in seq_len(k_size)) {
          dist <- sqrt(((i - 1) - centerK)^2 + ((j - 1) - centerK)^2)
          if(dist <= outerRadius) {
            k_ker[1, 1, i, j] <- 1
          }
        }
      }
      self$tpi_kernel <- torch::nn_buffer(k_ker)
      self$tpi_area   <- torch::nn_buffer(k_ker$sum())
    }

  },

  forward = function(inDTM) {
    neighborhood_sum <- torch::nnf_conv2d(inDTM, self$tpi_kernel,padding = self$outerRadius)

    neighborhood_mean <- neighborhood_sum$div(self$tpi_area)

    tpi <- inDTM-neighborhood_mean

    return(tpi)
  }
)

#' makeTPI
#'
#' Calculate a topographic position index (TPI) from a digital terrain model (DTM) using torch
#'
#' Calculate a topographic position index (TPI) from a digital terrain model (DTM) using torch. TPI is calculated
#' as the center cell elevation minus the mean elevation in a local moving window. Large, positive values indicate local
#' topographic high points while large, negative values indicate topographic low points. Near zero values indicate flat areas.
#' Larger windows can characterize the hillslope position of a cell while smaller windows can capture local topographic variability.
#' Radii are specified using cell counts as opposed to map distance.
#'
#' @param dtm Input SpatRaster object representing bare earth surface elevations.
#' @param cellSize Resolution of raster grid. Default is 1 m.
#' @param innerRadius = Inner radius when using annulus moving window. Default is 2 cells.
#' @param outerRadius = Outer radius when using a circle or annulus moving window. Default is 10 cells.
#' @param mode Either "circle" or "annulus". Default is "circle".
#' @param writeRaster TRUE or FALSE. Save output to disk. Default is TRUE.
#' @param outName Name of output raster with full file path and extension.
#' @param device "cpu" or "cuda". Use "cuda" for GPU computation. Without using the GPU,
#' implementation will not be significantly faster than using non-tensor-based computation.
#' Defaults is "cpu".
#' @examples
#' \dontrun{
#' pth <- "OUTPUT PATH"
#' dtm <- rast(paste0(pth, "dtm.tif"))
#' tpiA3_11 <- makeTPI(dtm,cellSize=1,
#' innerRadius=3,
#' outerRadius=11,
#' mode="circle",
#' writeRaster=TRUE,
#' outName=paste0(pth, "tpiA3_11.tif"),
#' device="cuda")
#' }
#' @export
makeTPI <- function(dtm,
                    cellSize=1,
                    innerRadius=2,
                    outerRadius=10,
                    mode="circle",
                    writeRaster=FALSE,
                    outName,
                    device="cpu"){

  dtmA <- terra::as.array(dtm)
  dtmT <- torch::torch_tensor(dtmA)$permute(c(3,1,2))$to(device=device)

  doTPI <- makeTPIModule(cellSize=1,
                         mode=mode,
                         innerRadius=innerRadius,
                         outerRadius=outerRadius)$to(device=device)

  tpi <- doTPI(dtmT)

  tpiA <- terra::as.array(tpi$permute(c(2,3,1))$cpu())
  tpiR <- terra::rast(tpiA)

  terra::ext(tpiR) <- terra::ext(dtm)
  terra::crs(tpiR) <- terra::crs(dtm)

  if(writeRaster ==TRUE){
    terra::writeRaster(tpiR, outName, overwrite=TRUE)
  }

  return(tpiR)
}



# Topographic Roughness Index ---------------------------------------------

makeTRIModule <- torch::nn_module(
  "TRIConv",

  initialize = function(roughRadius = 7) {
    self$rr <- roughRadius
    k <- 2 * self$rr + 1

    # 1) compute all (Δi,Δj) inside the circle
    offsets <- list()
    center <- self$rr + 1
    for (i in seq_len(k)) {
      for (j in seq_len(k)) {
        if (sqrt((i - center)^2 + (j - center)^2) <= self$rr) {
          offsets[[length(offsets) + 1]] <- c(i, j)
        }
      }
    }
    N <- length(offsets)

    # 2) first conv: N filters of size k×k, padding=rr, no bias
    self$diff_conv <- torch::nn_conv2d(
      in_channels  = 1,
      out_channels = N,
      kernel_size  = c(k, k),
      padding      = self$rr,
      bias         = FALSE
    )

    # 3) initialize those filters: +1 at neighbor, -1 at center
    torch::with_no_grad({
      self$diff_conv$weight$zero_()
      for (m in seq_len(N)) {
        off <- offsets[[m]]
        # +1 at neighbor
        self$diff_conv$weight[m, 1, off[1],   off[2]  ] <- 1
        # -1 at center
        self$diff_conv$weight[m, 1, center, center]   <- -1
      }
    })
    # freeze diff_conv weights
    self$diff_conv$weight$requires_grad_(FALSE)

    # 4) second conv: 1×1 conv to sum N channels (mean if we scale weights)
    self$sum_conv <- torch::nn_conv2d(
      in_channels  = N,
      out_channels = 1,
      kernel_size  = 1,
      bias         = FALSE
    )
    torch::with_no_grad({
      self$sum_conv$weight$fill_(1.0 / N)
    })
    # freeze sum_conv weights
    self$sum_conv$weight$requires_grad_(FALSE)
  },

  forward = function(x) {
    # x: [B,1,H,W]
    diffs <- self$diff_conv(x)$abs()   # [B,N,H,W]
    tri   <- self$sum_conv(diffs)      # [B,1,H,W]
    return(tri)
  }
)



#' makeTRI
#'
#' Calculate a topographic roughness index (TRI) from a digital terrain model (DTM) using torch
#'
#' Calculate a topographic roughness index (TPI) from a digital terrain model (DTM) using torch.
#' Us calculated as the square root of the standard deviation of slope in a local moving window.
#' Output can be noisy. Radii are specified using cell counts as opposed to map distance.
#'
#' @param dtm Input SpatRaster object representing bare earth surface elevations.
#' @param cellSize Resolution of raster grid. Default is 1 m.
#' @param roughRadius radius of circular moving window. Default is 7 cells.
#' @param writeRaster TRUE or FALSE. Save output to disk. Default is TRUE.
#' @param outName Name of output raster with full file path and extension.
#' @param device "cpu" or "cuda". Use "cuda" for GPU computation. Without using the GPU,
#' implementation will not be significantly faster than using non-tensor-based computation.
#' Defaults is "cpu".
#' @examples
#' \dontrun{
#' pth <- "OUTPUT PATH"
#' dtm <- rast(paste0(pth, "dtm.tif"))
#' tri11 <- makeTRI(dtm,
#' cellSize=1,
#' roughRadius=11,
#' writeRaster=TRUE,
#' outName=paste0(pth, "tri11f.tif"),
#' device="cuda")
#' }
#' @export
makeTRI <- function(dtm, cellSize=1,
                    roughRadius=7,
                    writeRaster=FALSE,
                    outName,
                    device="cpu"){

  dtmA <- terra::as.array(dtm)
  dtmT <- torch::torch_tensor(dtmA)$permute(c(3,1,2))$to(device=device)

  doTRI <- makeTRIModule(roughRadius=roughRadius)$to(device=device)

  tri <- doTRI(dtmT)

  triA <- terra::as.array(tri$permute(c(2,3,1))$cpu())
  triR <- terra::rast(triA)

  terra::ext(triR) <- terra::ext(dtm)
  terra::crs(triR) <- terra::crs(dtm)

  if(writeRaster ==TRUE){
    terra::writeRaster(triR, outName, overwrite=TRUE)
  }

  return(triR)
}


# Curvature -------------------------------------------------------


makeCrvModule <- torch::nn_module(
  "Curvature",

  initialize = function(cellSize=1,
                        mode = "mean",
                        smoothRadius=11) {
    #
    # 0. Store user-defined parameters
    #
    self$cellSize      <- cellSize
    self$mode          <- mode
    self$smoothRadius  <- smoothRadius

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
    # 5. Smoothness Kernel
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

    if(self$mode == "profile"){
      crvPro <- (p_$pow(2) * r_ + 2.0 * p_ * q_ * s_ + q_$pow(2) * t_) /
        (slope_sq$pow(1.5) + 1e-12)
      return(crvPro)
    }else if(self$mode == "planform"){
      crvPln <- (q_$pow(2) * r_ - 2.0 * p_ * q_ * s_ + p_$pow(2) * t_) /
        (slope_sq$pow(1.5) + 1e-12)
      return(crvPln)
    }else{
      crvPln <- (q_$pow(2) * r_ - 2.0 * p_ * q_ * s_ + p_$pow(2) * t_) /
        (slope_sq$pow(1.5) + 1e-12)
      crvPro <- (p_$pow(2) * r_ + 2.0 * p_ * q_ * s_ + q_$pow(2) * t_) /
        (slope_sq$pow(1.5) + 1e-12)
      crvMn <- (crvPln+crvPro)/2.0
      return(crvMn)
    }
  }
)

#' makeCrv
#'
#' Calculate surface curvatures from input digital terrain model (DTM) using torch
#'
#' Calculate surface curvatures from input digital terrain model (DTM) using torch.
#' "mean" = mean curvature or average of profile and planform curvature; "profile" = curvature
#' in the direction of maximum slope; "planform" = curvature in the direction perpendicular to
#' the direction of maximum slope. Radii are specified using cell counts as opposed to map distance.
#' Gaussian smoothing using a circular window of a user-defined radius is applied prior to calculating
#' curvatures to minimize the impact of local noise/variability.
#'
#' @param dtm Input SpatRaster object representing bare earth surface elevations.
#' @param cellSize Resolution of raster grid. Default is 1 m.
#' @param mode "mean", "profile", or "planform". Default is "mean".
#' @param smoothRadius radius of circular moving window. Default is 7 cells.
#' @param writeRaster TRUE or FALSE. Save output to disk. Default is TRUE.
#' @param outName Name of output raster with full file path and extension.
#' @param device "cpu" or "cuda". Use "cuda" for GPU computation. Without using the GPU,
#' implementation will not be significantly faster than using non-tensor-based computation.
#' Defaults is "cpu".
#' @examples
#' \dontrun{
#' pth <- "OUTPUT PATH"
#' dtm <- rast(paste0(pth, "dtm.tif"))
#' crvPro7 <- makeCrv(dtm,
#' cellSize=1,
#' mode="profile",
#' smoothRadius=7,
#' writeRaster=TRUE,
#' outName=paste0(pth, "crvPro7.tif"),
#' device="cuda")
#' }
#' @export
makeCrv <- function(dtm,
                    cellSize=1,
                    mode="mean",
                    smoothRadius=7,
                    writeRaster=FALSE,
                    outName,
                    device="cpu"){

  dtmA <- terra::as.array(dtm)
  dtmT <- torch::torch_tensor(dtmA)$permute(c(3,1,2))$to(device=device)

  doCrv <- makeCrvModule(cellSize=1,
                         mode=mode,
                         smoothRadius=smoothRadius)$to(device=device)

  crv <- doCrv(dtmT)

  crvA <- terra::as.array(crv$permute(c(2,3,1))$cpu())
  crvR <- terra::rast(crvA)

  terra::ext(crvR) <- terra::ext(dtm)
  terra::crs(crvR) <- terra::crs(dtm)

  if(writeRaster ==TRUE){
    terra::writeRaster(crvR, outName, overwrite=TRUE)
  }

  return(crvR)
}




# terrainVis --------------------------------------------------------------

makeTerrVisModule <- torch::nn_module(
  "lspModule",

  initialize = function(cellSize=1,
                        innerRadius=2,
                        outerRadius=5,
                        hsRadius=50) {
    #
    # 0. Store user-defined parameters
    #
    self$cellSize      <- cellSize
    self$innerRadius   <- innerRadius
    self$outerRadius   <- outerRadius
    self$hsRadius      <- hsRadius

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

    #
    # Register them as buffers (NOT as parameters)
    #
    self$kx_slope <- torch::nn_buffer(kx_init$view(c(1,1,3,3)))  # original slope kernel
    self$ky_slope <- torch::nn_buffer(ky_init$view(c(1,1,3,3)))

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

  },

  forward = function(inDTM) {

    # 1. Slope calculation

    dx <- torch::nnf_conv2d(inDTM, self$kx_slope, padding = 1)
    dy <- torch::nnf_conv2d(inDTM, self$ky_slope, padding = 1)
    dx <- dx/(8*self$cellSize)
    dy <- dy/(8*self$cellSize)
    gradMag <- torch::torch_sqrt((dx*dx)+(dy*dy))
    slp <- torch::torch_atan(gradMag)*57.2958
    slp <- torch::torch_sqrt(slp)
    slp <- torch::torch_clamp(slp, 0, 10)/(10.0)


    # 2. Local TPI

    neighborhood_sum <- torch::nnf_conv2d(inDTM, self$annulus_kernel,
                                          padding = self$outerRadius)
    neighborhood_mean <- neighborhood_sum$div(self$annulus_area)
    tpiL <- inDTM - neighborhood_mean
    tpiL <- torch::torch_clamp(tpiL, -10, 10)
    tpiL <- (tpiL + 10.0) / 20.0


    # 3. Hillslope TPI (conditional)


    hs_sum <- torch::nnf_conv2d(inDTM, self$hs_kernel, padding = self$hsRadius)
    hs_mean <- hs_sum$div(self$hs_area)
    tpiHS <- inDTM - hs_mean
    tpiHS <- torch::torch_clamp(tpiHS, -10, 10)
    tpiHS <- (tpiHS + 10.0) / 20.0


    outLSPs <- torch::torch_cat(list(tpiHS, slp, tpiL),
                                dim = 1)

    return(outLSPs)
  }
)


#' makeTerrainVisTerra
#'
#' Make three band terrain stack from input digital terrain model using torch
#'
#' This function creates a three-band raster stack from an input digital terrain
#' model (DTM) of bare earth surface elevations using torch. This implementation is
#' faster than the terra-based implementation, especially when using GPU-based computation.
#'
#' The first band is a topographic position index (TPI) calculated using a moving
#' window with a 50 m circular radius. The second band is the square root of slope
#' calculated in degrees. The third band is a TPI calculated using an annulus moving
#' window with an inner radius of 2 and outer radius of 5 meters. The TPI values are
#' clamped to a range of -10 to 10 then linearly rescaled from 0 and 1. The square
#' root of slope is clamped to a ange of 0 to 10 then linearly rescaled from 0 to 1.
#' Values are provided in floating point.
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
#' @param cellSize Resolution of the grid relative to coordinate reference system
#' units (e.g., meters).
#' @param innerRadius inner radius of annulus moving window used for local TPI calculation.
#' Default is 2.
#' @param outerRadius outer radius of annulus moving window used for local TPI calculation.
#' @param hsRadius outer radisu for circular moving window used for hillslope TPI calculation.
#' @param writeRaster TRUE or FALSE. Save output to disk. Default is TRUE.
#' @param outName Name of output raster with full file path and extension.
#' @param device Device on which to perform calculations. "cpu" or "cuda". Default is "cpu".
#' Recommend "cuda". Recommend "cuda" to speed up calculations.
#' @return Three-band raster grid written to disk in TIFF format and spatRaster object.
#' @examples
#' \dontrun{
#' pth <- "OUTPUT PATH"
#' dtm <- rast(paste0(pth, "dtm.tif"))
#' tVis <- makeTerrainVisTorch(dtm,
#' cellSize=1,
#' innerRadius=2,
#' outerRadius=5,
#' hsRadius=50,
#' writeRaster=TRUE,
#' outName=paste0(pth, "tVisTorch.tif"),
#' device="cuda")
#' }
#' @export
makeTerrainVisTorch <- function(dtm,
                                cellSize=1,
                                innerRadius = 2,
                                outerRadius = 10,
                                hsRadius = 50,
                                writeRaster=FALSE,
                                outName,
                                device="cpu"){

  dtmA <- terra::as.array(dtm)
  dtmT <- torch::torch_tensor(dtmA)$permute(c(3,1,2))$to(device=device)

  doTV <- makeTerrVisModule(cellSize=1,
                             innerRadius=innerRadius,
                             outerRadius=outerRadius,
                             hsRadius=hsRadius)$to(device=device)

  tv <- doTV(dtmT)

  tvA <- terra::as.array(tv$permute(c(2,3,1))$cpu())
  tvR <- terra::rast(tvA)

  terra::ext(tvR) <- terra::ext(dtm)
  terra::crs(tvR) <- terra::crs(dtm)

  if(writeRaster ==TRUE){
    terra::writeRaster(tvR, outName, overwrite=TRUE)
  }

  return(tvR)
}


#' makeTerrainVisTerra
#'
#' Make three band terrain stack from input digital terrain model using terra
#'
#' This function creates a three-band raster stack from an input digital terrain
#' model (DTM) of bare earth surface elevations using terra. This implementation is
#' slower than the torch-based implementation, especially when the torh-based implementation
#' uses GPU-based computation.
#'
#' The first band is a topographic position index (TPI) calculated using a moving
#' window with a 50 m circular radius. The second band is the square root of slope
#' calculated in degrees. The third band is a TPI calculated using an annulus moving
#' window with an inner radius of 2 and outer radius of 5 meters. The TPI values are
#' clamped to a range of -10 to 10 then linearly rescaled from 0 and 1. The square
#' root of slope is clamped to a ange of 0 to 10 then linearly rescaled from 0 to 1.
#' Values are provided in floating point.
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
#' @param cellSize Resolution of the grid relative to coordinate reference system
#' units (e.g., meters).
#' @param innerRadius inner radius of annulus moving window used for local TPI calculation.
#' Default is 2.
#' @param outerRadius outer radius of annulus moving window used for local TPI calculation.
#' @param hsRadius outer radisu for circular moving window used for hillslope TPI calculation.
#' @param writeRaster TRUE or FALSE. Save output to disk. Default is TRUE.
#' @param outName Name of output raster with full file path and extension.
#' @return Three-band raster grid written to disk in TIFF format and spatRaster object.
#' @examples
#' \dontrun{
#' pth <- "OUTPUT PATH"
#' dtm <- rast(paste0(pth, "dtm.tif"))
#' tVisT <- makeTerrainVisTerra(dtm,
#' cellSize=1,
#' innerRadius=2,
#' outerRadius=5,
#' hsRadius=50,
#' writeRaster=TRUE,
#' outName=paste0(pth, "tVisTerra.tif"))
#' }
#' @export
makeTerrainVisTerra <- function(dtm,
                                cellSize,
                                innerRadius=2,
                                outerRadius=10,
                                hsRadius=50,
                                writeRaster=FALSE,
                                outName){
  slp <- terra::terrain(dtm, v = "slope", neighbors = 8, unit = "degrees")
  slp1 <- sqrt(slp)
  slp2 <- terra::clamp(slp1, 0, 10, values = TRUE)
  slp3 <- slp2/(10)
  message("Completed Slope.")

  fAnnulus <- MultiscaleDTM::annulus_window(radius = c(innerRadius,outerRadius), unit = "map", resolution = cellSize)
  fCircle <- MultiscaleDTM::circle_window(radius = c(hsRadius), unit = "map", resolution = cellSize)
  tpiC <- dtm - terra::focal(dtm, fCircle, "mean", na.rm = TRUE)
  tpiA <- dtm - terra::focal(dtm, fAnnulus, "mean", na.rm = TRUE)
  tpiC2 <- terra::clamp(tpiC, -10, 10, values = TRUE)
  tpiA2 <- terra::clamp(tpiA, -10, 10, values = TRUE)
  tpiC3 <- (tpiC2 - (-10))/(10-(-10))
  tpiA3 <- (tpiA2 - (-10))/(10-(-10))
  message("Completed TPIs.")

  stackOut <- c(tpiC3, slp3, tpiA3)
  names(stackOut) <- c("tpi1", "sqrtslp", "tpi2")

  if(writeRaster==TRUE){
    terra::writeRaster(stackOut, filename, overwrite = TRUE)
  }

  return(stackOut)

  message("Results stacked and written to disk.")
}
