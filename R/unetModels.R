#Squeeze and excitation module
seModule <- torch::nn_module(
  initialize = function(inChn, ratio = 8) {
    self$avg_pool <- torch::nn_adaptive_avg_pool2d(1)
    self$seMod <- torch::nn_sequential(
      torch::nn_linear(inChn, inChn %/% ratio, bias = FALSE),
      torch::nn_relu(inplace = TRUE),
      torch::nn_linear(inChn %/% ratio, inChn, bias = FALSE),
      torch::nn_sigmoid()
    )
  },

  forward = function(inputs) {
    b <- dim(inputs)[1]
    c <- dim(inputs)[2]
    x <- self$avg_pool(inputs)
    x <- x$view(c(b, -1))
    x <- self$seMod(x)
    x <- x$view(c(b, c, 1, 1))
    x <- inputs * x
    return(x)
  }
)


#Use 1x1 2D convolution to change the number of feature maps
featReduce <- torch::nn_module(
  initialize = function(inChn,
                         outChn,
                         actFunc="relu",
                         negative_slope=0.01){
    self$conv1_1 <- torch::nn_sequential(
      torch::nn_conv2d(inChn,
                       outChn,
                       kernel_size=c(1,1),
                       stride=1,
                       padding=0),
      torch::nn_batch_norm2d(outChn),
      if(actFunc == "lrelu"){
        torch::nn_leaky_relu(inplace=TRUE,
                             negative_slope=negative_slope)
      }else if(actFunc == "swish"){
        torch::nn_silu(inplace=TRUE)
      }else{
        torch::nn_relu(inplace=TRUE)
      }
    )
},

  forward = function(x){
    xx <- self$conv1_1(x)
    return(xx)
    }
)

#Use transpose convolution to upsample tensors in the decoder
upConvBlk <- torch::nn_module(
  initialize = function(inChn,
                         outChn,
                         actFunc="relu",
                         negative_slope=0.01){

    self$upConv <- torch::nn_sequential(
      torch::nn_conv_transpose2d(inChn,
                                 outChn,
                                 kernel_size=c(2,2),
                                 stride=2),
      torch::nn_batch_norm2d(outChn),
    if(actFunc == "lrelu"){
      torch::nn_leaky_relu(inplace=TRUE,
                           negative_slope=negative_slope)
    }else if(actFunc == "swish"){
      torch::nn_silu(inplace=TRUE)
    }else{
      torch::nn_relu(inplace=TRUE)
    }
    )
  },

  forward = function(x){
    xx <- self$upConv(x)
    return(xx)
  }
)


#Used to created feature maps in the encoder and decoder
doubleConvBlk <- torch::nn_module(
  initialize = function(inChn,
                         outChn,
                         actFunc="relu",
                         negative_slope=0.01){

    self$dConv <- torch::nn_sequential(
      torch::nn_conv2d(inChn,
                       outChn,
                       kernel_size=c(3,3),
                       stride=1,
                       padding=1),
      torch::nn_batch_norm2d(outChn),
      if(actFunc == "lrelu"){
        torch::nn_leaky_relu(inplace=TRUE,
                             negative_slope=negative_slope)
      }else if(actFunc == "swish"){
        torch::nn_silu(inplace=TRUE)
      }else{
        torch::nn_relu(inplace=TRUE)
      },
      torch::nn_conv2d(outChn,
                       outChn,
                       kernel_size=c(3,3),
                       stride=1,
                       padding=1),
      torch::nn_batch_norm2d(outChn),
      if(actFunc == "lrelu"){
        torch::nn_leaky_relu(inplace=TRUE,
                             negative_slope=negative_slope)
      }else if(actFunc == "swish"){
        torch::nn_silu(inplace=TRUE)
      }else{
        torch::nn_relu(inplace=TRUE)
      }
      )
  },

  forward = function(x){
    xx <- self$dConv(x)
    return(xx)
  }
)


#Used to created feature maps in the encoder and decoder
#Includes residual connection
doubleConvBlkR <- torch::nn_module(
  initialize = function(inChn,
                         outChn,
                         actFunc="relu",
                         negative_slope=0.01){

    self$dConv <- torch::nn_sequential(
      torch::nn_conv2d(inChn,
                       outChn,
                       kernel_size=c(3,3),
                       stride=1,
                       padding=1),
      torch::nn_batch_norm2d(outChn),
      if(actFunc == "lrelu"){
        torch::nn_leaky_relu(inplace=TRUE,
                             negative_slope=negative_slope)
      }else if(actFunc == "swish"){
        torch::nn_silu(inplace=TRUE)
      }else{
        torch::nn_relu(inplace=TRUE)
      },
      torch::nn_conv2d(outChn,
                       outChn,
                       kernel_size=c(3,3),
                       stride=1,
                       padding=1),
      torch::nn_batch_norm2d(outChn),
      if(actFunc == "lrelu"){
        torch::nn_leaky_relu(inplace=TRUE,
                             negative_slope=negative_slope)
      }else if(actFunc == "swish"){
        torch::nn_silu(inplace=TRUE)
      }else{
        torch::nn_relu(inplace=TRUE)
      }
    )

    self$skipPath <- featReduce(inChn=inChn,
                                outChn=outChn,
                                actFunc=actFunc,
                                negative_slope = negative_slope)
  },

  forward = function(x){
    xx <- self$dConv(x)
    xSP <- self$skipPath(x)
    return(xx + xSP)
  }
)


upSamp <- torch::nn_module(
  initialize = function(scale_factor,
                        mode = "bilinear",
                        align_corners = FALSE) {
    self$scale_factor = scale_factor
    self$mode = mode
    self$align_corners = align_corners
  },

  forward = function(x) {
    torch::nnf_interpolate(x,
                           scale_factor = self$scale_factor,
                           mode = self$mode,
                           align_corners = self$align_corners)
  }
)

#Attention mechanism
attnBlk <- torch::nn_module(
  #Ahttps://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
  #https://www.youtube.com/watch?v=KOF38xAvo8I
  initialize = function(xChn, gChn) {

    self$W_gate <- torch::nn_sequential(
      torch::nn_conv2d(gChn, xChn ,
                       kernel_size = c(1,1),
                       stride = 1,
                       padding = 0,
                       bias = TRUE),
      torch::nn_batch_norm2d(xChn)
    )

    self$W_x <- torch::nn_sequential(
      torch::nn_conv2d(xChn,
                       xChn,
                       kernel_size = c(1,1),
                       stride = 2,
                       padding = 0,
                       bias = TRUE),
      torch::nn_batch_norm2d(xChn)
    )

    self$psi <- torch::nn_sequential(
      torch::nn_conv2d(xChn,
                       1,
                       kernel_size = c(1,1),
                       stride = 1,
                       padding = 0,
                       bias = TRUE),
      torch::nn_batch_norm2d(1),
      torch::nn_sigmoid(),
      upSamp(scale_factor=2,
             mode="bilinear",
             align_corners = FALSE)
    )
  },

  forward = function(scIn, gateIn){
    g1 <- self$W_gate(gateIn)
    x1 <- self$W_x(scIn)
    psi <- torch::nnf_relu(g1 + x1, inplace=FALSE)
    psi <- self$psi(psi)
    out <- scIn * psi
    return(out)
  }
)

#Classification head
classifierBlk <- torch::nn_module(
  initialize = function(inChn, nCls){
    self$classifier <- torch::nn_conv2d(inChn,
                                     nCls,
                                     kernel_size=c(1,1),
                                     stride=1,
                                     padding=0)
  },

  forward = function(x){
    xx <- self$classifier(x)

    return(xx)
  }
)

#Define bottleneck component
bottleneck <- torch::nn_module(
  initialize = function(inChn,
                         outChn = 256,
                         actFunc = "relu",
                         negative_slope = 0.01){

      self$btnk <- doubleConvBlk(inChn=inChn,
                                 outChn=outChn,
                                 actFunc=actFunc,
                                 negative_slope=negative_slope)
  },

  forward = function(x){
    xb <- self$btnk(x)
    return(xb)
  }
)

#Define bottleneck component
#includes residual connections
bottleneckR <- torch::nn_module(
  initialize = function(inChn,
                         outChn = 256,
                         actFunc = "relu",
                         negative_slope = 0.01){

    self$btnk <- doubleConvBlk(inChn=inChn,
                               outChn=outChn,
                               actFunc=actFunc,
                               negative_slope=negative_slope)

    self$skipPath <- featReduce(inChn=inChn,
                                outChn=outChn,
                                actFunc=actFunc,
                                negative_slope = negative_slope)
  },

  forward = function(x){
    xb <- self$btnk(x)
    xSC <- self$skipPath(x)
    return(xb+xSC)
  }
)


#Atrous Spatial Pyramid Pooling (ASPP) for use in bottleneck
asppComp <- torch::nn_module(

  initialize = function(inChn,
                        outChn,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        actFunc="relu",
                        negative_slope=negative_slope){

    self$aspp <-  torch::nn_sequential(
      torch::nn_conv2d(inChn,
                       outChn,
                       kernel_size,
                       stride,
                       padding,
                       dilation),
      torch::nn_batch_norm2d(outChn),
      if(actFunc == "l
         relu"){
        torch::nn_leaky_relu(inplace=TRUE,
                             negative_slope=negative_slope)
      }else if(actFunc == "swish"){
        torch::nn_silu(inplace=TRUE)
      }else{
        torch::nn_relu(inplace=TRUE)
      }
      )
  },

  forward = function(x){
    xx <- self$aspp(x)
    return(xx)
  }
)

#Combine ASPP components to create ASPP module
asppBlk <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        dilChn=c(16,16,16,16,16),
                        dilRates=c(1,2,4,8,16),
                        actFunc="relu",
                        negative_slope=0.01){

    self$a1 <-asppComp(inChn=inChn,
                       outChn=dilChn[1],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[1],
                       dilation = dilRates[1],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a2 <-asppComp(inChn=inChn,
                       outChn=dilChn[2],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[2],
                       dilation = dilRates[2],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a3 <-asppComp(inChn=inChn,
                       outChn=dilChn[3],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[3],
                       dilation = dilRates[3],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a4 <-asppComp(inChn=inChn,
                       outChn=dilChn[4],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[4],
                       dilation = dilRates[4],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a5 <-asppComp(inChn=inChn,
                       outChn=dilChn[5],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[5],
                       dilation = dilRates[5],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$conv1_1 <- featReduce(dilChn[1]+dilChn[2]+dilChn[3]+dilChn[4]+dilChn[5],
                               outChn=outChn,
                               actFunc=actFunc,
                               negative_slope=negative_slope)

  },

  forward = function(x){
    x1 <- self$a1(x)
    x2 <- self$a2(x)
    x3 <- self$a3(x)
    x4 <- self$a4(x)
    x5 <- self$a5(x)
    xx <- torch::torch_cat(list(x1,x2,x3,x4,x5), dim=2)
    xx <- self$conv1_1(xx)

    return(xx)

  }
)




#Combine ASPP components to create ASPP module
#includes residual connection
asppBlkR <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        dilChn=c(16,16,16,16,16),
                        dilRates=c(1,2,4,8,16),
                        actFunc="relu",
                        negative_slope=0.01){

    self$a1 <-asppComp(inChn=inChn,
                       outChn=dilChn[1],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[1],
                       dilation = dilRates[1],
                       actFunc="relu",
                       negative_slope=negative_slope)

    self$a2 <-asppComp(inChn=inChn,
                       outChn=dilChn[2],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[2],
                       dilation = dilRates[2],
                       actFunc="relu",
                       negative_slope=negative_slope)

    self$a3 <-asppComp(inChn=inChn,
                       outChn=dilChn[3],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[3],
                       dilation = dilRates[3],
                       actFunc="relu",
                       negative_slope=negative_slope)

    self$a4 <-asppComp(inChn=inChn,
                       outChn=dilChn[4],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[4],
                       dilation = dilRates[4],
                       actFunc="relu",
                       negative_slope=negative_slope)

    self$a5 <-asppComp(inChn=inChn,
                       outChn=dilChn[5],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[5],
                       dilation = dilRates[5],
                       actFunc="relu",
                       negative_slope=negative_slope)

    self$conv1_1 <- featReduce(dilChn[1]+dilChn[2]+dilChn[3]+dilChn[4]+dilChn[5],
                               outChn=outChn,
                               actFunc=actFunc,
                               negative_slope=negative_slope)

    self$skipPath <- featReduce(inChn=inChn,
                                outChn=outChn,
                                actFunc=actFunc,
                                negative_slope = negative_slope)

  },

  forward = function(x){
    x1 <- self$a1(x)
    x2 <- self$a2(x)
    x3 <- self$a3(x)
    x4 <- self$a4(x)
    x5 <- self$a5(x)
    xx <- torch::torch_cat(list(x1,x2,x3,x4,x5), dim=2)
    xx <- self$conv1_1(xx)

    xSC <- self$skipPath(x)

    return(xx+xSC)

  }
)


#' defineUNet
#'
#' Define a UNet architecture for geospatial semantic segmentation.
#'
#' Define a UNet architecture with 4 blocks in the encoder, a bottleneck
#' block, and 4 blocks in the decoder. UNet can accept a variable number of input
#' channels, and the user can define the number of feature maps produced in each
#' encoder and decoder block and the bottleneck. Users can also choose to (1) replace
#' all ReLU activation functions with leaky ReLU or swish, (2) implement attention
#' gates along the skip connections, (3) implement squeeze and excitation modules within
#' the encoder blocks, (4) add residual connections within all blocks, (5) replace the
#' bottleneck with a modified atrous spatial pyramid pooling (ASPP) module, and/or (6)
#' implement deep supervision using predictions generated at each stage in the decoder.
#'
#' @param inChn Number of channels, bands, or predictor variables in the input
#' image or raster data. Default is 3.
#' @param nCls Number of classes being differentiated. For a binary classification,
#' this can be either 1 or 2. If 2, the problem is treated as a multiclass problem,
#' and a multiclass loss metric should be used. Default is 3.
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
#' @param dilRates Vector of 5 values specifying the dilation rates used in the ASPP module.
#' Default is 1, 2, 4, 6, and 16.
#' @param dilChn Vector of 5 values specifying the number of channels to produce at each dilation
#' rate within the ASPP module. Default is 16 for each dilation rate or 80 channels overall.
#' @param negative_slope If actFunc = "lrelu", specifies the negative slope term
#' to use. Default is 0.01.
#' @param seRatio Ratio to use in squeeze and excitation module. The default is 8.
#' @return Unet model instance as torch nnn_module
#' @examples
#' # example code
#' #Generate example data as torch tensor
#' tensorIn <- torch::torch_rand(c(12,4,128,128))
#'
#'  #Instantiate model
#'  model <- defineUNet(inChn = 4,
#'                     nCls = 3,
#'                     actFunc = "lrelu",
#'                     useAttn = TRUE,
#'                     useSE = TRUE,
#'                     useRes = TRUE,
#'                     useASPP = TRUE,
#'                     useDS = TRUE,
#'                     enChn = c(16,32,64,128),
#'                     dcChn = c(128,64,32,16),
#'                     btnChn = 256,
#'                     dilRates=c(1,2,4,8,16),
#'                     dilChn=c(16,16,16,16,16),
#'                     negative_slope = 0.01,
#'                     seRatio=8)
#'
#'  #Predict data with model
#'  pred <- model(tensorIn)
#' @export
defineUNet <- torch::nn_module(
  "UNet",
  initialize  = function(inChn = 3,
                          nCls = 3,
                          actFunc = "relu",
                          useAttn = FALSE,
                          useSE = FALSE,
                          useRes = FALSE,
                          useASPP = FALSE,
                          useDS = FALSE,
                          enChn = c(16,32,64,128),
                          dcChn = c(128,64,32,16),
                          btnChn = 256,
                          dilRates=c(1,2,4,8,16),
                          dilChn=c(16,16,16,16,16),
                          negative_slope = 0.01,
                          seRatio=8){

    self$inChn = inChn
    self$nCls = nCls
    self$actFunc = actFunc
    self$useAttn = useAttn
    self$useSE = useSE
    self$useRes = useRes
    self$useASPP = useASPP
    self$useDS = useDS
    self$enChn = enChn
    self$dcChn = dcChn
    self$btnChn = btnChn
    self$dilRates = dilRates
    self$dilChn = dilChn
    self$negative_slope = negative_slope
    self$seRatio = seRatio

    if(useRes == TRUE){
      self$e1 <- geodl:::doubleConvBlkR(inChn=inChn,
                                outChn=enChn[1],
                                actFunc=actFunc,
                                negative_slope=negative_slope)
      self$e2 <- geodl:::doubleConvBlkR(inChn=enChn[1],
                                outChn=enChn[2],
                                actFunc=actFunc,
                                negative_slope=negative_slope)
      self$e3 <- geodl:::doubleConvBlkR(inChn=enChn[2],
                                outChn=enChn[3],
                                actFunc=actFunc,
                                negative_slope=negative_slope)
      self$e4 <- geodl:::doubleConvBlkR(inChn=enChn[3],
                                outChn=enChn[4],
                                actFunc=actFunc,
                                negative_slope=negative_slope)

      self$dUp1 <- geodl:::upConvBlk(inChn=btnChn,
                             outChn=btnChn)
      self$dUp2 <- geodl:::upConvBlk(inChn=dcChn[1],
                             outChn=dcChn[1])
      self$dUp3 <- geodl:::upConvBlk(inChn=dcChn[2],
                             outChn=dcChn[2])
      self$dUp4 <- geodl:::upConvBlk(inChn=dcChn[3],
                             outChn=dcChn[3])
      self$d1 <- geodl:::doubleConvBlkR(inChn=btnChn+enChn[4],
                                outChn=dcChn[1],
                                actFunc=actFunc,
                                negative_slope=negative_slope)
      self$d2 <- geodl:::doubleConvBlkR(inChn=dcChn[1]+enChn[3],
                                outChn=dcChn[2],
                                actFunc=actFunc,
                                negative_slope=negative_slope)
      self$d3 <- geodl:::doubleConvBlkR(inChn=dcChn[2]+enChn[2],
                                outChn=dcChn[3],
                                actFunc=actFunc,
                                negative_slope=negative_slope)
      self$d4 <- geodl:::doubleConvBlkR(inChn=dcChn[3]+enChn[1],
                                outChn=dcChn[4],
                                actFunc=actFunc,
                                negative_slope=negative_slope)

    }else{
      self$e1 <- geodl:::doubleConvBlk(inChn=inChn,
                               outChn=enChn[1],
                               actFunc=actFunc,
                               negative_slope=negative_slope)
      self$e2 <- geodl:::doubleConvBlk(inChn=enChn[1],
                               outChn=enChn[2],
                               actFunc=actFunc,
                               negative_slope=negative_slope)
      self$e3 <- geodl:::doubleConvBlk(inChn=enChn[2],
                               outChn=enChn[3],
                               actFunc=actFunc,
                               negative_slope=negative_slope)
      self$e4 <- geodl:::doubleConvBlk(inChn=enChn[3],
                               outChn=enChn[4],
                               actFunc=actFunc,
                               negative_slope=negative_slope)

      self$dUp1 <- geodl:::upConvBlk(inChn=btnChn,
                             outChn=btnChn)
      self$dUp2 <- geodl:::upConvBlk(inChn=dcChn[1],
                             outChn=dcChn[1])
      self$dUp3 <- geodl:::upConvBlk(inChn=dcChn[2],
                             outChn=dcChn[2])
      self$dUp4 <- geodl:::upConvBlk(inChn=dcChn[3],
                             outChn=dcChn[3])
      self$d1 <- geodl:::doubleConvBlk(inChn=btnChn+enChn[4],
                               outChn=dcChn[1],
                               actFunc=actFunc,
                               negative_slope=negative_slope)
      self$d2 <- geodl:::doubleConvBlk(inChn=dcChn[1]+enChn[3],
                               outChn=dcChn[2],
                               actFunc=actFunc,
                               negative_slope=negative_slope)
      self$d3 <- geodl:::doubleConvBlk(inChn=dcChn[2]+enChn[2],
                               outChn=dcChn[3],
                               actFunc=actFunc,
                               negative_slope=negative_slope)
      self$d4 <- geodl:::doubleConvBlk(inChn=dcChn[3]+enChn[1],
                               outChn=dcChn[4],
                               actFunc=actFunc,
                               negative_slope=negative_slope)
    }

    if(useASPP == FALSE & useRes == FALSE){
      self$btn <-geodl::: bottleneck(inChn=enChn[4],
                             outChn=btnChn,
                             actFunc=actFunc,
                             negative_slope=negative_slope)
    }else if(useASPP == FALSE & useRes == TRUE){
      self$btn <- geodl:::bottleneckR(inChn=enChn[4],
                              outChn=btnChn,
                              actFunc=actFunc,
                              negative_slope=negative_slope)
    }else if(useASPP == TRUE & useRes == FALSE){
      self$btn <- geodl:::asppBlk(inChn=enChn[4],
                          outChn=btnChn,
                          dilChn=dilChn,
                          dilRates=dilRates,
                          actFunc=actFunc,
                          negative_slope=negative_slope)
    }else{
      self$btn <- geodl:::asppBlkR(inChn=enChn[4],
                           outChn=btnChn,
                           dilChn=dilChn,
                           dilRates=dilRates,
                           actFunc=actFunc,
                           negative_slope=negative_slope)
    }

    if(useSE == TRUE){
      self$se1 <- geodl:::seModule(inChn=enChn[1],
                           ratio=seRatio)
      self$se2 <- geodl:::seModule(inChn=enChn[2],
                           ratio=seRatio)
      self$se3 <- geodl:::seModule(inChn=enChn[3],
                           ratio=seRatio)
      self$se4 <- geodl:::seModule(inChn=enChn[4],
                           ratio=seRatio)
    }

    if(useAttn == TRUE){
      self$ag1 <- geodl:::attnBlk(enChn[1], dcChn[3])
      self$ag2 <- geodl:::attnBlk(enChn[2], dcChn[2])
      self$ag3 <- geodl:::attnBlk(enChn[3], dcChn[1])
      self$ag4 <- geodl:::attnBlk(enChn[4], btnChn)
    }

    self$c4 <- geodl:::classifierBlk(inChn=dcChn[4],
                             nCls=nCls)

    if(useDS == TRUE){
      self$upSamp2 <- torch::nn_upsample(scale_factor=2,
                                        mode="bilinear",
                                        align_corners=TRUE)
      self$upSamp4 <- torch::nn_upsample(scale_factor=4,
                                        mode="bilinear",
                                        align_corners=TRUE)
      self$upSamp8 <- torch::nn_upsample(scale_factor=8,
                                        mode="bilinear",
                                        align_corners=TRUE)
      self$c3 <- geodl:::classifierBlk(inChn=dcChn[3],
                               nCls=nCls)
      self$c2 <- geodl:::classifierBlk(inChn=dcChn[2],
                               nCls=nCls)
      self$c1 <- geodl:::classifierBlk(inChn=dcChn[1],
                               nCls=nCls)
    }

  },

  forward = function(x){

    e1x <- self$e1(x)
    if(self$useSE == TRUE){
      e1x <- self$se1(e1x)
    }

    e1xMP <- torch::nnf_max_pool2d(e1x,
                                   kernel_size=c(2,2),
                                   stride=2,
                                   padding=0)

    e2x <- self$e2(e1xMP)
    if(self$useSE == TRUE){
      e2x <- self$se2(e2x)
    }

    e2xMP <- torch::nnf_max_pool2d(e2x,
                                   kernel_size=c(2,2),
                                   stride=2,
                                   padding=0)

    e3x <- self$e3(e2xMP)
    if(self$useSE == TRUE){
      e3x <- self$se3(e3x)
    }

    e3xMP <- torch::nnf_max_pool2d(e3x,
                                   kernel_size=c(2,2),
                                   stride=2,
                                   padding=0)

    e4x <- self$e4(e3xMP)
    if(self$useSE == TRUE){
      e4x <- self$se4(e4x)
    }

    e4xMP <- torch::nnf_max_pool2d(e4x,
                                   kernel_size=c(2,2),
                                   stride=2,
                                   padding=0)

    btnx <- self$btn(e4xMP)

    if(self$useAttn == TRUE){
      e4x <- self$ag4(e4x, btnx)
    }
    d1Upx <- self$dUp1(btnx)
    d1Cat <- torch::torch_cat(list(d1Upx, e4x), dim=2)
    d1x <- self$d1(d1Cat)

    if(self$useAttn == TRUE){
      e3x <- self$ag3(e3x, d1x)
    }
    d2Upx <- self$dUp2(d1x)
    d2Cat <- torch::torch_cat(list(d2Upx, e3x), dim=2)
    d2x <- self$d2(d2Cat)

    if(self$useAttn == TRUE){
      e2x <- self$ag2(e2x, d2x)
    }
    d3Upx <- self$dUp3(d2x)
    d3Cat <- torch::torch_cat(list(d3Upx, e2x), dim=2)
    d3x <- self$d3(d3Cat)

    if(self$useAttn == TRUE){
      e1x <- self$ag1(e1x, d3x)
    }
    d4Upx <- self$dUp4(d3x)
    d4Cat <- torch::torch_cat(list(d4Upx, e1x), dim=2)
    d4x <- self$d4(d4Cat)

    c4x <- self$c4(d4x)

    if(self$useDS == TRUE){
      d3xUp <- self$upSamp2(d3x)
      d2xUp <- self$upSamp4(d2x)
      d1xUp <- self$upSamp8(d1x)
      c3x <- self$c3(d3xUp)
      c2x <- self$c2(d2xUp)
      c1x <- self$c1(d1xUp)
      return(list(c4x, c3x, c2x, c1x))
    }else{
      return(c4x)
    }
  }
)



#' defineMobileUNet
#'
#' Define a UNet architecture for geospatial semantic segmentation with a MobileNet-v2 backbone.
#'
#' Define a UNet architecture with a MobileNet-v2 backbone or encoder. This UNet implementation was
#' inspired by a blog post by Sigrid Keydana available
#' [here](https://blogs.rstudio.com/ai/posts/2021-10-29-segmentation-torch-android/). This architecture
#' has 6 blocks in the encoder (including the bottleneck) and 5 blocks in the decoder. The user is able to implement
#' deep supervision (useDS = TRUE) and attention gates along the skip connections (useAttn = TRUE). This model
#' requires three input bands or channels.
#'
#' @param nCls Number of classes being differentiated. For a binary classification,
#' this can be either 1 or 2. If 2, the problem is treated as a multiclass problem,
#' and a multiclass loss metric should be used. Default is 3.
#' @param pretrainedEncoder TRUE or FALSE. Whether or not to initialized using pre-trained ImageNet weights for the
#' MobileNet-v2 encoder. Default is TRUE.
#' @param freezeEncoder TRUE or FALSE. Whether or not to freeze the encoder during training. The default is TRUE.
#' If TRUE, only the decoder component is trained.
#' @param actFunc Defines activation function to use throughout the network (note that MobileNet-v2 layers are
#' not impacted). "relu" = rectified linear unit (ReLU); "lrelu" = leaky ReLU; "swish" = swish. Default is "relu".
#' @param useAttn TRUE or FALSE. Whether to add attention gates along the skip connections.
#' Default is FALSE or no attention gates are added.
#' @param useDS TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four predictions are
#' made, one at each of the four largest decoder block resolutions, and the predictions are returned as a list object
#' containing the 4 predictions. If FALSE, only the final prediction at the original resolution is
#' returned. Default is FALSE or deep supervision is not implemented.
#' @param dcChn Vector of 4 integers defining the number of output feature
#' maps for each of the 4 decoder blocks. Default is 128, 64, 32, and 16.
#' @param negative_slope If actFunc = "lrelu", specifies the negative slope term
#' to use. Default is 0.01.
#' @return ModileUNet model instance as torch nn_module
#' @examples
#' #Generate example data as torch tensor
#' tensorIn <- torch::torch_rand(c(12,3,128,128))
#'
#' #Instantiate model
#' model <- defineMobileUNet(nCls = 3,
#'                           pretrainedEncoder = FALSE,
#'                           freezeEncoder = FALSE,
#'                           actFunc = "relu",
#'                           useAttn = TRUE,
#'                           useDS = TRUE,
#'                           dcChn = c(256,128,64,32,16),
#'                           negative_slope = 0.01)
#'
#' pred <- model(tensorIn)
#' @export
defineMobileUNet <- torch::nn_module(
  "MobileUNet",
  initialize  = function(nCls = 3,
                         pretrainedEncoder = TRUE,
                         freezeEncoder = TRUE,
                         actFunc = "relu",
                         useAttn = FALSE,
                         useDS = FALSE,
                         dcChn = c(256,128,64,32,16),
                         negative_slope = 0.01){

    self$nCls = nCls
    self$pretrainedEncoder = pretrainedEncoder
    self$freezeEncoder = freezeEncoder
    self$actFunc = actFunc
    self$useAttn = useAttn
    self$useDS = useDS
    self$dcChn = dcChn
    self$negative_slope = negative_slope

    self$base_model <- torchvision::model_mobilenet_v2(pretrained = pretrainedEncoder)

    self$stages <- torch::nn_module_list(list(
      torch::nn_identity(),
      self$base_model$features[1:2],
      self$base_model$features[3:4],
      self$base_model$features[5:7],
      self$base_model$features[8:14],
      self$base_model$features[15:18]
    ))

    self$e1 <- torch::nn_sequential(self$stages[[1]])

    self$e2 <- torch::nn_sequential(self$stages[[2]])

    self$e3 <- torch::nn_sequential(self$stages[[3]])

    self$e4 <- torch::nn_sequential(self$stages[[4]])

    self$e5 <-  torch::nn_sequential(self$stages[[5]])

    self$btn <- torch::nn_sequential(self$stages[[6]])

    if(freezeEncoder == TRUE){
      for (par in self$parameters) {
        par$requires_grad_(FALSE)
      }
    }

    self$dUp1 <- geodl:::upConvBlk(inChn=320,
                           outChn=320)
    self$dUp2 <- geodl:::upConvBlk(inChn=dcChn[1],
                           outChn=dcChn[1])
    self$dUp3 <- geodl:::upConvBlk(inChn=dcChn[2],
                           outChn=dcChn[2])
    self$dUp4 <- geodl:::upConvBlk(inChn=dcChn[3],
                           outChn=dcChn[3])
    self$dUp5 <- geodl:::upConvBlk(inChn=dcChn[4],
                           outChn=dcChn[4])

    self$d1 <- geodl:::doubleConvBlk(inChn=320+96,
                             outChn=dcChn[1],
                             actFunc=actFunc,
                             negative_slope=negative_slope)
    self$d2 <- geodl:::doubleConvBlk(inChn=dcChn[1]+32,
                             outChn=dcChn[2],
                             actFunc=actFunc,
                             negative_slope=negative_slope)
    self$d3 <- geodl:::doubleConvBlk(inChn=dcChn[2]+24,
                             outChn=dcChn[3],
                             actFunc=actFunc,
                             negative_slope=negative_slope)
    self$d4 <- geodl:::doubleConvBlk(inChn=dcChn[3]+16,
                             outChn=dcChn[4],
                             actFunc=actFunc,
                             negative_slope=negative_slope)
     self$d5 <- geodl:::doubleConvBlk(inChn=dcChn[4]+3,
                             outChn=dcChn[5],
                             actFunc=actFunc,
                             negative_slope=negative_slope)

  if(useAttn == TRUE){
    self$ag1 <- geodl:::attnBlk(3, dcChn[4])
    self$ag2 <- geodl:::attnBlk(16, dcChn[3])
    self$ag3 <- geodl:::attnBlk(24, dcChn[2])
    self$ag4 <- geodl:::attnBlk(32, dcChn[1])
    self$ag5 <- geodl:::attnBlk(96, 320)
  }

  self$c4 <- geodl:::classifierBlk(inChn=dcChn[5],
                           nCls=nCls)

  if(useDS == TRUE){
    self$upSamp2 <- torch::nn_upsample(scale_factor=2,
                                      mode="bilinear",
                                      align_corners=TRUE)
    self$upSamp4 <- torch::nn_upsample(scale_factor=4,
                                      mode="bilinear",
                                      align_corners=TRUE)
    self$upSamp8 <- torch::nn_upsample(scale_factor=8,
                                      mode="bilinear",
                                      align_corners=TRUE)
    self$c3 <- geodl:::classifierBlk(inChn=dcChn[4],
                             nCls=nCls)
    self$c2 <- geodl:::classifierBlk(inChn=dcChn[3],
                             nCls=nCls)
    self$c1 <- geodl:::classifierBlk(inChn=dcChn[2],
                             nCls=nCls)
  }

  },

  forward = function(x){

    e1x <- self$e1(x)
    e2x <- self$e2(e1x)
    e3x <- self$e3(e2x)
    e4x <- self$e4(e3x)
    e5x <- self$e5(e4x)
    btnx <- self$btn(e5x)

    if(self$useAttn == TRUE){
      e5x <- self$ag5(e5x, btnx)
    }
    d1Upx <- self$dUp1(btnx)
    d1Cat <- torch::torch_cat(list(d1Upx, e5x), dim=2)
    d1x <- self$d1(d1Cat)

    if(self$useAttn == TRUE){
      e4x <- self$ag4(e4x, d1x)
    }
    d2Upx <- self$dUp2(d1x)
    d2Cat <- torch::torch_cat(list(d2Upx, e4x), dim=2)
    d2x <- self$d2(d2Cat)

    if(self$useAttn == TRUE){
      e3x <- self$ag3(e3x, d2x)
    }
    d3Upx <- self$dUp3(d2x)
    d3Cat <- torch::torch_cat(list(d3Upx, e3x), dim=2)
    d3x <- self$d3(d3Cat)

    if(self$useAttn == TRUE){
      e2x <- self$ag2(e2x, d3x)
    }
    d4Upx <- self$dUp4(d3x)
    d4Cat <- torch::torch_cat(list(d4Upx, e2x), dim=2)
    d4x <- self$d4(d4Cat)

    if(self$useAttn == TRUE){
      e1x <- self$ag1(e1x, d4x)
    }

    d5Upx <- self$dUp5(d4x)
    d5Cat <- torch::torch_cat(list(d5Upx, e1x), dim=2)
    d5x <- self$d5(d5Cat)

    c4x <- self$c4(d5x)

    if(self$useDS == TRUE){
      d4xUp <- self$upSamp2(d4x)
      d3xUp <- self$upSamp4(d3x)
      d2xUp <- self$upSamp8(d2x)
      c3x <- self$c3(d4xUp)
      c2x <- self$c2(d3xUp)
      c1x <- self$c1(d2xUp)
      return(list(pred1 = c4x,
                  pred2 = c3x,
                  pred4 = c2x,
                  pred8 = c1x)
      )
    }else{
      return(c4x)
    }
  }
)
