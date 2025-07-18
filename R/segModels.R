#Squeeze and excitation module
seModule <- torch::nn_module(
  initialize = function(inChn, ratio = 8) {
    self$avg_pool <- torch::nn_adaptive_avg_pool2d(1)
    self$seMod <- torch::nn_sequential(
      torch::nn_linear(inChn, inChn %/% ratio, bias = FALSE),
      torch::nn_relu(inplace = TRUE),
      torch::nn_linear(inChn %/% ratio, inChn, bias = TRUE),
      torch::nn_sigmoid()
    )
  },

  forward = function(inputs) {
    b <- dim(inputs)[1]
    c1 <- dim(inputs)[2]
    x <- self$avg_pool(inputs)
    x <- x$view(c(b, -1))
    x <- self$seMod(x)
    x <- x$view(c(b, c1, 1, 1))
    x <- inputs * x
    return(x)
  }
)


simpleConvBlk <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        actFunc="relu",
                        negative_slope=0.01){

    self$conv3_3 <- torch::nn_sequential(
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
      })
  },

  forward = function(x){
    x <- self$conv3_3(x)
    return(x)
  }
)



#Use 1x1 2D convolution to change the number of feature maps
featReduce <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        actFunc="relu",
                        dobnAct=TRUE,
                        negative_slope=0.01){

    self$dobnAct <- dobnAct

    self$conv1_1 <- torch::nn_sequential(
      torch::nn_conv2d(inChn,
                       outChn,
                       kernel_size=c(1,1),
                       stride=1,
                       padding=0)
      )

    if(dobnAct == TRUE){
      self$bnAct <- torch::nn_sequential(
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
    }

},

  forward = function(x){
    xx <- self$conv1_1(x)

    if(self$dobnAct == TRUE){
     return(self$bnAct(xx))
    }else{
      return(xx)
    }
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
      torch::nn_batch_norm2d(outChn)
    )

    self$skipPath <- featReduce(inChn=inChn,
                                outChn=outChn,
                                actFunc=actFunc,
                                dobnAct=FALSE,
                                negative_slope = negative_slope)

    self$finalAct <- torch::nn_sequential(
    if(actFunc == "lrelu"){
      torch::nn_leaky_relu(inplace=TRUE,
                           negative_slope=negative_slope)
    }else if(actFunc == "swish"){
      torch::nn_silu(inplace=TRUE)
    }else{
      torch::nn_relu(inplace=TRUE)
    })

  },

  forward = function(x){
    res <- self$dConv(x)
    skip <- self$skipPath(x)
    out = res + skip
    return(self$finalAct(out))
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
    x <- torch::nnf_interpolate(x,
                           scale_factor = self$scale_factor,
                           mode = self$mode,
                           align_corners = self$align_corners)
    return(x)
  }
)

#Define bottleneck component
interpUp <- torch::nn_module(
  classname = "interpUp",
  initialize = function(sFactor = 2){

    self$sFactor <- sFactor
  },

  forward = function(x){
    xUp <-  torch::nnf_interpolate(x, scale_factor = self$sFactor, mode = "bilinear", align_corners = TRUE)
    return(xUp)
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

    self$btnk <- doubleConvBlkR(inChn=inChn,
                               outChn=outChn,
                               actFunc=actFunc,
                               negative_slope=negative_slope)
  },

  forward = function(x){
    x <- self$btnk(x)
    return(x)
  }
)

#Atrous Spatial Pyramid Pooling (ASPP) for use in bottleneck
asppComp <- torch::nn_module(
  classname = "AsppComp",
  initialize = function(inChn,
                        outChn,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        actFunc       = "relu",
                        negative_slope = 0.01) {

    self$aspp <- torch::nn_sequential(
      torch::nn_conv2d(
        in_channels = inChn,
        out_channels = outChn,
        kernel_size  = kernel_size,
        stride       = stride,
        padding      = padding,
        dilation     = dilation
      ),
      torch::nn_batch_norm2d(outChn),
      if (actFunc == "lrelu") {
        torch::nn_leaky_relu(inplace = TRUE,
                             negative_slope = negative_slope)
      } else if (actFunc == "swish") {
        torch::nn_silu(inplace = TRUE)
      } else {
        torch::nn_relu(inplace = TRUE)
      }
    )
  },

  forward = function(x) {
    self$aspp(x)
  }
)


global_avg_pool2d <- torch::nn_module(
  classname = "GlobalAvgPool2d",
  initialize = function() {
    # No parameters needed
  },
  forward = function(x) {
    # x shape: (N, C, H, W)
    out <- torch::nnf_adaptive_avg_pool2d(x, output_size = c(1, 1))
    hw  <- dim(x)[3]
    # out shape: (N, C, 1, 1)
    out = torch::nnf_interpolate(out, size=c(hw,hw), mode="bilinear", align_corners=FALSE)
    return(out)
  }
)

#Combine ASPP components to create ASPP module
asppBlk <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        dilChn=c(256,256,256,256),
                        dilRates=c(6,12,18),
                        actFunc="relu",
                        negative_slope=0.01){

    self$a1 <- featReduce(inChn=inChn,
                          outChn=dilChn[1],
                          actFunc=actFunc,
                          negative_slope=negative_slope)

    self$a2 <-asppComp(inChn=inChn,
                       outChn=dilChn[2],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[1],
                       dilation = dilRates[1],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a3 <-asppComp(inChn=inChn,
                       outChn=dilChn[3],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[2],
                       dilation = dilRates[2],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a4 <-asppComp(inChn=inChn,
                       outChn=dilChn[4],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[3],
                       dilation = dilRates[3],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a5 <-global_avg_pool2d()

    self$conv1_1 <- featReduce(dilChn[1]+dilChn[2]+dilChn[3]+dilChn[4]+inChn,
                               outChn=outChn,
                               actFunc=actFunc,
                               dobnAct=TRUE,
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
asppBlkR <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        dilChn=c(256,256,256,256),
                        dilRates=c(6,12,18),
                        actFunc="relu",
                        negative_slope=0.01){

    self$a1 <- featReduce(inChn=inChn,
                          outChn=dilChn[1],
                          actFunc=actFunc,
                          negative_slope=negative_slope)

    self$a2 <-asppComp(inChn=inChn,
                       outChn=dilChn[2],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[1],
                       dilation = dilRates[1],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a3 <-asppComp(inChn=inChn,
                       outChn=dilChn[3],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[2],
                       dilation = dilRates[2],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a4 <-asppComp(inChn=inChn,
                       outChn=dilChn[4],
                       kernel_size=c(3,3),
                       stride=1,
                       padding = dilRates[3],
                       dilation = dilRates[3],
                       actFunc=actFunc,
                       negative_slope=negative_slope)

    self$a5 <- global_avg_pool2d()

    self$conv1_1 <- featReduce(dilChn[1]+dilChn[2]+dilChn[3]+dilChn[4]+inChn,
                               outChn=outChn,
                               actFunc=actFunc,
                               negative_slope=negative_slope)

    self$skipPath <- featReduce(inChn=inChn,
                                outChn=outChn,
                                actFunc=actFunc,
                                dobnAct=FALSE,
                                negative_slope = negative_slope)

    self$finalAct <- torch::nn_sequential(
      if(actFunc == "lrelu"){
        torch::nn_leaky_relu(inplace=TRUE,
                             negative_slope=negative_slope)
      }else if(actFunc == "swish"){
        torch::nn_silu(inplace=TRUE)
      }else{
        torch::nn_relu(inplace=TRUE)
      })


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

    xx <- xx+xSC

    return(self$finalAct(xx))
  }
)


quadConvBlkR <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        actFunc="relu",
                        doRes = FALSE,
                        negative_slope=0.01){

    self$inChn <- inChn
    self$outChn <- outChn
    self$actFunc <- actFunc
    self$doRes <- doRes
    self$negative_slope <- negative_slope

    self$qConv <- torch::nn_sequential(
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

    if(self$doRes == TRUE){
      self$skipPath <- featReduce(inChn=inChn,
                                  outChn=outChn,
                                  actFunc=actFunc,
                                  dobnAct=FALSE,
                                  negative_slope = negative_slope)

      self$finalAct <- torch::nn_sequential(
        if(actFunc == "lrelu"){
          torch::nn_leaky_relu(inplace=TRUE,
                               negative_slope=negative_slope)
        }else if(actFunc == "swish"){
          torch::nn_silu(inplace=TRUE)
        }else{
          torch::nn_relu(inplace=TRUE)
        })
    }

  },

  forward = function(x){

    if(self$doRes == TRUE){
      res <- self$qConv(x)
      skip <- self$skipPath(x)
      out <- res + skip
      return(self$finalAct(out))
    }else{
      x <- self$qConv(x)
      return(self$finalAct(x))
    }
  }
)


upSampConv <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        scale_factor,
                        mode = "bilinear",
                        align_corners = FALSE,
                        actFunc="relu",
                        negative_slope=0.01) {
    self$scale_factor = scale_factor
    self$mode = mode
    self$align_corners = align_corners
    self$actFunc = actFunc
    self$negative_slope = negative_slope

    self$Conv1_1 <- torch::nn_sequential(
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
      })
  },

  forward = function(x) {
    x <- torch::nnf_interpolate(x,
                                scale_factor = self$scale_factor,
                                mode = self$mode,
                                align_corners = self$align_corners)
    x <- self$Conv1_1(x)
    return(x)
  }
)


dwnSampConv <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        strd,
                        actFunc="relu",
                        negative_slope=0.01) {
    self$inChn = inChn
    self$outChn = outChn
    self$strd = strd
    self$actFunc = actFunc
    self$negative_slope = negative_slope

    self$strideConv <- torch::nn_sequential(
      torch::nn_conv2d(inChn,
                       outChn,
                       kernel_size=c(3,3),
                       stride=strd,
                       padding=1),
      torch::nn_batch_norm2d(outChn),
      if(actFunc == "lrelu"){
        torch::nn_leaky_relu(inplace=TRUE,
                             negative_slope=negative_slope)
      }else if(actFunc == "swish"){
        torch::nn_silu(inplace=TRUE)
      }else{
        torch::nn_relu(inplace=TRUE)
      })
  },

  forward = function(x) {
    x <- self$strideConv(x)
    return(x)
  }
)

#Use transpose convolution to upsample tensors in the decoder
upConvBlkDWS <- torch::nn_module(
  initialize = function(inChn,
                        outChn,
                        actFunc="relu",
                        negative_slope=0.01){

    self$upConv <- torch::nn_sequential(
      torch::nn_conv_transpose2d(inChn,
                                 outChn,
                                 kernel_size=c(2,2),
                                 groups=inChn,
                                 stride=2),

      self$pointwise <- torch::nn_conv2d(
        in_channels = outChn,
        out_channels = outChn,
        kernel_size = 1,
        bias = TRUE
      ),

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

dwsSEMS <- torch::nn_module(
  "DepthwiseSeparableSEMS",

  initialize = function(inFMs,
                        outFMs,
                        kPerFM=1,
                        rRatio=8,
                        negative_slope=0.01) {

    self$inFMs <- inFMs
    self$outFMs <- outFMs
    self$kPerFM <- kPerFM
    self$rRatio <- rRatio
    self$negative_slope <- negative_slope


    self$cnn1 <- torch::nn_conv2d(
      in_channels = inFMs,
      out_channels = outFMs,
      kernel_size = 3,
      stride = 1,
      padding = 1,
      bias = TRUE
    )

    self$dws3 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 3,
      stride = 1,
      padding = 1,
      groups = outFMs,
      bias = TRUE
    )

    self$dws5 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 5,
      stride = 1,
      padding = 2,
      groups = outFMs,
      bias = TRUE
    )

    self$dws7 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 7,
      stride = 1,
      padding = 3,
      groups = outFMs,
      bias = TRUE
    )

    self$dws9 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 9,
      stride = 1,
      padding = 4,
      groups = outFMs,
      bias = TRUE
    )

    self$dwsD2 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 3,
      stride = 1,
      padding = 2,
      dilation=2,
      groups = outFMs,
      bias = TRUE
    )

    self$dwsD3 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 3,
      stride = 1,
      padding = 3,
      dilation=3,
      groups = outFMs,
      bias = TRUE
    )

    self$dwsD4 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 5,
      stride = 1,
      padding = 4,
      dilation=2,
      groups = outFMs,
      bias = TRUE
    )

    self$pointwise <- torch::nn_conv2d(
      in_channels = outFMs * kPerFM * 7,
      out_channels = outFMs,
      kernel_size = 1,
      bias = TRUE
    )

    self$batchnorm <- torch::nn_batch_norm2d(num_features = outFMs)

    self$activation <- torch::nn_leaky_relu(negative_slope = negative_slope,
                                            inplace = TRUE)

    # Squeeze-and-Excitation Module
    self$se_pool <- torch::nn_adaptive_avg_pool2d(output_size = 1)

    self$se_fc1 <- torch::nn_linear(outFMs, outFMs %/% rRatio, bias = FALSE)
    self$se_relu <- torch::nn_relu(inplace = TRUE)

    self$se_fc2 <- torch::nn_linear(outFMs %/% rRatio, outFMs, bias = FALSE)
    self$se_sigmoid <- torch::nn_sigmoid()
  },

  forward = function(x) {
    x <- self$cnn1(x)
    xDWS3 <- self$dws3(x)
    xDWS5 <- self$dws5(x)
    xDWS7 <- self$dws7(x)
    xDWS9 <- self$dws9(x)
    xDWSD2 <- self$dwsD2(x)
    xDWSD3 <- self$dwsD3(x)
    xDWSD4 <- self$dwsD4(x)

    x <- torch_cat(list(xDWS3,
                        xDWS5,
                        xDWS7,
                        xDWS9,
                        xDWSD2,
                        xDWSD3,
                        xDWSD4),
                   dim = 2)

    x <- self$pointwise(x)
    x <- self$batchnorm(x)
    x <- self$activation(x)

    # Squeeze-and-Excitation
    se <- self$se_pool(x)$view(c(x$size(1), -1))  # Global Avg Pool
    se <- self$se_fc1(se)
    se <- self$se_relu(se)
    se <- self$se_fc2(se)
    se <- self$se_sigmoid(se)$view(c(x$size(1), x$size(2), 1, 1))  # Reshape for channel-wise scaling

    x <- x * se  # Scale input feature maps
    return(x)
  }
)


dwsMS <- torch::nn_module(
  "DepthwiseSeparableMS",

  initialize = function(inFMs,
                        outFMs,
                        kPerFM=1,
                        negative_slope=0.01) {

    self$inFMs <- inFMs
    self$outFMs <- outFMs
    self$kPerFM <- kPerFM
    self$negative_slope <- negative_slope


    self$cnn1 <- torch::nn_conv2d(
      in_channels = inFMs,
      out_channels = outFMs,
      kernel_size = 3,
      stride = 1,
      padding = 1,
      bias = TRUE
    )

    self$dws3 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 3,
      stride = 1,
      padding = 1,
      groups = outFMs,
      bias = TRUE
    )

    self$dws5 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 5,
      stride = 1,
      padding = 2,
      groups = outFMs,
      bias = TRUE
    )

    self$dws7 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 7,
      stride = 1,
      padding = 3,
      groups = outFMs,
      bias = TRUE
    )

    self$dws9 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 9,
      stride = 1,
      padding = 4,
      groups = outFMs,
      bias = TRUE
    )

    self$dwsD2 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 3,
      stride = 1,
      padding = 2,
      dilation=2,
      groups = outFMs,
      bias = TRUE
    )

    self$dwsD3 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 3,
      stride = 1,
      padding = 3,
      dilation=3,
      groups = outFMs,
      bias = TRUE
    )

    self$dwsD4 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 5,
      stride = 1,
      padding = 4,
      dilation=2,
      groups = outFMs,
      bias = TRUE
    )

    self$pointwise <- torch::nn_conv2d(
      in_channels = outFMs * kPerFM * 7,
      out_channels = outFMs,
      kernel_size = 1,
      bias = TRUE
    )

    self$batchnorm <- torch::nn_batch_norm2d(num_features = outFMs)

    self$activation <- torch::nn_leaky_relu(negative_slope = negative_slope,
                                            inplace = TRUE)
  },

  forward = function(x) {
    x <- self$cnn1(x)
    xDWS3 <- self$dws3(x)
    xDWS5 <- self$dws5(x)
    xDWS7 <- self$dws7(x)
    xDWS9 <- self$dws9(x)
    xDWSD2 <- self$dwsD2(x)
    xDWSD3 <- self$dwsD3(x)
    xDWSD4 <- self$dwsD4(x)

    x <- torch_cat(list(xDWS3,
                        xDWS5,
                        xDWS7,
                        xDWS9,
                        xDWSD2,
                        xDWSD3,
                        xDWSD4),
                   dim = 2)

    x <- self$pointwise(x)
    x <- self$batchnorm(x)
    x <- self$activation(x)

    return(x)
  }
)



dwsSE <- torch::nn_module(
  "DepthwiseSeparableSE",

  initialize = function(inFMs,
                        outFMs,
                        kPerFM=1,
                        rRatio=8,
                        negative_slope=0.01) {

    self$inFMs <- inFMs
    self$outFMs <- outFMs
    self$kPerFM <- kPerFM
    self$rRatio <- rRatio
    self$negative_slope <- negative_slope

    self$cnn1 <- torch::nn_conv2d(
      in_channels = inFMs,
      out_channels = outFMs,
      kernel_size = 3,
      stride = 1,
      padding = 1,
      bias = TRUE
    )

    self$dws3 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 3,
      stride = 1,
      padding = 1,
      groups = outFMs,
      bias = TRUE
    )


    self$pointwise <- torch::nn_conv2d(
      in_channels = outFMs * kPerFM,
      out_channels = outFMs,
      kernel_size = 1,
      bias = TRUE
    )

    self$batchnorm <- torch::nn_batch_norm2d(num_features = outFMs)

    self$activation <- torch::nn_leaky_relu(negative_slope = negative_slope,
                                            inplace = TRUE)

    # Squeeze-and-Excitation Module
    self$se_pool <- torch::nn_adaptive_avg_pool2d(output_size = 1)

    self$se_fc1 <- torch::nn_linear(outFMs, outFMs %/% rRatio, bias = FALSE)
    self$se_relu <- torch::nn_relu(inplace = TRUE)

    self$se_fc2 <- torch::nn_linear(outFMs %/% rRatio, outFMs, bias = FALSE)
    self$se_sigmoid <- torch::nn_sigmoid()
  },

  forward = function(x) {
    x <- self$cnn1(x)
    x <- self$batchnorm(x)
    x <- self$activation(x)
    x <- self$dws3(x)
    x <- self$pointwise(x)
    x <- self$batchnorm(x)
    x <- self$activation(x)

    # Squeeze-and-Excitation
    se <- self$se_pool(x)$view(c(x$size(1), -1))  # Global Avg Pool
    se <- self$se_fc1(se)
    se <- self$se_relu(se)
    se <- self$se_fc2(se)
    se <- self$se_sigmoid(se)$view(c(x$size(1), x$size(2), 1, 1))  # Reshape for channel-wise scaling

    x <- x * se  # Scale input feature maps
    return(x)
  }
)


dws <- torch::nn_module(
  "DepthwiseSeparable",

  initialize = function(inFMs,
                        outFMs,
                        kPerFM=1,
                        negative_slope=0.01) {

    self$inFMs <- inFMs
    self$outFMs <- outFMs
    self$kPerFM <- kPerFM
    self$negative_slope <- negative_slope

    self$cnn1 <- torch::nn_conv2d(
      in_channels = inFMs,
      out_channels = outFMs,
      kernel_size = 3,
      stride = 1,
      padding = 1,
      bias = TRUE
    )

    self$dws3 <- torch::nn_conv2d(
      in_channels = outFMs,
      out_channels = outFMs * kPerFM,
      kernel_size = 3,
      stride = 1,
      padding = 1,
      groups = outFMs,
      bias = TRUE
    )


    self$pointwise <- torch::nn_conv2d(
      in_channels = outFMs * kPerFM,
      out_channels = outFMs,
      kernel_size = 1,
      bias = TRUE
    )

    self$batchnorm <- torch::nn_batch_norm2d(num_features = outFMs)

    self$activation <- torch::nn_leaky_relu(negative_slope = negative_slope,
                                            inplace = TRUE)
  },

  forward = function(x) {
    x <- self$cnn1(x)
    x <- self$batchnorm(x)
    x <- self$activation(x)
    x <- self$dws3(x)
    x <- self$pointwise(x)
    x <- self$batchnorm(x)
    x <- self$activation(x)

    return(x)
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
#' @param dilRates Vector of 3 values specifying the dilation rates used in the ASPP module.
#' Default is 6, 12, and 18.
#' @param dilChn Vector of 4 values specifying the number of channels to produce at each dilation
#' rate within the ASPP module. Default is 256 for each dilation rate.
#' @param negative_slope If actFunc = "lrelu", specifies the negative slope term
#' to use. Default is 0.01.
#' @param seRatio Ratio to use in squeeze and excitation module. The default is 8.
#' @return Unet model instance as torch nnn_module
#' @examples
#' \donttest{
#' require(torch)
#' model <- defineUNet(inChn = 4,
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
#'                     dilRates=c(6,12,18),
#'                     dilChn=c(256,256,256,256),
#'                     negative_slope = 0.01,
#'                     seRatio=8)
#'  t1 <- torch::torch_rand(c(12,4,128,128))
#'  p1 <- model(t1)
#'  }
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
                          dilRates=c(6,12,18),
                          dilChn=c(256,256,256,256),
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
#' deep supervision (useDS = TRUE) and attention gates along the skip connections (useAttn = TRUE). If ImageNet weights
#' are used and more then three predictor variables are provided, ImageNet weights in the layer of the encoder block are
#' averaged. If three channels or predictor variables are provided, the user can specify to user the ImageNet weights or
#' average them.
#'
#' #' @param inChn Number of input channels or predictor variables. Default is 3.
#' @param nCls Number of classes being differentiated. For a binary classification,
#' this can be either 1 or 2. If 2, the problem is treated as a multiclass problem,
#' and a multiclass loss metric should be used. Default is 3.
#' @param pretrainedEncoder TRUE or FALSE. Whether or not to initialized using pre-trained
#' ImageNet weights for the MobileNet-v2 encoder. Default is TRUE.
#' @param freezeEncoder TRUE or FALSE. Whether or not to freeze the encoder during training. T
#' he default is TRUE. If TRUE, only the decoder component is trained.
#' @param avgImNetWeights TRUE or FALSE. If three predictor variables are provided
#' and ImageNet weights are used, whether or not to use the original weights or average them.
#' Default is FALSE.
#' @param actFunc Defines activation function to use throughout the network (note
#' that MobileNet-v2 layers are not impacted). "relu" = rectified linear unit (ReLU);
#' "lrelu" = leaky ReLU; "swish" = swish. Default is "relu".
#' @param useAttn TRUE or FALSE. Whether to add attention gates along the skip connections.
#' Default is FALSE or no attention gates are added.
#' @param useDS TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four
#' predictions are made, one at each of the four largest decoder block resolutions, and
#' the predictions are returned as a list object containing the 4 predictions. If FALSE,
#' only the final prediction at the original resolution is returned. Default is FALSE
#' or deep supervision is not implemented.
#' @param dcChn Vector of 4 integers defining the number of output feature
#' maps for each of the 4 decoder blocks. Default is 128, 64, 32, and 16.
#' @param negative_slope If actFunc = "lrelu", specifies the negative slope term
#' to use. Default is 0.01.
#' @return ModileUNet model instance as torch nn_module
#' @examples
#' \donttest{
#' require(torch)
#' model <- defineMobileUNet(inChn = 4,
#'                           nCls = 7,
#'                           pretrainedEncoder = TRUE,
#'                           freezeEncoder = FALSE,
#'                           avgImNetWeights = TRUE,
#'                           actFunc = "relu",
#'                           useAttn = TRUE,
#'                           useDS = TRUE,
#'                           dcChn = c(256,128,64,32,16),
#'                           negative_slope = 0.01)
#' t1 <- torch::torch_rand(c(12,4,128,128))
#' p1 <- model(t1)
#' }
#' @export
defineMobileUNet <- torch::nn_module(
  "MobileUNet",

  initialize = function(inChn = 3,
                        nCls = 3,
                        pretrainedEncoder = TRUE,
                        freezeEncoder = TRUE,
                        avgImNetWeights = FALSE,
                        actFunc = "relu",
                        useAttn = FALSE,
                        useDS = FALSE,
                        dcChn = c(256,128,64,32,16),
                        negative_slope = 0.01){

    # Store settings
    self$inChn             <- inChn
    self$nCls              <- nCls
    self$pretrainedEncoder <- pretrainedEncoder
    self$freezeEncoder     <- freezeEncoder
    self$avgImNetWeights   <- avgImNetWeights
    self$actFunc           <- actFunc
    self$useAttn           <- useAttn
    self$useDS             <- useDS
    self$dcChn             <- dcChn
    self$negative_slope    <- negative_slope

    # 1) Load pretrained MobileNetV2
    self$base_model <- torchvision::model_mobilenet_v2(
      pretrained = self$pretrainedEncoder
    )

    # 2) Extract the list of feature‐blocks
    n_feats    <- length(self$base_model$features)
    orig_feats <- vector("list", n_feats)
    for (i in seq_len(n_feats)) {
      orig_feats[[i]] <- self$base_model$features[[i]]
    }

    # 3) Pull out the very first block (Conv→BN→ReLU6)
    first_block <- orig_feats[[1]]
    old_conv    <- first_block[[1]]
    orig_in     <- old_conv$in_channels  # e.g. 3

    # 4) If we need to rebuild for a different # of input channels...
    if (self$avgImNetWeights || self$inChn != orig_in) {
      # compute averaged weight
      old_w      <- old_conv$weight
      mean_w     <- old_w$mean(dim = 2, keepdim = TRUE)
      out_ch     <- old_w$size(1)
      k_h        <- old_w$size(3)
      k_w        <- old_w$size(4)
      new_in     <- self$inChn
      new_w      <- mean_w$expand(c(out_ch, new_in, k_h, k_w))

      # build new Conv2d
      new_conv <- torch::nn_conv2d(
        in_channels  = new_in,
        out_channels = out_ch,
        kernel_size  = c(k_h, k_w),
        stride       = old_conv$stride,
        padding      = old_conv$padding,
        bias         = !is.null(old_conv$bias)
      )
      new_conv$weight <- torch::nn_parameter(new_w$clone())

      # keep the old BN & ReLU6
      orig_bn    <- first_block[[2]]
      orig_relu6 <- first_block[[3]]

      # rebuild the full features Sequential
      first_block_new <- torch::nn_sequential(new_conv, orig_bn, orig_relu6)
      all_blocks      <- c(list(first_block_new), orig_feats[-1])
      self$base_model$features <- do.call(torch::nn_sequential, all_blocks)

      cat("→ First conv rebuilt: weight size is now ",
          self$base_model$features[[1]][[1]]$weight$size(), "\n")
    }

    # 5) Optionally freeze encoder
    if (self$freezeEncoder) {
      for (p in self$base_model$parameters) {
        p$requires_grad_(FALSE)
      }
    }

    # 6) Split into stages for U-Net skip connections
    self$stages <- torch::nn_module_list(list(
      torch::nn_identity(),
      self$base_model$features[1:2],
      self$base_model$features[3:4],
      self$base_model$features[5:7],
      self$base_model$features[8:14],
      self$base_model$features[15:18]
    ))
    self$e1  <- torch::nn_sequential(self$stages[[1]])
    self$e2  <- torch::nn_sequential(self$stages[[2]])
    self$e3  <- torch::nn_sequential(self$stages[[3]])
    self$e4  <- torch::nn_sequential(self$stages[[4]])
    self$e5  <- torch::nn_sequential(self$stages[[5]])
    self$btn <- torch::nn_sequential(self$stages[[6]])

    # 7) Decoder upsampling + double conv blocks
    self$dUp1 <- geodl:::upConvBlk(inChn = 320,       outChn = 320)
    self$dUp2 <- geodl:::upConvBlk(inChn = dcChn[1],   outChn = dcChn[1])
    self$dUp3 <- geodl:::upConvBlk(inChn = dcChn[2],   outChn = dcChn[2])
    self$dUp4 <- geodl:::upConvBlk(inChn = dcChn[3],   outChn = dcChn[3])
    self$dUp5 <- geodl:::upConvBlk(inChn = dcChn[4],   outChn = dcChn[4])

    # note: final skip uses self$inChn instead of hard-coded “3”
    skip1_ch <- self$inChn

    self$d1 <- geodl:::doubleConvBlk(320 + 96,  dcChn[1], actFunc, negative_slope)
    self$d2 <- geodl:::doubleConvBlk(dcChn[1] + 32, dcChn[2], actFunc, negative_slope)
    self$d3 <- geodl:::doubleConvBlk(dcChn[2] + 24, dcChn[3], actFunc, negative_slope)
    self$d4 <- geodl:::doubleConvBlk(dcChn[3] + 16, dcChn[4], actFunc, negative_slope)
    self$d5 <- geodl:::doubleConvBlk(dcChn[4] + skip1_ch, dcChn[5], actFunc, negative_slope)

    # 8) Optional attention gates
    if (useAttn) {
      self$ag1 <- geodl:::attnBlk(skip1_ch,    dcChn[4])
      self$ag2 <- geodl:::attnBlk(16,           dcChn[3])
      self$ag3 <- geodl:::attnBlk(24,           dcChn[2])
      self$ag4 <- geodl:::attnBlk(32,           dcChn[1])
      self$ag5 <- geodl:::attnBlk(96,          320)
    }

    # 9) Classifier + deep supervision
    self$c4 <- geodl:::classifierBlk(dcChn[5], nCls)
    if (useDS) {
      self$upSamp2 <- torch::nn_upsample(scale_factor=2, mode="bilinear", align_corners=TRUE)
      self$upSamp4 <- torch::nn_upsample(scale_factor=4, mode="bilinear", align_corners=TRUE)
      self$upSamp8 <- torch::nn_upsample(scale_factor=8, mode="bilinear", align_corners=TRUE)
      self$c3 <- geodl:::classifierBlk(dcChn[4], nCls)
      self$c2 <- geodl:::classifierBlk(dcChn[3], nCls)
      self$c1 <- geodl:::classifierBlk(dcChn[2], nCls)
    }
  },

  forward = function(x) {
    e1x  <- self$e1(x);   e2x  <- self$e2(e1x)
    e3x  <- self$e3(e2x); e4x  <- self$e4(e3x)
    e5x  <- self$e5(e4x); btnx <- self$btn(e5x)
    if (self$useAttn) e5x <- self$ag5(e5x, btnx)

    d1Upx <- self$dUp1(btnx)
    d1Cat <- torch::torch_cat(list(d1Upx, e5x), dim=2); d1x <- self$d1(d1Cat)

    if (self$useAttn) e4x <- self$ag4(e4x, d1x)
    d2Upx <- self$dUp2(d1x)
    d2Cat <- torch::torch_cat(list(d2Upx, e4x), dim=2); d2x <- self$d2(d2Cat)

    if (self$useAttn) e3x <- self$ag3(e3x, d2x)
    d3Upx <- self$dUp3(d2x)
    d3Cat <- torch::torch_cat(list(d3Upx, e3x), dim=2); d3x <- self$d3(d3Cat)

    if (self$useAttn) e2x <- self$ag2(e2x, d3x)
    d4Upx <- self$dUp4(d3x)
    d4Cat <- torch::torch_cat(list(d4Upx, e2x), dim=2); d4x <- self$d4(d4Cat)

    if (self$useAttn) e1x <- self$ag1(e1x, d4x)
    d5Upx <- self$dUp5(d4x)
    d5Cat <- torch::torch_cat(list(d5Upx, e1x), dim=2); d5x <- self$d5(d5Cat)

    c4x <- self$c4(d5x)
    if (self$useDS) {
      u2 <- self$upSamp2(d4x); u4 <- self$upSamp4(d3x); u8 <- self$upSamp8(d2x)
      c3x <- self$c3(u2); c2x <- self$c2(u4); c1x <- self$c1(u8)
      return(list(pred1=c4x, pred2=c3x, pred4=c2x, pred8=c1x))
    } else {
      return(c4x)
    }
  }
)





#' defineUnet3p
#'
#' Define a UNet3+ architecture for use in luz training loop.
#'
#' Define a UNet3+ architecture for use in luz training loop. User can specify the
#' number of output feature maps from each encoder and decoder block and the bottleneck
#' block. Deep supervision can also be implemented. A variable number of input predictor
#' variables and output classes can be defined.
#'
#' The architecture was inspired by:
#'
#' Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020, May.
#' Unet 3+: A full-scale connected unet for medical image segmentation. In ICASSP 2020-2020 IEEE international
#' conference on acoustics, speech and signal processing (ICASSP) (pp. 1055-1059). IEEE.
#'
#' @param inChn Number of channels, bands, or predictor variables in the input
#' image or raster data. Default is 3.
#' @param nCls Number of classes being differentiated. For a binary classification,
#' this can be either 1 or 2. If 2, the problem is treated as a multiclass problem,
#' and a multiclass loss metric should be used. Default is 3.
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
#' @param negative_slope Specifies the negative slope term for leaky ReLU activation. Default is 0.01.
#' @return UNet3+ model using nn_module().
#' @examples
#' \donttest{
#' library(torch)
#' u3pMod <- defineUNet3p(inChn=4,nCls=2,useDS=FALSE,enChn=c(16,32,64,128),dcCh=c(128,64,32,16),btnChn=256,negative_slope=0.01)$to(device="cuda")
#' t1 <- torch_rand(12,4,256,256)$to(device="cuda")
#' p1 <- u3pMod(t1)
#' }
#' @export
defineUNet3p <- torch::nn_module(
  classname = "UNet3p",

  # Define the constructor
  initialize = function(inChn=3,
                        nCls=2,
                        useDS = FALSE,
                        enChn = c(16,32,64,128),
                        dcChn = c(128,64,32,16),
                        btnChn = 256,
                        negative_slope=0.01){

    self$inChn <- inChn
    self$nCls <- nCls
    self$enChn<- enChn
    self$dcChn <- dcChn
    self$btnChn <- btnChn
    self$negative_slope <- negative_slope
    self$useDS <- useDS

    self$maxP2 <- torch::nn_max_pool2d(kernel_size = 2, stride = 2)
    self$maxP4 <- torch::nn_max_pool2d(kernel_size = 4, stride = 4)
    self$maxP8 <- torch::nn_max_pool2d(kernel_size = 8, stride = 8)

    self$up2 <- interpUp(sFactor=2)
    self$up4 <- interpUp(sFactor=4)
    self$up8 <- interpUp(sFactor=8)

    self$encoder1 <- doubleConvBlk(inChn,
                                   enChn[1],
                                   actFunc="lrelu",
                                   negative_slope=negative_slope)

    self$encoder2 <- doubleConvBlk(enChn[1],
                                   enChn[2],
                                   actFunc="lrelu",
                                   negative_slope=negative_slope)

    self$encoder3 <- doubleConvBlk(enChn[2],
                                   enChn[3],
                                   actFunc="lrelu",
                                   negative_slope=negative_slope)


    self$encoder4 <- doubleConvBlk(enChn[3],
                                   enChn[4],
                                   actFunc="lrelu",
                                   negative_slope=negative_slope)

    self$bottleneck <- doubleConvBlk(enChn[4],
                                     btnChn,
                                     actFunc="lrelu",
                                     negative_slope=negative_slope)

    self$decoder1up <- upConvBlk(btnChn,
                                 btnChn)

    self$decoder1 <- doubleConvBlk(enChn[4] + enChn[3] + enChn[2] + enChn[1] + btnChn,
                                   dcChn[1],
                                   actFunc="lrelu",
                                   negative_slope=negative_slope)

    self$decoder2up <- upConvBlk(dcChn[1],
                                 dcChn[1])

    self$decoder2 <- doubleConvBlk(enChn[3] + enChn[2] + enChn[1] + dcChn[1] + btnChn,
                                   dcChn[2],
                                   actFunc="lrelu",
                                   negative_slope=negative_slope)

    self$decoder3up <- upConvBlk(dcChn[2],
                                 dcChn[2])

    self$decoder3 <- doubleConvBlk(enChn[2] + enChn[1] + dcChn[2] + dcChn[1] + btnChn,
                                   dcChn[3],
                                   actFunc="lrelu",
                                   negative_slope=negative_slope)

    self$decoder4up <- upConvBlk(dcChn[3],
                                 dcChn[3])

    self$decoder4 <- doubleConvBlk(enChn[1] + dcChn[3] + dcChn[2],
                                   dcChn[4],
                                   actFunc="lrelu",
                                   negative_slope=negative_slope)

    self$ch4 <- classifierBlk(dcChn[4],
                              nCls=nCls)

    self$ch3 <- classifierBlk(dcChn[3],
                              nCls=nCls)

    self$ch2 <- classifierBlk(dcChn[2],
                              nCls=nCls)

    self$ch1 <- classifierBlk(dcChn[1],
                              nCls=nCls)
  },

  # Define the forward pass
  forward = function(x) {

    # Encoder main path
    e1 <- self$encoder1(x)
    e1p2 <- self$maxP2(e1)
    e1p4 <- self$maxP4(e1)
    e1p8 <- self$maxP8(e1)

    e2 <- self$encoder2(e1p2)
    e2p2 <- self$maxP2(e2)
    e2p4 <- self$maxP4(e2)

    e3 <- self$encoder3(e2p2)
    e3p2 <- self$maxP2(e3)

    e4 <- self$encoder4(e3p2)
    e4p2 <- self$maxP2(e4)

    # Bottleneck
    bOut <- self$bottleneck(e4p2)
    bUp4 <- torch::nnf_interpolate(bOut,
                                   scale_factor = 4,
                                   mode = "bilinear",
                                   align_corners = TRUE)
    bUp8 <- torch::nnf_interpolate(bOut,
                                   scale_factor = 8,
                                   mode = "bilinear",
                                   align_corners = TRUE)

    # Decoder
    d1upOut <- self$decoder1up(bOut)
    d1In <- torch::torch_cat(list(d1upOut, e4, e3p2, e2p4, e1p8), dim = 2)
    d1Out <- self$decoder1(d1In)
    d1Up4 <- torch::nnf_interpolate(d1Out,
                                    scale_factor = 4,
                                    mode = "bilinear",
                                    align_corners = TRUE)

    d2upOut <- self$decoder2up(d1Out)
    d2In <- torch::torch_cat(list(d2upOut, bUp4, e3, e2p2, e1p4), dim = 2)
    d2Out <- self$decoder2(d2In)
    d2Up4 <- torch::nnf_interpolate(d2Out,
                                    scale_factor = 4,
                                    mode = "bilinear",
                                    align_corners = TRUE)

    d3upOut <- self$decoder3up(d2Out)
    d3In <- torch::torch_cat(list(d3upOut, bUp8, d1Up4, e2, e1p2), dim = 2)
    d3Out <- self$decoder3(d3In)

    d4upOut <- self$decoder4up(d3Out)
    d4In <- torch::torch_cat(list(d4upOut, e1, d2Up4), dim = 2)
    d4Out <- self$decoder4(d4In)

    # Classifier head
    c4x <- self$ch4(d4Out)

    if(self$useDS == TRUE){
      d3xUp <- self$up2(d3Out)
      d2xUp <- self$up4(d2Out)
      d1xUp <- self$up8(d1Out)
      c3x <- self$ch3(d3xUp)
      c2x <- self$ch2(d2xUp)
      c1x <- self$ch1(d1xUp)
      return(list(pred1 = c4x,
                  pred2 = c3x,
                  pred4 = c2x,
                  pred8 = c1x))
    }else{
      return(c4x)
    }
  }
)



#' defineHRNet
#'
#' Define a modified HRNet architecture inspired by:
#'
#' Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., Liu, D., Mu, Y.,
#' Tan, M., Wang, X. and Liu, W., 2020. Deep high-resolution representation learning
#' for visual recognition. IEEE transactions on pattern analysis and machine intelligence,
#' 43(10), pp.3349-3364.
#'
#' User can specify the number of input predictor variables or channels and the number of
#' output classes. The number of feature maps to generate throughout Levels 1 through 4
#' of the encoder and the decoder block. Residual connections can  be added around convolutional
#' layers within the blocks. The user can also choose between ReLU, leak ReLU, and swish.
#'
#' @param inChn Number of input predictor variables or channels. Default is 3.
#' @param nCls Number of output classes. Default is 2
#' @param l1FMs Number of feature maps to produce throughout the Level 1 layers. Default is 32.
#' @param l2FMs Number of feature maps to produce throughout the Level 2 layers. Default is 64.
#' @param l3FMs Number of feature maps to produce throughout the Level 3 layers. Default is 128.
#' @param l4FMs Number of feature maps to produce throughout the Level 4 layers. Default is 256.
#' @param dcFMs Number of feature maps to produce throughout the decoder blocks. Default is 256.
#' @param dilRates Vector of 3 values specifying the dilation rates used in the ASPP module.
#' Default is 6, 12, and 18.
#' @param dilChn Vector of 4 values specifying the number of channels to produce at each dilation
#' rate within the ASPP module. Default is 256 for each dilation rate.
#' @param doRes TRUE or FALSE. Whether or not to include residual connections in convolution
#' blocks of the encoder. Default is TRUE.
#' @param actFunc Defines activation function to use throughout the network. "relu" = rectified
#' linear unit (ReLU); "lrelu" = leaky ReLU; "swish" = swish. Default is "relu".
#' @param negative_slope If actFunc = "lrelu", specifies the negative slope term
#' to use. Default is 0.01.
#' @return HRNet model instance as torch nn_module
#' @examples
#' \donttest{
#' library(torch)
#' hrMod <- defineHRNet(inChn=8,
#'                      nCls=2,
#'                      l1FMs=64,
#'                      l2FMs=128,
#'                      l3FMs=128,
#'                      l4FMs=128,
#'                      dcChn=256,
#'                      dilRates=c(6,12,18),
#'                      dilChn=c(128,128,128),
#'                      actFunc = "relu",
#'                      negative_slope = 0.01)$to(device="cuda")
#'
#' t1 <- torch_rand(12,8,256,256)$to(device="cuda")
#' p1 <- hrMod(t1)
#' }
#' @export
defineHRNet <- torch::nn_module(
  "MobileUNet",
  initialize  = function(inChn=3,
                         nCls = 2,
                         l1FMs = 32,
                         l2FMs = 64,
                         l3FMs = 128,
                         l4FMs = 256,
                         dcFMs = 256,
                         dilChn = c(256,256,256, 256),
                         dilRates = c(6, 12, 18),
                         doRes = TRUE,
                         actFunc = "lrelu",
                         negative_slope = 0.01){

    self$inChn = inChn
    self$nCls = nCls
    self$l1FMs = l1FMs
    self$l2FMs = l2FMs
    self$l3FMs = l3FMs
    self$l4FMs = l4FMs
    self$dilChn = dilChn
    self$dilRates = dilRates
    self$doRes = doRes
    self$actFunc = actFunc
    self$negative_slope = negative_slope

    self$s1l1a <- doubleConvBlkR(inChn=inChn,outChn=l1FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s1l1aDown2 <- dwnSampConv(inChn=l1FMs,l1FMs,strd=2,actFunc=actFunc,negative_slope=negative_slope)

    self$s1l1b <- simpleConvBlk(inChn=l1FMs,outChn=l1FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s1l2b <- simpleConvBlk(inChn=l1FMs,outChn=l2FMs,actFunc=actFunc,negative_slope=negative_slope)

    self$s2l1a <- doubleConvBlkR(inChn=l1FMs,outChn=l1FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s2l1aDown2 <- dwnSampConv(inChn=l1FMs,l1FMs,strd=2,actFunc=actFunc,negative_slope=negative_slope)
    self$s2l1aDown4 <- dwnSampConv(inChn=l1FMs,l1FMs,strd=4,actFunc=actFunc,negative_slope=negative_slope)
    self$s2l2a <- doubleConvBlkR(inChn=l2FMs,outChn=l2FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s2l2aUp2 <- upSampConv(inChn=l2FMs,outChn=l2FMs,scale_factor=2,mode = "bilinear",align_corners = FALSE,actFunc=actFunc, negative_slope=negative_slope)
    self$s2l2aDown2 <- dwnSampConv(inChn=l2FMs,l2FMs,strd=2,actFunc=actFunc,negative_slope=negative_slope)

    self$s2l1b  <- simpleConvBlk(inChn=l1FMs+l2FMs,outChn=l1FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s2l2b <- simpleConvBlk(inChn=l1FMs+l2FMs,outChn=l2FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s2l3b <- simpleConvBlk(inChn=l1FMs+l2FMs,outChn=l3FMs,actFunc=actFunc,negative_slope=negative_slope)

    self$s3l1a <- doubleConvBlkR(inChn=l1FMs,outChn=l1FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l1aDown2 <- dwnSampConv(inChn=l1FMs,l1FMs,strd=2,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l1aDown4 <- dwnSampConv(inChn=l1FMs,l1FMs,strd=4,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l1aDown8 <- dwnSampConv(inChn=l1FMs,l1FMs,strd=8,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l2a <- doubleConvBlkR(inChn=l2FMs,outChn=l2FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l2aUp2 <- upSampConv(inChn=l2FMs,outChn=l2FMs,scale_factor=2,mode = "bilinear",align_corners = FALSE,actFunc=actFunc, negative_slope=negative_slope)
    self$s3l2aDown2 <- dwnSampConv(inChn=l2FMs,l2FMs,strd=2,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l2aDown4 <- dwnSampConv(inChn=l2FMs,l2FMs,strd=4,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l3a <- doubleConvBlkR(inChn=l3FMs,outChn=l3FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l3aUp4 <- upSampConv(inChn=l3FMs,outChn=l3FMs,scale_factor=4,mode = "bilinear",align_corners = FALSE,actFunc=actFunc, negative_slope=negative_slope)
    self$s3l3aUp2 <- upSampConv(inChn=l3FMs,outChn=l3FMs,scale_factor=2,mode = "bilinear",align_corners = FALSE,actFunc=actFunc, negative_slope=negative_slope)
    self$s3l3aDown2 <- dwnSampConv(inChn=l3FMs,l3FMs,strd=2,actFunc=actFunc,negative_slope=negative_slope)

    self$s3l1b <- simpleConvBlk(inChn=l1FMs+l2FMs+l3FMs,outChn=l1FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l2b <- simpleConvBlk(inChn=l1FMs+l2FMs+l3FMs,outChn=l2FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l3b <- simpleConvBlk(inChn=l1FMs+l2FMs+l3FMs,outChn=l3FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s3l4b <- simpleConvBlk(inChn=l1FMs+l2FMs+l3FMs,outChn=l4FMs,actFunc=actFunc,negative_slope=negative_slope)

    self$s4l1a <- doubleConvBlkR(inChn=l1FMs,outChn=l1FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s4l2a<- doubleConvBlkR(inChn=l2FMs,outChn=l2FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s4l3a <- doubleConvBlkR(inChn=l3FMs,outChn=l3FMs,actFunc=actFunc,negative_slope=negative_slope)
    self$s4l4a <- asppBlk(inChn=l4FMs,outChn=l4FMs,dilRates=dilRates, dilChn=dilChn,actFunc=actFunc,negative_slope=negative_slope)

    self$s4l2aUp2 <- upSampConv(inChn=l2FMs,outChn=l2FMs,scale_factor=2,mode = "bilinear",align_corners = FALSE,actFunc=actFunc, negative_slope=negative_slope)
    self$s4l3aUp4 <- upSampConv(inChn=l3FMs,outChn=l3FMs,scale_factor=4,mode = "bilinear",align_corners = FALSE,actFunc=actFunc, negative_slope=negative_slope)
    self$s4l4aUp8 <- upSampConv(inChn=l4FMs,outChn=l4FMs,scale_factor=8,mode = "bilinear",align_corners = FALSE,actFunc=actFunc, negative_slope=negative_slope)

    self$dcBlk <- doubleConvBlkR(inChn=l1FMs+l2FMs+l3FMs+l4FMs,outChn=dcFMs,actFunc=actFunc,negative_slope=negative_slope)

    self$ch <- classifierBlk(dcFMs, nCls)

  },

  forward = function(x){

    x <- self$s1l1a(x)
    xS1L1D2 <- self$s1l1aDown2(x)

    x <- self$s1l1b(x)
    x2 <- self$s1l2b(xS1L1D2)

    x <- self$s2l1a(x)

    xS2L1D2 <- self$s2l1aDown2(x)
    xS2L1D4 <- self$s2l1aDown4(x)
    x2 <- self$s2l2a(x2)
    x2S2L2U2 <- self$s2l2aUp2(x2)
    x2S2L2D2 <- self$s2l2aDown2(x2)

    x <- torch::torch_cat(list(x, x2S2L2U2), dim=2)
    x2 <-torch::torch_cat(list(x2, xS2L1D2), dim=2)
    x3 <- torch::torch_cat(list(x2S2L2D2, xS2L1D4), dim=2)

    x <- self$s2l1b(x)
    x2 <- self$s2l2b(x2)
    x3 <- self$s2l3b(x3)

    x <- self$s3l1a(x)
    xS3L1D2 <- self$s3l1aDown2(x)
    xS3L1D4 <- self$s3l1aDown4(x)
    xS3L1D8 <- self$s3l1aDown8(x)
    x2 <- self$s3l2a(x2)
    x2S3L2U2 <- self$s3l2aUp2(x2)
    x2S3L2D2 <- self$s3l2aDown2(x2)
    x2S3L2D4 <- self$s3l2aDown4(x2)
    x3 <- self$s3l3a(x3)
    x3S3L3U4 <- self$s3l3aUp4(x3)
    x3S3L3U2 <- self$s3l3aUp2(x3)
    x3S3L3D2 <- self$s3l3aDown2(x3)

    x <- torch::torch_cat(list(x, x2S3L2U2, x3S3L3U4), dim=2)
    x2 <-torch::torch_cat(list(x2, xS3L1D2, x3S3L3U2), dim=2)
    x3 <- torch::torch_cat(list(x3, xS3L1D4, x2S3L2D2), dim=2)
    x4 <- torch::torch_cat(list(xS3L1D8, x2S3L2D4, x3S3L3D2), dim=2)

    x <- self$s3l1b(x)
    x2 <- self$s3l2b(x2)
    x3 <- self$s3l3b(x3)
    x4 <- self$s3l4b(x4)

    x <- self$s4l1a(x)
    x2 <- self$s4l2a(x2)
    x3 <- self$s4l3a(x3)
    x4 <- self$s4l4a(x4)

    x2 <- self$s4l2aUp2(x2)
    x3 <- self$s4l3aUp4(x3)
    x4 <- self$s4l4aUp8(x4)

    x <- torch::torch_cat(list(x, x2, x3, x4), dim=2)

    x <- self$dcBlk(x)

    x <- self$ch(x)

    return(x)
  })



