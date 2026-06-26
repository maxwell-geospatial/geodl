# Internal helper: collect parameters from one or more nn_module instances into a
# flat unnamed list of tensors, skipping NULL entries and modules with no params.
.collect_params <- function(...) {
  params <- list()
  for (mod in list(...)) {
    if (!is.null(mod)) {
      p <- unname(mod$parameters)
      if (length(p) > 0L) params <- c(params, p)
    }
  }
  params
}

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

#Conv block
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
        }else if(actFunc == "gelu"){
          torch::nn_gelu()
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
    }else if(actFunc == "gelu"){
      torch::nn_gelu()
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
    }else if(actFunc == "gelu"){
      torch::nn_gelu()
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


#Upsample with blinear interpolation
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

#Channel attention component of CBAM
#Pools spatially with avg-pool and max-pool, passes both through a shared MLP, then gates the input channel-wise
cBAMChannelAttn <- torch::nn_module(
  initialize = function(inChn, ratio = 8) {
    self$avg_pool <- torch::nn_adaptive_avg_pool2d(1)
    self$max_pool <- torch::nn_adaptive_max_pool2d(1)
    self$mlp <- torch::nn_sequential(
      torch::nn_linear(inChn, inChn %/% ratio, bias = FALSE),
      torch::nn_relu(inplace = TRUE),
      torch::nn_linear(inChn %/% ratio, inChn, bias = FALSE)
    )
    self$sigmoid <- torch::nn_sigmoid()
  },
  forward = function(x) {
    b  <- dim(x)[1]
    c1 <- dim(x)[2]
    avg   <- self$avg_pool(x)$view(c(b, c1))
    mx    <- self$max_pool(x)$view(c(b, c1))
    scale <- self$sigmoid(self$mlp(avg) + self$mlp(mx))$view(c(b, c1, 1L, 1L))
    return(x * scale)
  }
)

#Spatial attention component of CBAM
#Pools channel-wise with avg and max, concatenates, applies a conv+sigmoid, then gates the input spatially
cBAMSpatialAttn <- torch::nn_module(
  initialize = function(kernelSize = 7) {
    padding <- kernelSize %/% 2L
    self$conv <- torch::nn_sequential(
      torch::nn_conv2d(2L, 1L,
                       kernel_size = c(kernelSize, kernelSize),
                       stride      = 1L,
                       padding     = padding,
                       bias        = FALSE),
      torch::nn_batch_norm2d(1L),
      torch::nn_sigmoid()
    )
  },
  forward = function(x) {
    avg_map <- torch::torch_mean(x, dim = 2L, keepdim = TRUE)
    max_map <- torch::torch_amax(x, dim = 2L, keepdim = TRUE)
    cat_map <- torch::torch_cat(list(avg_map, max_map), dim = 2L)
    return(x * self$conv(cat_map))
  }
)

#Full CBAM block: channel attention followed by spatial attention
cbamBlk <- torch::nn_module(
  initialize = function(inChn, ratio = 8, kernelSize = 7) {
    self$chanAttn <- cBAMChannelAttn(inChn = inChn, ratio = ratio)
    self$spatAttn <- cBAMSpatialAttn(kernelSize = kernelSize)
  },
  forward = function(x) {
    x <- self$chanAttn(x)
    x <- self$spatAttn(x)
    return(x)
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

#Global average pooling
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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

# Block with 4 convolutions
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
        }else if(actFunc == "gelu"){
          torch::nn_gelu()
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
      }else if(actFunc == "gelu"){
        torch::nn_gelu()
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
#' rectified linear unit (ReLU); "lrelu" = leaky ReLU; "swish" = swish; "gelu" = GELU.
#' Default is "relu".
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
#' @param stageLRs Optional numeric vector of length 9 specifying a base learning
#' rate for each stage of the network: encoder stages e1, e2, e3, e4, the
#' bottleneck, and decoder stages d1, d2, d3, d4 (in that order). When provided,
#' call \code{model$get_param_groups()} to obtain a list of optimizer parameter
#' groups with per-stage learning rates. Default is NULL (single learning rate).
#' @return Unet model instance as torch nn_module
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
                          seRatio=8,
                          stageLRs = NULL){

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
    self$stageLRs = stageLRs

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
  },

  get_param_groups = function(stageLRs = self$stageLRs) {
    if (is.null(stageLRs)) {
      return(list(list(params = unname(self$parameters))))
    }
    if (length(stageLRs) != 9L) {
      stop("stageLRs must be a numeric vector of length 9 for defineUNet: ",
           "encoder stages e1, e2, e3, e4, bottleneck, decoder stages d1, d2, d3, d4")
    }
    list(
      list(params = .collect_params(self$e1, if (self$useSE) self$se1),           lr = stageLRs[1]),
      list(params = .collect_params(self$e2, if (self$useSE) self$se2),           lr = stageLRs[2]),
      list(params = .collect_params(self$e3, if (self$useSE) self$se3),           lr = stageLRs[3]),
      list(params = .collect_params(self$e4, if (self$useSE) self$se4),           lr = stageLRs[4]),
      list(params = .collect_params(self$btn),                                     lr = stageLRs[5]),
      list(params = .collect_params(self$dUp1, self$d1,
                                    if (self$useAttn) self$ag4),                  lr = stageLRs[6]),
      list(params = .collect_params(self$dUp2, self$d2,
                                    if (self$useAttn) self$ag3),                  lr = stageLRs[7]),
      list(params = .collect_params(self$dUp3, self$d3,
                                    if (self$useAttn) self$ag2),                  lr = stageLRs[8]),
      list(params = .collect_params(self$dUp4, self$d4,
                                    if (self$useAttn) self$ag1, self$c4,
                                    if (self$useDS) self$c1,
                                    if (self$useDS) self$c2,
                                    if (self$useDS) self$c3),                     lr = stageLRs[9])
    )
  },

  load_weights = function(path, encoderOnly = FALSE, freezeEncoder = FALSE) {
    state <- torch::torch_load(path)
    if (encoderOnly) {
      pfx <- c("e1.", "e2.", "e3.", "e4.")
      if (self$useSE) pfx <- c(pfx, "se1.", "se2.", "se3.", "se4.")
      keep <- vapply(names(state), function(k) any(startsWith(k, pfx)), logical(1L))
      self$load_state_dict(state[keep], strict = FALSE)
    } else {
      self$load_state_dict(state)
    }
    if (freezeEncoder) self$freeze_encoder(TRUE)
    invisible(self)
  },

  freeze_encoder = function(freeze = TRUE) {
    mods <- list(self$e1, self$e2, self$e3, self$e4)
    if (self$useSE) mods <- c(mods, list(self$se1, self$se2, self$se3, self$se4))
    for (mod in mods) for (p in mod$parameters) p$requires_grad_(!freeze)
    invisible(self)
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
#' @param inChn Number of input channels or predictor variables. Default is 3.
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
#' "lrelu" = leaky ReLU; "swish" = swish; "gelu" = GELU. Default is "relu".
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
#' @param stageLRs Optional numeric vector of length 11 specifying a base learning
#' rate for each stage: encoder stages e1, e2, e3, e4, e5, the bottleneck, and
#' decoder stages d1, d2, d3, d4, d5 (in that order). Call
#' \code{model$get_param_groups()} to obtain optimizer parameter groups with
#' per-stage learning rates. Default is NULL (single learning rate).
#' @return ModileUNet model instance as torch nn_module
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
                        negative_slope = 0.01,
                        stageLRs = NULL){

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
    self$stageLRs          <- stageLRs

    self$base_model <- torchvision::model_mobilenet_v2(
      pretrained = self$pretrainedEncoder
    )

    n_feats    <- length(self$base_model$features)
    orig_feats <- vector("list", n_feats)
    for (i in seq_len(n_feats)) {
      orig_feats[[i]] <- self$base_model$features[[i]]
    }

    first_block <- orig_feats[[1]]
    old_conv    <- first_block[[1]]
    orig_in     <- old_conv$in_channels

    if (self$avgImNetWeights || self$inChn != orig_in) {
      old_w      <- old_conv$weight
      mean_w     <- old_w$mean(dim = 2, keepdim = TRUE)
      out_ch     <- old_w$size(1)
      k_h        <- old_w$size(3)
      k_w        <- old_w$size(4)
      new_in     <- self$inChn
      new_w      <- mean_w$expand(c(out_ch, new_in, k_h, k_w))

      new_conv <- torch::nn_conv2d(
        in_channels  = new_in,
        out_channels = out_ch,
        kernel_size  = c(k_h, k_w),
        stride       = old_conv$stride,
        padding      = old_conv$padding,
        bias         = !is.null(old_conv$bias)
      )
      new_conv$weight <- torch::nn_parameter(new_w$clone())

      orig_bn    <- first_block[[2]]
      orig_relu6 <- first_block[[3]]

      first_block_new <- torch::nn_sequential(new_conv, orig_bn, orig_relu6)
      all_blocks      <- c(list(first_block_new), orig_feats[-1])
      self$base_model$features <- do.call(torch::nn_sequential, all_blocks)

      cat("First conv rebuilt: weight size is now ",
          self$base_model$features[[1]][[1]]$weight$size(), "\n")
    }

    # 5) Optionally freeze encoder
    if (self$freezeEncoder) {
      for (p in self$base_model$parameters) {
        p$requires_grad_(FALSE)
      }
    }

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

    self$dUp1 <- geodl:::upConvBlk(inChn = 320,       outChn = 320)
    self$dUp2 <- geodl:::upConvBlk(inChn = dcChn[1],   outChn = dcChn[1])
    self$dUp3 <- geodl:::upConvBlk(inChn = dcChn[2],   outChn = dcChn[2])
    self$dUp4 <- geodl:::upConvBlk(inChn = dcChn[3],   outChn = dcChn[3])
    self$dUp5 <- geodl:::upConvBlk(inChn = dcChn[4],   outChn = dcChn[4])

    skip1_ch <- self$inChn

    self$d1 <- geodl:::doubleConvBlk(320 + 96,  dcChn[1], actFunc, negative_slope)
    self$d2 <- geodl:::doubleConvBlk(dcChn[1] + 32, dcChn[2], actFunc, negative_slope)
    self$d3 <- geodl:::doubleConvBlk(dcChn[2] + 24, dcChn[3], actFunc, negative_slope)
    self$d4 <- geodl:::doubleConvBlk(dcChn[3] + 16, dcChn[4], actFunc, negative_slope)
    self$d5 <- geodl:::doubleConvBlk(dcChn[4] + skip1_ch, dcChn[5], actFunc, negative_slope)

    if (useAttn) {
      self$ag1 <- geodl:::attnBlk(skip1_ch,    dcChn[4])
      self$ag2 <- geodl:::attnBlk(16,           dcChn[3])
      self$ag3 <- geodl:::attnBlk(24,           dcChn[2])
      self$ag4 <- geodl:::attnBlk(32,           dcChn[1])
      self$ag5 <- geodl:::attnBlk(96,          320)
    }

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
  },

  get_param_groups = function(stageLRs = self$stageLRs) {
    if (is.null(stageLRs)) {
      return(list(list(params = unname(self$parameters))))
    }
    if (length(stageLRs) != 11L) {
      stop("stageLRs must be a numeric vector of length 11 for defineMobileUNet: ",
           "encoder stages e1-e5, bottleneck, decoder stages d1-d5")
    }
    list(
      list(params = .collect_params(self$e1),  lr = stageLRs[1]),
      list(params = .collect_params(self$e2),  lr = stageLRs[2]),
      list(params = .collect_params(self$e3),  lr = stageLRs[3]),
      list(params = .collect_params(self$e4),  lr = stageLRs[4]),
      list(params = .collect_params(self$e5),  lr = stageLRs[5]),
      list(params = .collect_params(self$btn), lr = stageLRs[6]),
      list(params = .collect_params(self$dUp1, self$d1,
                                    if (self$useAttn) self$ag5),                  lr = stageLRs[7]),
      list(params = .collect_params(self$dUp2, self$d2,
                                    if (self$useAttn) self$ag4),                  lr = stageLRs[8]),
      list(params = .collect_params(self$dUp3, self$d3,
                                    if (self$useAttn) self$ag3),                  lr = stageLRs[9]),
      list(params = .collect_params(self$dUp4, self$d4,
                                    if (self$useAttn) self$ag2),                  lr = stageLRs[10]),
      list(params = .collect_params(self$dUp5, self$d5,
                                    if (self$useAttn) self$ag1, self$c4,
                                    if (self$useDS) self$c1,
                                    if (self$useDS) self$c2,
                                    if (self$useDS) self$c3),                     lr = stageLRs[11])
    )
  },

  load_weights = function(path, encoderOnly = FALSE, freezeEncoder = FALSE) {
    state <- torch::torch_load(path)
    if (encoderOnly) {
      pfx  <- c("e1.", "e2.", "e3.", "e4.", "e5.")
      keep <- vapply(names(state), function(k) any(startsWith(k, pfx)), logical(1L))
      self$load_state_dict(state[keep], strict = FALSE)
    } else {
      self$load_state_dict(state)
    }
    if (freezeEncoder) self$freeze_encoder(TRUE)
    invisible(self)
  },

  freeze_encoder = function(freeze = TRUE) {
    for (mod in list(self$e1, self$e2, self$e3, self$e4, self$e5))
      for (p in mod$parameters) p$requires_grad_(!freeze)
    invisible(self)
  }
)





#' defineUnet3p
#'
#' Define a UNet3+ architecture for use in luz training loop.
#'
#' Define a UNet3+-like architecture for use in luz training loop. User can specify the
#' number of output feature maps from each encoder and decoder block and the bottleneck
#' block. Deep supervision can also be implemented. The default bottleneck block can be replaced
#' with a atrous spatial pyramid pooling module. Leaky ReLU is used throughout.
#' A variable number of input predictor variables and output classes can be defined.
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
#' @param enChn Vector of 4 integers defining the number of output
#' feature maps for each of the four encoder blocks. Default is 16, 32, 64, and 128.
#' maps for each of the 4 decoder blocks. Default is 128, 64, 32, and 16.
#' @param outChn Number of output channels for each decoder block. Default is 64.
#' @param btnChn Number of output feature maps from the bottleneck block. Default is 256.
#' @param useASPP TRUE or FALSE. Whether to use an ASPP module as the bottleneck as opposed to a
#' double convolution operation. Default is FALSE or the ASPP module is not used as the bottleneck.
#' @param dilRates Vector of 3 values specifying the dilation rates used in the ASPP module.
#' Default is 6, 12, and 18.
#' @param dilChn Vector of 4 values specifying the number of channels to produce at each dilation
#' rate within the ASPP module. Default is 256 for each dilation rate.
#' @param actFunc Activation function to use throughout. "relu" = ReLU; "lrelu" = leaky ReLU;
#' "swish" = SiLU/Swish; "gelu" = GELU. Default is "lrelu".
#' @param negative_slope Specifies the negative slope term for leaky ReLU activation. Default is 0.01.
#' @param useDS TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four predictions are
#' made, one at each decoder block resolution, and the predictions are returned as a list object
#' containing the 4 predictions. If FALSE, only the final prediction at the original resolution is
#' returned. Default is FALSE or deep supervision is not implemented.
#' @param stageLRs Optional numeric vector of length 9 specifying a base learning
#' rate for each stage: encoder stages encoder1, encoder2, encoder3, encoder4
#' (including their cross-scale projections), the bottleneck (including its
#' decoder projections), and decoder stages d1, d2, d3, d4 (in that order).
#' Call \code{model$get_param_groups()} to obtain optimizer parameter groups.
#' Default is NULL (single learning rate).
#' @return UNet3+ model using nn_module().
#' @export
defineUNet3p <- torch::nn_module(
  classname = "UNet3p",

  # Define the constructor
  initialize = function(inChn=3,
                        nCls=2,
                        actFunc="lrelu",
                        enChn = c(16,32,64,128),
                        outChn = 64,
                        btnChn = 256,
                        useASPP = FALSE,
                        dilRates=c(6,12,18),
                        dilChn=c(256,256,256,256),
                        negative_slope=0.01,
                        useDS = FALSE,
                        stageLRs = NULL){

    self$inChn <- inChn
    self$nCls <- nCls
    self$enChn <- enChn
    self$outChn <- outChn
    self$btnChn <- btnChn
    self$negative_slope <- negative_slope
    self$useDS <- useDS
    self$useASPP <- useASPP
    self$dilRates <- dilRates
    self$dilChn <- dilChn
    self$stageLRs <- stageLRs

    self$maxP2 <- torch::nn_max_pool2d(kernel_size=2, stride=2)

    self$up2 <- torch::nn_upsample(scale_factor=2,  mode="bilinear", align_corners=TRUE)
    self$up4 <- torch::nn_upsample(scale_factor=4,  mode="bilinear", align_corners=TRUE)
    self$up8 <- torch::nn_upsample(scale_factor=8,  mode="bilinear", align_corners=TRUE)

    # Encoder blocks
    self$encoder1 <- doubleConvBlk(inChn,    enChn[1], actFunc=actFunc, negative_slope=negative_slope)
    self$encoder2 <- doubleConvBlk(enChn[1], enChn[2], actFunc=actFunc, negative_slope=negative_slope)
    self$encoder3 <- doubleConvBlk(enChn[2], enChn[3], actFunc=actFunc, negative_slope=negative_slope)
    self$encoder4 <- doubleConvBlk(enChn[3], enChn[4], actFunc=actFunc, negative_slope=negative_slope)

    # Encoder -> each decoder level projections (downsample as needed)
    self$e1d4 <- simpleConvBlk(enChn[1], outChn, actFunc=actFunc, negative_slope=negative_slope)
    self$e1d3 <- torch::nn_sequential(torch::nn_max_pool2d(kernel_size=2, stride=2),
                                       simpleConvBlk(enChn[1], outChn, actFunc=actFunc, negative_slope=negative_slope))
    self$e1d2 <- torch::nn_sequential(torch::nn_max_pool2d(kernel_size=4, stride=4),
                                       simpleConvBlk(enChn[1], outChn, actFunc=actFunc, negative_slope=negative_slope))
    self$e1d1 <- torch::nn_sequential(torch::nn_max_pool2d(kernel_size=8, stride=8),
                                       simpleConvBlk(enChn[1], outChn, actFunc=actFunc, negative_slope=negative_slope))

    self$e2d3 <- simpleConvBlk(enChn[2], outChn, actFunc=actFunc, negative_slope=negative_slope)
    self$e2d2 <- torch::nn_sequential(torch::nn_max_pool2d(kernel_size=2, stride=2),
                                       simpleConvBlk(enChn[2], outChn, actFunc=actFunc, negative_slope=negative_slope))
    self$e2d1 <- torch::nn_sequential(torch::nn_max_pool2d(kernel_size=4, stride=4),
                                       simpleConvBlk(enChn[2], outChn, actFunc=actFunc, negative_slope=negative_slope))

    self$e3d2 <- simpleConvBlk(enChn[3], outChn, actFunc=actFunc, negative_slope=negative_slope)
    self$e3d1 <- torch::nn_sequential(torch::nn_max_pool2d(kernel_size=2, stride=2),
                                       simpleConvBlk(enChn[3], outChn, actFunc=actFunc, negative_slope=negative_slope))

    self$e4d1 <- simpleConvBlk(enChn[4], outChn, actFunc=actFunc, negative_slope=negative_slope)

    # Bottleneck
    if(useASPP == FALSE){
      self$btn <- doubleConvBlk(enChn[4], btnChn, actFunc=actFunc, negative_slope=negative_slope)
    }else{
      self$btn <- geodl:::asppBlkR(inChn=enChn[4], outChn=btnChn, dilChn=dilChn,
                                   dilRates=dilRates, actFunc=actFunc, negative_slope=negative_slope)
    }

    # Bottleneck -> each decoder level (upsample to match resolution)
    self$bd1 <- upConvBlk(btnChn,  outChn, actFunc=actFunc, negative_slope=negative_slope)
    self$bd2 <- torch::nn_sequential(interpUp(sFactor=4),
                                     simpleConvBlk(btnChn, outChn, actFunc=actFunc, negative_slope=negative_slope))
    self$bd3 <- torch::nn_sequential(interpUp(sFactor=8),
                                     simpleConvBlk(btnChn, outChn, actFunc=actFunc, negative_slope=negative_slope))
    self$bd4 <- torch::nn_sequential(interpUp(sFactor=16),
                                     simpleConvBlk(btnChn, outChn, actFunc=actFunc, negative_slope=negative_slope))

    # Decoder -> higher decoder level projections (upsample as needed)
    self$d1d2 <- upConvBlk(5*outChn, outChn, actFunc=actFunc, negative_slope=negative_slope)
    self$d1d3 <- torch::nn_sequential(interpUp(sFactor=4),
                                       simpleConvBlk(5*outChn, outChn, actFunc=actFunc, negative_slope=negative_slope))
    self$d1d4 <- torch::nn_sequential(interpUp(sFactor=8),
                                       simpleConvBlk(5*outChn, outChn, actFunc=actFunc, negative_slope=negative_slope))
    self$d2d3 <- upConvBlk(5*outChn, outChn, actFunc=actFunc, negative_slope=negative_slope)
    self$d2d4 <- torch::nn_sequential(interpUp(sFactor=4),
                                       simpleConvBlk(5*outChn, outChn, actFunc=actFunc, negative_slope=negative_slope))
    self$d3d4 <- upConvBlk(5*outChn, outChn, actFunc=actFunc, negative_slope=negative_slope)

    # Independent processing block for each decoder node (separate learned weights per scale)
    self$d1Blk <- simpleConvBlk(5*outChn, 5*outChn, actFunc=actFunc, negative_slope=negative_slope)
    self$d2Blk <- simpleConvBlk(5*outChn, 5*outChn, actFunc=actFunc, negative_slope=negative_slope)
    self$d3Blk <- simpleConvBlk(5*outChn, 5*outChn, actFunc=actFunc, negative_slope=negative_slope)
    self$d4Blk <- simpleConvBlk(5*outChn, 5*outChn, actFunc=actFunc, negative_slope=negative_slope)

    self$ch <- classifierBlk(5*outChn, nCls=nCls)

  },

  # Define the forward pass
  forward = function(x) {

    # Encoder
    x     <- self$encoder1(x)
    xe1d4 <- self$e1d4(x)
    xe1d3 <- self$e1d3(x)
    xe1d2 <- self$e1d2(x)
    xe1d1 <- self$e1d1(x)
    x <- self$maxP2(x)

    x     <- self$encoder2(x)
    xe2d3 <- self$e2d3(x)
    xe2d2 <- self$e2d2(x)
    xe2d1 <- self$e2d1(x)
    x <- self$maxP2(x)

    x     <- self$encoder3(x)
    xe3d2 <- self$e3d2(x)
    xe3d1 <- self$e3d1(x)
    x <- self$maxP2(x)

    x     <- self$encoder4(x)
    xe4d1 <- self$e4d1(x)
    x <- self$maxP2(x)

    # Bottleneck
    x    <- self$btn(x)
    xbd2 <- self$bd2(x)
    xbd3 <- self$bd3(x)
    xbd4 <- self$bd4(x)
    x    <- self$bd1(x)

    # Decoder d1 (1/8 resolution): bottleneck + all encoder scales downsampled
    d1In  <- torch::torch_cat(list(x, xe1d1, xe2d1, xe3d1, xe4d1), dim=2)
    d1Out <- self$d1Blk(d1In)
    xd1d2 <- self$d1d2(d1Out)
    xd1d3 <- self$d1d3(d1Out)
    xd1d4 <- self$d1d4(d1Out)

    # Decoder d2 (1/4 resolution): d1 upsampled + encoder scales + bottleneck
    d2In  <- torch::torch_cat(list(xd1d2, xe1d2, xe2d2, xe3d2, xbd2), dim=2)
    d2Out <- self$d2Blk(d2In)
    xd2d3 <- self$d2d3(d2Out)
    xd2d4 <- self$d2d4(d2Out)

    # Decoder d3 (1/2 resolution): d2 + d1 + encoder scales + bottleneck
    d3In  <- torch::torch_cat(list(xd2d3, xe1d3, xe2d3, xd1d3, xbd3), dim=2)
    d3Out <- self$d3Blk(d3In)
    xd3d4 <- self$d3d4(d3Out)

    # Decoder d4 (full resolution): d3 + d2 + d1 + encoder scale 1 + bottleneck
    d4In  <- torch::torch_cat(list(xd3d4, xe1d4, xd2d4, xd1d4, xbd4), dim=2)
    d4Out <- self$d4Blk(d4In)

    # Classifier head
    c4x <- self$ch(d4Out)

    if(self$useDS == TRUE){
      c3x <- self$ch(self$up2(d3Out))
      c2x <- self$ch(self$up4(d2Out))
      c1x <- self$ch(self$up8(d1Out))
      return(list(pred1=c4x, pred2=c3x, pred4=c2x, pred8=c1x))
    }else{
      return(c4x)
    }
  },

  get_param_groups = function(stageLRs = self$stageLRs) {
    if (is.null(stageLRs)) {
      return(list(list(params = unname(self$parameters))))
    }
    if (length(stageLRs) != 9L) {
      stop("stageLRs must be a numeric vector of length 9 for defineUNet3p: ",
           "encoder stages encoder1-encoder4, bottleneck, decoder stages d1-d4")
    }
    list(
      list(params = .collect_params(self$encoder1, self$e1d4, self$e1d3,
                                    self$e1d2, self$e1d1),                        lr = stageLRs[1]),
      list(params = .collect_params(self$encoder2, self$e2d3, self$e2d2,
                                    self$e2d1),                                   lr = stageLRs[2]),
      list(params = .collect_params(self$encoder3, self$e3d2, self$e3d1),         lr = stageLRs[3]),
      list(params = .collect_params(self$encoder4, self$e4d1),                    lr = stageLRs[4]),
      list(params = .collect_params(self$btn, self$bd1, self$bd2, self$bd3,
                                    self$bd4),                                    lr = stageLRs[5]),
      list(params = .collect_params(self$d1Blk, self$d1d2, self$d1d3,
                                    self$d1d4),                                   lr = stageLRs[6]),
      list(params = .collect_params(self$d2Blk, self$d2d3, self$d2d4),           lr = stageLRs[7]),
      list(params = .collect_params(self$d3Blk, self$d3d4),                      lr = stageLRs[8]),
      list(params = .collect_params(self$d4Blk, self$ch),                        lr = stageLRs[9])
    )
  },

  load_weights = function(path, encoderOnly = FALSE, freezeEncoder = FALSE) {
    state <- torch::torch_load(path)
    if (encoderOnly) {
      pfx  <- c("encoder1.", "encoder2.", "encoder3.", "encoder4.",
                "e1d4.", "e1d3.", "e1d2.", "e1d1.",
                "e2d3.", "e2d2.", "e2d1.",
                "e3d2.", "e3d1.", "e4d1.")
      keep <- vapply(names(state), function(k) any(startsWith(k, pfx)), logical(1L))
      self$load_state_dict(state[keep], strict = FALSE)
    } else {
      self$load_state_dict(state)
    }
    if (freezeEncoder) self$freeze_encoder(TRUE)
    invisible(self)
  },

  freeze_encoder = function(freeze = TRUE) {
    mods <- list(self$encoder1, self$encoder2, self$encoder3, self$encoder4,
                 self$e1d4, self$e1d3, self$e1d2, self$e1d1,
                 self$e2d3, self$e2d2, self$e2d1,
                 self$e3d2, self$e3d1, self$e4d1)
    for (mod in mods) for (p in mod$parameters) p$requires_grad_(!freeze)
    invisible(self)
  }
)





#' defineEfficientUNetB2
#'
#' Define a UNet architecture for geospatial semantic segmentation with an EfficientNet-B2 backbone.
#'
#' Define a UNet architecture with an EfficientNet-B2 backbone or encoder. The architecture
#' has 5 encoder blocks (e1-e5), a bottleneck, and 5 decoder blocks (d1-d5). The final decoder
#' block (d5) uses the original input image as its skip connection, matching the design of
#' \code{defineMobileUNet}. The user can load ImageNet weights, apply averaged ImageNet weights
#' when the input channel count differs from 3, replace the bottleneck with an ASPP module,
#' add attention gates along the skip connections, freeze the encoder, and enable deep
#' supervision.
#'
#' @param inChn Number of input channels or predictor variables. Default is 3.
#' @param nCls Number of classes being differentiated. For a binary classification,
#' this can be either 1 or 2. If 2, the problem is treated as a multiclass problem,
#' and a multiclass loss metric should be used. Default is 3.
#' @param pretrainedEncoder TRUE or FALSE. Whether or not to initialize using pre-trained
#' ImageNet weights for the EfficientNet-B2 encoder. Default is TRUE.
#' @param freezeEncoder TRUE or FALSE. Whether or not to freeze the encoder during training.
#' The default is FALSE. If TRUE, only the decoder component is trained.
#' @param avgImNetWeights TRUE or FALSE. If three predictor variables are provided and
#' ImageNet weights are used, whether or not to use the original weights or average them.
#' If the input has more or fewer than 3 channels, the stem weights are always averaged
#' regardless of this argument. Default is FALSE.
#' @param actFunc Defines the activation function used in the decoder and bottleneck (note
#' that EfficientNet-B2 encoder layers are not impacted; they use SiLU internally). "relu" =
#' rectified linear unit (ReLU); "lrelu" = leaky ReLU; "swish" = SiLU/swish; "gelu" = GELU.
#' Default is "relu".
#' @param useAttn TRUE or FALSE. Whether to add attention gates along the skip connections.
#' Default is FALSE.
#' @param useASPP TRUE or FALSE. Whether to replace the sequential bottleneck (EfficientNet-B2
#' stages 6-8) with an Atrous Spatial Pyramid Pooling (ASPP) module. If TRUE, the ASPP output
#' channel count is set by \code{btnChn}. Default is FALSE.
#' @param useDS TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four
#' predictions are made, one at each of the four largest decoder block resolutions, and
#' the predictions are returned as a list object containing the 4 predictions. If FALSE,
#' only the final prediction at the original resolution is returned. Default is FALSE.
#' @param dcChn Vector of 5 integers defining the number of output feature maps for each
#' of the 5 decoder blocks. Default is 256, 128, 64, 32, and 16.
#' @param btnChn Number of output channels from the ASPP bottleneck when \code{useASPP = TRUE}.
#' Ignored when \code{useASPP = FALSE} (the sequential bottleneck always outputs 352 channels).
#' Default is 256.
#' @param dilRates Vector of 3 dilation rates for the ASPP module. Default is 6, 12, and 18.
#' @param dilChn Vector of 4 channel counts for each ASPP branch. Default is 256 for each.
#' @param negative_slope If \code{actFunc = "lrelu"}, specifies the negative slope term.
#' Default is 0.01.
#' @param stageLRs Optional numeric vector of length 11 specifying a base learning rate for
#' each stage: encoder stages e1, e2, e3, e4, e5, the bottleneck, and decoder stages d1, d2,
#' d3, d4, d5 (in that order). Call \code{model$get_param_groups()} to obtain optimizer
#' parameter groups with per-stage learning rates. Default is NULL (single learning rate).
#' @return EfficientUNetB2 model instance as torch nn_module.
#' @export
defineEfficientUNetB2 <- torch::nn_module(
  "EfficientUNetB2",

  initialize = function(inChn             = 3,
                        nCls              = 3,
                        pretrainedEncoder = TRUE,
                        freezeEncoder     = FALSE,
                        avgImNetWeights   = FALSE,
                        actFunc           = "relu",
                        useAttn           = FALSE,
                        useASPP           = FALSE,
                        useDS             = FALSE,
                        dcChn             = c(256, 128, 64, 32, 16),
                        btnChn            = 256,
                        dilRates          = c(6, 12, 18),
                        dilChn            = c(256, 256, 256, 256),
                        negative_slope    = 0.01,
                        stageLRs          = NULL) {

    # Store settings
    self$inChn             <- inChn
    self$nCls              <- nCls
    self$pretrainedEncoder <- pretrainedEncoder
    self$freezeEncoder     <- freezeEncoder
    self$avgImNetWeights   <- avgImNetWeights
    self$actFunc           <- actFunc
    self$useAttn           <- useAttn
    self$useASPP           <- useASPP
    self$useDS             <- useDS
    self$dcChn             <- dcChn
    self$btnChn            <- btnChn
    self$stageLRs          <- stageLRs

    # --------------------------------------------------
    # EfficientNet-B2 backbone
    # --------------------------------------------------
    self$base_model <- torchvision::model_efficientnet_b2(
      pretrained = pretrainedEncoder
    )

    # --------------------------------------------------
    # Handle input channels / ImageNet weight averaging
    # --------------------------------------------------
    orig_conv <- self$base_model$features[[1]][[1]]
    orig_in   <- orig_conv$in_channels

    if (avgImNetWeights || inChn != orig_in) {
      old_w  <- orig_conv$weight
      mean_w <- old_w$mean(dim = 2L, keepdim = TRUE)
      out_ch <- old_w$size(1L)
      k_h    <- old_w$size(3L)
      k_w    <- old_w$size(4L)
      new_w  <- mean_w$expand(c(out_ch, inChn, k_h, k_w))

      new_conv <- torch::nn_conv2d(
        in_channels  = inChn,
        out_channels = out_ch,
        kernel_size  = c(k_h, k_w),
        stride       = orig_conv$stride,
        padding      = orig_conv$padding,
        bias         = !is.null(orig_conv$bias)
      )
      new_conv$weight <- torch::nn_parameter(new_w$clone())

      orig_bn  <- self$base_model$features[[1]][[2]]
      orig_act <- self$base_model$features[[1]][[3]]
      self$base_model$features[[1]] <- torch::nn_sequential(new_conv, orig_bn, orig_act)

      cat("First conv rebuilt: weight size is now ",
          self$base_model$features[[1]][[1]]$weight$size(), "\n")
    }

    # --------------------------------------------------
    # Optionally freeze encoder
    # --------------------------------------------------
    if (freezeEncoder) {
      for (p in self$base_model$parameters) p$requires_grad_(FALSE)
    }

    # --------------------------------------------------
    # Encoder blocks  (e1 = stem /2, e5 = deepest stage /16)
    # --------------------------------------------------
    self$e1  <- torch::nn_sequential(self$base_model$features[[1]])
    self$e2  <- torch::nn_sequential(self$base_model$features[[2]])
    self$e3  <- torch::nn_sequential(self$base_model$features[[3]])
    self$e4  <- torch::nn_sequential(self$base_model$features[[4]])
    self$e5  <- torch::nn_sequential(self$base_model$features[[5]])

    # Capture true encoder output widths
    self$encChn <- c(
      self$base_model$features[[1]][[1]]$out_channels,
      self$base_model$features[[2]]$out_channels,
      self$base_model$features[[3]]$out_channels,
      self$base_model$features[[4]]$out_channels,
      self$base_model$features[[5]]$out_channels
    )

    # --------------------------------------------------
    # Bottleneck: sequential stages 6-8 or ASPP
    # --------------------------------------------------
    # When useASPP = FALSE the sequential btn always outputs 352 ch (EfficientNet-B2 stage 8).
    btn_out_chn <- if (useASPP) btnChn else 352L

    if (useASPP) {
      self$btn <- geodl:::asppBlkR(
        inChn          = self$encChn[5],
        outChn         = btnChn,
        dilChn         = dilChn,
        dilRates       = dilRates,
        actFunc        = actFunc,
        negative_slope = negative_slope
      )
    } else {
      self$btn <- torch::nn_sequential(self$base_model$features[6:8])
    }

    # --------------------------------------------------
    # Decoder up-conv blocks
    # --------------------------------------------------
    self$dUp1 <- geodl:::upConvBlk(btn_out_chn, btn_out_chn)
    self$dUp2 <- geodl:::upConvBlk(dcChn[1],    dcChn[1])
    self$dUp3 <- geodl:::upConvBlk(dcChn[2],    dcChn[2])
    self$dUp4 <- geodl:::upConvBlk(dcChn[3],    dcChn[3])
    self$dUp5 <- geodl:::upConvBlk(dcChn[4],    dcChn[4])

    # Decoder conv blocks.
    # d5 skips from the original input x (H x W, inChn channels), not from e1x (H/2).
    # This mirrors defineMobileUNet's identity e1 stage and avoids a spatial mismatch.
    self$d1 <- geodl:::doubleConvBlk(btn_out_chn      + self$encChn[5], dcChn[1], actFunc, negative_slope)
    self$d2 <- geodl:::doubleConvBlk(dcChn[1]         + self$encChn[4], dcChn[2], actFunc, negative_slope)
    self$d3 <- geodl:::doubleConvBlk(dcChn[2]         + self$encChn[3], dcChn[3], actFunc, negative_slope)
    self$d4 <- geodl:::doubleConvBlk(dcChn[3]         + self$encChn[2], dcChn[4], actFunc, negative_slope)
    self$d5 <- geodl:::doubleConvBlk(dcChn[4]         + inChn,          dcChn[5], actFunc, negative_slope)

    # --------------------------------------------------
    # Optional attention gates (one per skip connection)
    # --------------------------------------------------
    if (useAttn) {
      self$ag5 <- geodl:::attnBlk(self$encChn[5], btn_out_chn)  # e5x (H/16) gated by btnx (H/32)
      self$ag4 <- geodl:::attnBlk(self$encChn[4], dcChn[1])     # e4x (H/8)  gated by d1x  (H/16)
      self$ag3 <- geodl:::attnBlk(self$encChn[3], dcChn[2])     # e3x (H/4)  gated by d2x  (H/8)
      self$ag2 <- geodl:::attnBlk(self$encChn[2], dcChn[3])     # e2x (H/2)  gated by d3x  (H/4)
      self$ag1 <- geodl:::attnBlk(inChn,          dcChn[4])     # x   (H)    gated by d4x  (H/2)
    }

    # --------------------------------------------------
    # Classifier and optional deep-supervision heads
    # --------------------------------------------------
    self$c4 <- geodl:::classifierBlk(dcChn[5], nCls)

    if (useDS) {
      self$upSamp2 <- torch::nn_upsample(scale_factor = 2, mode = "bilinear", align_corners = TRUE)
      self$upSamp4 <- torch::nn_upsample(scale_factor = 4, mode = "bilinear", align_corners = TRUE)
      self$upSamp8 <- torch::nn_upsample(scale_factor = 8, mode = "bilinear", align_corners = TRUE)
      self$c3 <- geodl:::classifierBlk(dcChn[4], nCls)
      self$c2 <- geodl:::classifierBlk(dcChn[3], nCls)
      self$c1 <- geodl:::classifierBlk(dcChn[2], nCls)
    }
  },

  forward = function(x) {

    e1x  <- self$e1(x)
    e2x  <- self$e2(e1x)
    e3x  <- self$e3(e2x)
    e4x  <- self$e4(e3x)
    e5x  <- self$e5(e4x)
    btnx <- self$btn(e5x)

    if (self$useAttn) e5x <- self$ag5(e5x, btnx)
    d1x <- self$d1(torch::torch_cat(list(self$dUp1(btnx), e5x), dim = 2))

    if (self$useAttn) e4x <- self$ag4(e4x, d1x)
    d2x <- self$d2(torch::torch_cat(list(self$dUp2(d1x), e4x), dim = 2))

    if (self$useAttn) e3x <- self$ag3(e3x, d2x)
    d3x <- self$d3(torch::torch_cat(list(self$dUp3(d2x), e3x), dim = 2))

    if (self$useAttn) e2x <- self$ag2(e2x, d3x)
    d4x <- self$d4(torch::torch_cat(list(self$dUp4(d3x), e2x), dim = 2))

    if (self$useAttn) x <- self$ag1(x, d4x)
    d5x <- self$d5(torch::torch_cat(list(self$dUp5(d4x), x), dim = 2))

    c4x <- self$c4(d5x)

    if (self$useDS) {
      u2 <- self$upSamp2(d4x); u4 <- self$upSamp4(d3x); u8 <- self$upSamp8(d2x)
      c3x <- self$c3(u2); c2x <- self$c2(u4); c1x <- self$c1(u8)
      return(list(pred1=c4x, pred2=c3x, pred4=c2x, pred8=c1x))
    } else {
      return(c4x)
    }
  },

  get_param_groups = function(stageLRs = self$stageLRs) {
    if (is.null(stageLRs)) {
      return(list(list(params = unname(self$parameters))))
    }
    if (length(stageLRs) != 11L) {
      stop("stageLRs must be a numeric vector of length 11 for defineEfficientUNetB2: ",
           "encoder stages e1-e5, bottleneck, decoder stages d1-d5")
    }
    list(
      list(params = .collect_params(self$e1),  lr = stageLRs[1]),
      list(params = .collect_params(self$e2),  lr = stageLRs[2]),
      list(params = .collect_params(self$e3),  lr = stageLRs[3]),
      list(params = .collect_params(self$e4),  lr = stageLRs[4]),
      list(params = .collect_params(self$e5),  lr = stageLRs[5]),
      list(params = .collect_params(self$btn), lr = stageLRs[6]),
      list(params = .collect_params(self$dUp1, self$d1,
                                    if (self$useAttn) self$ag5),             lr = stageLRs[7]),
      list(params = .collect_params(self$dUp2, self$d2,
                                    if (self$useAttn) self$ag4),             lr = stageLRs[8]),
      list(params = .collect_params(self$dUp3, self$d3,
                                    if (self$useAttn) self$ag3),             lr = stageLRs[9]),
      list(params = .collect_params(self$dUp4, self$d4,
                                    if (self$useAttn) self$ag2),             lr = stageLRs[10]),
      list(params = .collect_params(self$dUp5, self$d5,
                                    if (self$useAttn) self$ag1,
                                    self$c4,
                                    if (self$useDS) self$c1,
                                    if (self$useDS) self$c2,
                                    if (self$useDS) self$c3),                lr = stageLRs[11])
    )
  },

  load_weights = function(path, encoderOnly = FALSE, freezeEncoder = FALSE) {
    state <- torch::torch_load(path)
    if (encoderOnly) {
      pfx  <- c("e1.", "e2.", "e3.", "e4.", "e5.")
      keep <- vapply(names(state), function(k) any(startsWith(k, pfx)), logical(1L))
      self$load_state_dict(state[keep], strict = FALSE)
    } else {
      self$load_state_dict(state)
    }
    if (freezeEncoder) self$freeze_encoder(TRUE)
    invisible(self)
  },

  freeze_encoder = function(freeze = TRUE) {
    for (mod in list(self$e1, self$e2, self$e3, self$e4, self$e5))
      for (p in mod$parameters) p$requires_grad_(!freeze)
    invisible(self)
  }
)



#' defineConvnextUNet
#'
#' Define a UNet architecture for geospatial semantic segmentation with a ConvNext-Tiny backbone.
#'
#' @param inChn Number of input channels or predictor variables. Default is 3.
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
#' that ConvNeXt encoder layers are not impacted). "relu" = rectified linear unit (ReLU);
#' "lrelu" = leaky ReLU; "swish" = swish; "gelu" = GELU. Default is "relu".
#' @param useDS TRUE or FALSE. Whether or not to use deep supervision. If TRUE, four
#' predictions are made, one at each of the four largest decoder block resolutions, and
#' the predictions are returned as a list object containing the 4 predictions. If FALSE,
#' only the final prediction at the original resolution is returned. Default is FALSE
#' or deep supervision is not implemented.
#' @param dcChn Vector of 4 integers defining the number of output feature
#' maps for each of the 4 decoder blocks. Default is 128, 64, 32, and 16.
#' @param negative_slope If actFunc = "lrelu", specifies the negative slope term
#' to use. Default is 0.01.
#' @param stageLRs Optional numeric vector of length 9 specifying a base learning
#' rate for each stage: encoder stages e1, e2, e3, e4, then decoder stages d1,
#' d2, d3, d4, d5 (in that order). The ConvNeXt-Tiny bottleneck is an identity
#' transform with no learnable parameters and is therefore excluded from the
#' stage count. Call \code{model$get_param_groups()} to obtain optimizer
#' parameter groups. Default is NULL (single learning rate).
#' @return ModileUNet model instance as torch nn_module
#' @export
defineConvNeXtTinyUNet <- torch::nn_module(
  "ConvNeXtTinyUNet",

  initialize = function(inChn = 3,
                        nCls = 3,
                        pretrainedEncoder = TRUE,
                        freezeEncoder = TRUE,
                        avgImNetWeights = FALSE,
                        actFunc = "relu",
                        useDS = FALSE,
                        dcChn = c(256,128,64,32,16),
                        negative_slope = 0.01,
                        stageLRs = NULL){

    self$inChn    <- inChn
    self$nCls     <- nCls
    self$useDS    <- useDS
    self$stageLRs <- stageLRs

    # --------------------------------------------------
    # ConvNeXt-Tiny backbone
    # --------------------------------------------------
    self$base_model <- torchvision::model_convnext_tiny_1k(
      pretrained = pretrainedEncoder
    )

    # ---- rebuild patch stem for multispectral input ----
    stem <- self$base_model$features[[1]]
    old_conv <- stem[[1]]   # 4x4 stride-4 conv
    if (avgImNetWeights || inChn != old_conv$in_channels) {

      old_w  <- old_conv$weight
      mean_w <- old_w$mean(dim = 2, keepdim = TRUE)

      new_w <- mean_w$expand(c(old_w$size(1), inChn,
                               old_w$size(3), old_w$size(4)))

      new_conv <- torch::nn_conv2d(
        in_channels  = inChn,
        out_channels = old_w$size(1),
        kernel_size  = c(old_w$size(3), old_w$size(4)),
        stride       = old_conv$stride,
        padding      = old_conv$padding,
        bias         = !is.null(old_conv$bias)
      )

      new_conv$weight <- torch::nn_parameter(new_w$clone())
      stem[[1]] <- new_conv
      self$base_model$features[[1]] <- stem
    }

    if (freezeEncoder) {
      for (p in self$base_model$parameters) p$requires_grad_(FALSE)
    }

    # --------------------------------------------------
    # ConvNeXt stages
    # --------------------------------------------------
    self$e1 <- torch::nn_sequential(self$base_model$features[[1]])  # 96  @ H/4
    self$e2 <- torch::nn_sequential(self$base_model$features[[2]])  # 192 @ H/8
    self$e3 <- torch::nn_sequential(self$base_model$features[[3]])  # 384 @ H/16
    self$e4 <- torch::nn_sequential(self$base_model$features[[4]])  # 768 @ H/32
    self$btn <- torch::nn_identity()

    # --------------------------------------------------
    # Decoder
    # --------------------------------------------------
    self$dUp1 <- geodl:::upConvBlk(768, 768)
    self$dUp2 <- geodl:::upConvBlk(dcChn[1], dcChn[1])
    self$dUp3 <- geodl:::upConvBlk(dcChn[2], dcChn[2])
    self$dUp4 <- geodl:::upConvBlk(dcChn[3], dcChn[3])
    self$dUp5 <- geodl:::upConvBlk(dcChn[4], dcChn[4])

    self$d1 <- geodl:::doubleConvBlk(768 + 384, dcChn[1], actFunc, negative_slope)
    self$d2 <- geodl:::doubleConvBlk(dcChn[1] + 192, dcChn[2], actFunc, negative_slope)
    self$d3 <- geodl:::doubleConvBlk(dcChn[2] + 96,  dcChn[3], actFunc, negative_slope)
    self$d4 <- geodl:::doubleConvBlk(dcChn[3],       dcChn[4], actFunc, negative_slope)
    self$d5 <- geodl:::doubleConvBlk(dcChn[4] + inChn, dcChn[5], actFunc, negative_slope)

    # restore resolution: ConvNeXt stem downsamples by 4
    self$finalUp <- torch::nn_upsample(scale_factor = 4,
                                       mode="bilinear",
                                       align_corners=TRUE)

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

    e1x <- self$e1(x)
    e2x <- self$e2(e1x)
    e3x <- self$e3(e2x)
    e4x <- self$e4(e3x)

    d1x <- self$d1(torch::torch_cat(list(self$dUp1(e4x), e3x), dim=2))
    d2x <- self$d2(torch::torch_cat(list(self$dUp2(d1x), e2x), dim=2))
    d3x <- self$d3(torch::torch_cat(list(self$dUp3(d2x), e1x), dim=2))
    d4x <- self$d4(self$dUp4(d3x))
    d5x <- self$d5(torch::torch_cat(list(self$dUp5(d4x), x), dim=2))

    d5x <- self$finalUp(d5x)
    out <- self$c4(d5x)

    if (self$useDS) {
      return(list(
        pred1 = out,
        pred2 = self$c3(self$finalUp(self$upSamp2(d4x))),
        pred4 = self$c2(self$finalUp(self$upSamp4(d3x))),
        pred8 = self$c1(self$finalUp(self$upSamp8(d2x)))
      ))
    } else {
      return(out)
    }
  },

  get_param_groups = function(stageLRs = self$stageLRs) {
    if (is.null(stageLRs)) {
      return(list(list(params = unname(self$parameters))))
    }
    if (length(stageLRs) != 9L) {
      stop("stageLRs must be a numeric vector of length 9 for defineConvNeXtTinyUNet: ",
           "encoder stages e1-e4, then decoder stages d1-d5 (no learnable bottleneck)")
    }
    list(
      list(params = .collect_params(self$e1), lr = stageLRs[1]),
      list(params = .collect_params(self$e2), lr = stageLRs[2]),
      list(params = .collect_params(self$e3), lr = stageLRs[3]),
      list(params = .collect_params(self$e4), lr = stageLRs[4]),
      list(params = .collect_params(self$dUp1, self$d1), lr = stageLRs[5]),
      list(params = .collect_params(self$dUp2, self$d2), lr = stageLRs[6]),
      list(params = .collect_params(self$dUp3, self$d3), lr = stageLRs[7]),
      list(params = .collect_params(self$dUp4, self$d4), lr = stageLRs[8]),
      list(params = .collect_params(self$dUp5, self$d5, self$c4,
                                    if (self$useDS) self$c1,
                                    if (self$useDS) self$c2,
                                    if (self$useDS) self$c3),               lr = stageLRs[9])
    )
  },

  load_weights = function(path, encoderOnly = FALSE, freezeEncoder = FALSE) {
    state <- torch::torch_load(path)
    if (encoderOnly) {
      pfx  <- c("e1.", "e2.", "e3.", "e4.")
      keep <- vapply(names(state), function(k) any(startsWith(k, pfx)), logical(1L))
      self$load_state_dict(state[keep], strict = FALSE)
    } else {
      self$load_state_dict(state)
    }
    if (freezeEncoder) self$freeze_encoder(TRUE)
    invisible(self)
  },

  freeze_encoder = function(freeze = TRUE) {
    for (mod in list(self$e1, self$e2, self$e3, self$e4))
      for (p in mod$parameters) p$requires_grad_(!freeze)
    invisible(self)
  }
)


#' defineDeepLabV3Plus
#'
#' Define a DeepLabv3+ architecture for geospatial semantic segmentation.
#'
#' Define a DeepLabv3+-like architecture with a custom 4-block encoder, an atrous
#' spatial pyramid pooling (ASPP) module, and a lightweight decoder that fuses
#' high-level ASPP features with low-level encoder features. Unlike the symmetric
#' UNet decoder, the DeepLabv3+ decoder uses a single skip connection from the
#' third encoder block (stride 4) and upsamples the ASPP output (stride 16) 4x
#' before fusion, then upsamples 4x again to reach the original resolution.
#'
#' The architecture is inspired by:
#'
#' Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F. and Adam, H., 2018.
#' Encoder-decoder with atrous separable convolution for semantic image
#' segmentation. In Proceedings of the European Conference on Computer Vision
#' (ECCV) (pp. 801-818).
#'
#' @param inChn Number of channels, bands, or predictor variables in the input
#' image or raster data. Default is 3.
#' @param nCls Number of classes being differentiated. For a binary classification,
#' this can be either 1 or 2. If 2, the problem is treated as a multiclass problem,
#' and a multiclass loss metric should be used. Default is 3.
#' @param actFunc Defines activation function to use throughout the network. "relu" =
#' rectified linear unit (ReLU); "lrelu" = leaky ReLU; "swish" = swish; "gelu" = GELU.
#' Default is "relu".
#' @param useRes TRUE or FALSE. Whether to include residual connections in the encoder
#' and decoder blocks. Default is FALSE.
#' @param enChn Vector of 4 integers defining the number of output feature maps for
#' each of the four encoder blocks. Default is 16, 32, 64, and 128.
#' @param btnChn Number of output feature maps from the ASPP module. Default is 256.
#' @param lowLevelChn Number of channels to project the low-level encoder features
#' (from the third encoder block at stride 4) to before fusing with the upsampled
#' ASPP output. Default is 48.
#' @param dcChn Number of output feature maps from the decoder convolution block.
#' Default is 256.
#' @param dilRates Vector of 3 values specifying the dilation rates used in the ASPP
#' module. Default is 6, 12, and 18.
#' @param dilChn Vector of 4 values specifying the number of channels to produce at
#' each dilation rate within the ASPP module. Default is 256 for each.
#' @param negative_slope If actFunc = "lrelu", specifies the negative slope term.
#' Default is 0.01.
#' @return DeepLabV3Plus model instance as torch nn_module
#' @export
defineDeepLabV3Plus <- torch::nn_module(
  "DeepLabV3Plus",

  initialize = function(inChn          = 3,
                        nCls           = 3,
                        actFunc        = "relu",
                        useRes         = FALSE,
                        enChn          = c(16, 32, 64, 128),
                        btnChn         = 256,
                        lowLevelChn    = 48,
                        dcChn          = 256,
                        dilRates       = c(6, 12, 18),
                        dilChn         = c(256, 256, 256, 256),
                        negative_slope = 0.01) {

    self$actFunc        <- actFunc
    self$useRes         <- useRes
    self$inChn          <- inChn
    self$nCls           <- nCls
    self$enChn          <- enChn
    self$btnChn         <- btnChn
    self$lowLevelChn    <- lowLevelChn
    self$dcChn          <- dcChn
    self$dilRates       <- dilRates
    self$dilChn         <- dilChn
    self$negative_slope <- negative_slope

    # Encoder: 4 blocks each followed by 2x max-pool downsampling (total stride 16)
    if (useRes) {
      self$e1 <- geodl:::doubleConvBlkR(inChn   = inChn,
                                        outChn  = enChn[1],
                                        actFunc = actFunc,
                                        negative_slope = negative_slope)
      self$e2 <- geodl:::doubleConvBlkR(inChn   = enChn[1],
                                        outChn  = enChn[2],
                                        actFunc = actFunc,
                                        negative_slope = negative_slope)
      self$e3 <- geodl:::doubleConvBlkR(inChn   = enChn[2],
                                        outChn  = enChn[3],
                                        actFunc = actFunc,
                                        negative_slope = negative_slope)
      self$e4 <- geodl:::doubleConvBlkR(inChn   = enChn[3],
                                        outChn  = enChn[4],
                                        actFunc = actFunc,
                                        negative_slope = negative_slope)
      self$dec <- geodl:::doubleConvBlkR(inChn   = btnChn + lowLevelChn,
                                         outChn  = dcChn,
                                         actFunc = actFunc,
                                         negative_slope = negative_slope)
    } else {
      self$e1 <- geodl:::doubleConvBlk(inChn   = inChn,
                                       outChn  = enChn[1],
                                       actFunc = actFunc,
                                       negative_slope = negative_slope)
      self$e2 <- geodl:::doubleConvBlk(inChn   = enChn[1],
                                       outChn  = enChn[2],
                                       actFunc = actFunc,
                                       negative_slope = negative_slope)
      self$e3 <- geodl:::doubleConvBlk(inChn   = enChn[2],
                                       outChn  = enChn[3],
                                       actFunc = actFunc,
                                       negative_slope = negative_slope)
      self$e4 <- geodl:::doubleConvBlk(inChn   = enChn[3],
                                       outChn  = enChn[4],
                                       actFunc = actFunc,
                                       negative_slope = negative_slope)
      self$dec <- geodl:::doubleConvBlk(inChn   = btnChn + lowLevelChn,
                                        outChn  = dcChn,
                                        actFunc = actFunc,
                                        negative_slope = negative_slope)
    }

    # ASPP bottleneck at stride 16
    self$aspp <- geodl:::asppBlk(inChn    = enChn[4],
                                  outChn  = btnChn,
                                  dilChn  = dilChn,
                                  dilRates = dilRates,
                                  actFunc  = actFunc,
                                  negative_slope = negative_slope)

    # 1x1 projection of low-level features (e3 at stride 4) to lowLevelChn channels
    self$lowLevelProj <- geodl:::featReduce(inChn   = enChn[3],
                                             outChn = lowLevelChn,
                                             actFunc = actFunc,
                                             dobnAct = TRUE,
                                             negative_slope = negative_slope)

    # Classification head
    self$cls <- geodl:::classifierBlk(inChn = dcChn,
                                       nCls  = nCls)
  },

  forward = function(x) {

    # Encoder path
    e1x   <- self$e1(x)
    e1xMP <- torch::nnf_max_pool2d(e1x, kernel_size = c(2,2), stride = 2, padding = 0)

    e2x   <- self$e2(e1xMP)
    e2xMP <- torch::nnf_max_pool2d(e2x, kernel_size = c(2,2), stride = 2, padding = 0)

    e3x   <- self$e3(e2xMP)                                                         # stride 4 (low-level)
    e3xMP <- torch::nnf_max_pool2d(e3x, kernel_size = c(2,2), stride = 2, padding = 0)

    e4x   <- self$e4(e3xMP)
    e4xMP <- torch::nnf_max_pool2d(e4x, kernel_size = c(2,2), stride = 2, padding = 0)

    # ASPP at stride 16
    asppOut <- self$aspp(e4xMP)

    # Upsample ASPP output 4x: stride 16 -> stride 4
    asppUp <- torch::nnf_interpolate(asppOut,
                                     scale_factor  = 4,
                                     mode          = "bilinear",
                                     align_corners = TRUE)

    # Project low-level features and fuse with upsampled ASPP output
    lowLevel <- self$lowLevelProj(e3x)
    fused    <- torch::torch_cat(list(asppUp, lowLevel), dim = 2)

    # Decoder convolutions
    decOut <- self$dec(fused)

    # Upsample 4x: stride 4 -> original resolution
    out <- torch::nnf_interpolate(decOut,
                                   scale_factor  = 4,
                                   mode          = "bilinear",
                                   align_corners = TRUE)

    # Classification
    out <- self$cls(out)

    return(out)
  }
)


#' defineCBAMUNet
#'
#' Define a UNet architecture with CBAM attention on skip connections.
#'
#' Define a UNet architecture with 4 encoder blocks, a bottleneck, and 4 decoder blocks.
#' Convolutional Block Attention Modules (CBAM) are applied to each encoder skip connection
#' before it is concatenated with the upsampled decoder features, allowing the network to
#' recalibrate features both channel-wise and spatially at each scale. The architecture
#' supports four activation function choices — ReLU, leaky ReLU, swish/SiLU, and GELU —
#' applied consistently throughout all encoder, decoder, and bottleneck blocks. Optional
#' residual connections, an ASPP bottleneck for multi-scale context, and deep supervision
#' are also available.
#'
#' @param inChn Number of channels, bands, or predictor variables in the input image or
#' raster data. Default is 3.
#' @param nCls Number of classes being differentiated. For a binary classification, this
#' can be either 1 or 2. If 2, the problem is treated as multiclass and a multiclass loss
#' should be used. Default is 3.
#' @param actFunc Activation function used throughout the network. "relu" = rectified linear
#' unit; "lrelu" = leaky ReLU; "swish" = Swish/SiLU; "gelu" = GELU. Default is "relu".
#' @param useRes TRUE or FALSE. Whether to include residual connections in all encoder,
#' decoder, and bottleneck blocks. Default is FALSE.
#' @param useASPP TRUE or FALSE. Whether to replace the standard double-convolution bottleneck
#' with an Atrous Spatial Pyramid Pooling (ASPP) module for multi-scale context aggregation.
#' Default is FALSE.
#' @param useDS TRUE or FALSE. Whether to use deep supervision. If TRUE, four predictions are
#' returned as a list (finest to coarsest resolution). If FALSE, only the final full-resolution
#' prediction is returned. Default is FALSE.
#' @param enChn Vector of 4 integers defining the number of output feature maps for each
#' encoder block. Default is c(16, 32, 64, 128).
#' @param dcChn Vector of 4 integers defining the number of output feature maps for each
#' decoder block. Default is c(128, 64, 32, 16).
#' @param btnChn Number of output feature maps from the bottleneck block. Default is 256.
#' @param dilRates Vector of 3 dilation rates used in the ASPP module. Default is c(6, 12, 18).
#' @param dilChn Vector of 4 values specifying the number of channels at each ASPP branch.
#' Default is c(256, 256, 256, 256).
#' @param negative_slope Negative slope for leaky ReLU when actFunc = "lrelu". Default is 0.01.
#' @param cbamRatio Channel reduction ratio used in the CBAM channel attention MLP. Larger
#' values produce a lighter module. Default is 8.
#' @param cbamKernelSize Kernel size for the CBAM spatial attention convolution. Must be odd.
#' Default is 7.
#' @param stageLRs Optional numeric vector of length 9 specifying a base learning
#' rate for each stage: encoder stages e1, e2, e3, e4 (each paired with its CBAM
#' module), the bottleneck, and decoder stages d1, d2, d3, d4 (in that order).
#' Call \code{model$get_param_groups()} to obtain optimizer parameter groups.
#' Default is NULL (single learning rate).
#' @return CBAMUNet model instance as a torch nn_module.
#' @export
defineCBAMUNet <- torch::nn_module(
  "CBAMUNet",
  initialize = function(inChn          = 3,
                         nCls           = 3,
                         actFunc        = "relu",
                         useRes         = FALSE,
                         useASPP        = FALSE,
                         useDS          = FALSE,
                         enChn          = c(16, 32, 64, 128),
                         dcChn          = c(128, 64, 32, 16),
                         btnChn         = 256,
                         dilRates       = c(6, 12, 18),
                         dilChn         = c(256, 256, 256, 256),
                         negative_slope = 0.01,
                         cbamRatio      = 8,
                         cbamKernelSize = 7,
                         stageLRs       = NULL) {

    self$useRes   <- useRes
    self$useASPP  <- useASPP
    self$useDS    <- useDS
    self$enChn    <- enChn
    self$dcChn    <- dcChn
    self$btnChn   <- btnChn
    self$stageLRs <- stageLRs

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

      self$dUp1 <- geodl:::upConvBlk(inChn=btnChn,   outChn=btnChn)
      self$dUp2 <- geodl:::upConvBlk(inChn=dcChn[1], outChn=dcChn[1])
      self$dUp3 <- geodl:::upConvBlk(inChn=dcChn[2], outChn=dcChn[2])
      self$dUp4 <- geodl:::upConvBlk(inChn=dcChn[3], outChn=dcChn[3])

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

      self$dUp1 <- geodl:::upConvBlk(inChn=btnChn,   outChn=btnChn)
      self$dUp2 <- geodl:::upConvBlk(inChn=dcChn[1], outChn=dcChn[1])
      self$dUp3 <- geodl:::upConvBlk(inChn=dcChn[2], outChn=dcChn[2])
      self$dUp4 <- geodl:::upConvBlk(inChn=dcChn[3], outChn=dcChn[3])

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
      self$btn <- geodl:::bottleneck(inChn=enChn[4],
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

    self$cbam1 <- geodl:::cbamBlk(inChn=enChn[1],
                           ratio=cbamRatio,
                           kernelSize=cbamKernelSize)
    self$cbam2 <- geodl:::cbamBlk(inChn=enChn[2],
                           ratio=cbamRatio,
                           kernelSize=cbamKernelSize)
    self$cbam3 <- geodl:::cbamBlk(inChn=enChn[3],
                           ratio=cbamRatio,
                           kernelSize=cbamKernelSize)
    self$cbam4 <- geodl:::cbamBlk(inChn=enChn[4],
                           ratio=cbamRatio,
                           kernelSize=cbamKernelSize)

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
    e1xMP <- torch::nnf_max_pool2d(e1x,
                                   kernel_size=c(2,2),
                                   stride=2,
                                   padding=0)

    e2x <- self$e2(e1xMP)
    e2xMP <- torch::nnf_max_pool2d(e2x,
                                   kernel_size=c(2,2),
                                   stride=2,
                                   padding=0)

    e3x <- self$e3(e2xMP)
    e3xMP <- torch::nnf_max_pool2d(e3x,
                                   kernel_size=c(2,2),
                                   stride=2,
                                   padding=0)

    e4x <- self$e4(e3xMP)
    e4xMP <- torch::nnf_max_pool2d(e4x,
                                   kernel_size=c(2,2),
                                   stride=2,
                                   padding=0)

    btnx <- self$btn(e4xMP)

    e4x <- self$cbam4(e4x)
    e3x <- self$cbam3(e3x)
    e2x <- self$cbam2(e2x)
    e1x <- self$cbam1(e1x)

    d1Upx <- self$dUp1(btnx)
    d1Cat <- torch::torch_cat(list(d1Upx, e4x), dim=2)
    d1x <- self$d1(d1Cat)

    d2Upx <- self$dUp2(d1x)
    d2Cat <- torch::torch_cat(list(d2Upx, e3x), dim=2)
    d2x <- self$d2(d2Cat)

    d3Upx <- self$dUp3(d2x)
    d3Cat <- torch::torch_cat(list(d3Upx, e2x), dim=2)
    d3x <- self$d3(d3Cat)

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
  },

  get_param_groups = function(stageLRs = self$stageLRs) {
    if (is.null(stageLRs)) {
      return(list(list(params = unname(self$parameters))))
    }
    if (length(stageLRs) != 9L) {
      stop("stageLRs must be a numeric vector of length 9 for defineCBAMUNet: ",
           "encoder stages e1-e4 (with CBAM), bottleneck, decoder stages d1-d4")
    }
    list(
      list(params = .collect_params(self$e1, self$cbam1),              lr = stageLRs[1]),
      list(params = .collect_params(self$e2, self$cbam2),              lr = stageLRs[2]),
      list(params = .collect_params(self$e3, self$cbam3),              lr = stageLRs[3]),
      list(params = .collect_params(self$e4, self$cbam4),              lr = stageLRs[4]),
      list(params = .collect_params(self$btn),                         lr = stageLRs[5]),
      list(params = .collect_params(self$dUp1, self$d1),               lr = stageLRs[6]),
      list(params = .collect_params(self$dUp2, self$d2),               lr = stageLRs[7]),
      list(params = .collect_params(self$dUp3, self$d3),               lr = stageLRs[8]),
      list(params = .collect_params(self$dUp4, self$d4, self$c4,
                                    if (self$useDS) self$c1,
                                    if (self$useDS) self$c2,
                                    if (self$useDS) self$c3),          lr = stageLRs[9])
    )
  },

  load_weights = function(path, encoderOnly = FALSE, freezeEncoder = FALSE) {
    state <- torch::torch_load(path)
    if (encoderOnly) {
      pfx  <- c("e1.", "e2.", "e3.", "e4.", "cbam1.", "cbam2.", "cbam3.", "cbam4.")
      keep <- vapply(names(state), function(k) any(startsWith(k, pfx)), logical(1L))
      self$load_state_dict(state[keep], strict = FALSE)
    } else {
      self$load_state_dict(state)
    }
    if (freezeEncoder) self$freeze_encoder(TRUE)
    invisible(self)
  },

  freeze_encoder = function(freeze = TRUE) {
    for (mod in list(self$e1, self$e2, self$e3, self$e4,
                     self$cbam1, self$cbam2, self$cbam3, self$cbam4))
      for (p in mod$parameters) p$requires_grad_(!freeze)
    invisible(self)
  }
)

