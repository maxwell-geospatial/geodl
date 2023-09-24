#' baseUnet
#'
#' Define a basic UNet architecture for semantic segmentation.
#'
#' Define a basic Unet architecture with 4 blocks in the encoder, a bottleneck
#' block, and 4 blocks in the decoder. UNet can accept a variable number of input
#' channels, and the user can define the number of feature maps produced in each
#' encoder and decoder block and the bottleneck. When the UNet is used to predict to
#' new data, it will return either the positive class logit, in the case of a binary
#' classification, or a logit for each class in the case of a multiclass classification.
#'
#' @param nChn Number of channels, bands, or predictor variables in the input
#' image or raster data. Default is 3.
#' @param nCls Number of classes being differentiated. For a binary classification,
#' this can be either 1 or 2. If 2, the problem is treated as a multiclass problem,
#' and a multiclass loss metric should be used.
#' @param encoderChn Vector of 4 integers defining the number of output
#' feature maps for each of the four encoder blocks. Default is 16, 32, 64, and 128.
#' @param decoderChn Vector of 4 integers defining the number of output feature
#' maps for each of the 4 decoder blocks. Default is 128, 64, 32, and 16.
#' @param botChn Number of output feature maps from the bottleneck block. Default
#' is 256.
#' @param useLeaky TRUE or FALSE. If TRUE, leaky ReLU activation is used as opposed
#' to ReLU. If FALSE, ReLU is used. Default is FALSE.
#' @param negative_slope If useLeaky is TRUE, specifies the negative slope term
#' to use. Default is 0.01.
#' @return Instantiated UNet model as subclass of torch::nn_module(). If used to
#' infer new data, will return a tensor of predicted logits.
#' @export
baseUNet <- torch::nn_module(
  "UNet",
  initialize = function(nChn = 3,
                        nCls,
                        encoderChn = c(16,32,64,128),
                        decoderChn = c(128,64,32,16),
                        botChn = 256,
                        useLeaky = FALSE,
                        negative_slope = 0.01){

    #self$nChn <- nChn
    #self$nCls <- nCls
    #self$encoderChn <- encoderChn
    #self$decoderChn <- decoderChn
    #self$botChn <- botChn
    #self$useLeaky <- useLeaky
    #self$negative_slope <- negative_slope

    self$up_conv <- function(inChannels, outChannels, useLeaky=FALSE, negative_slope=0.01) {
      return(
        torch::nn_sequential(
          torch::nn_conv_transpose2d(inChannels, outChannels, kernel_size=c(2,2), stride=2),
          torch::nn_batch_norm2d(outChannels),
          if(useLeaky == TRUE){
            torch::nn_relu(inplace=TRUE)
          }else{
            torch::nn_leaky_relu(inplace=TRUE)
          }
        )
      )
    }

    self$double_conv <- function(inChannels, outChannels, useLeaky=FALSE, negative_slope=0.01) {
      return(
        torch::nn_sequential(
          torch::nn_conv2d(inChannels, outChannels, kernel_size=c(3,3), stride=1, padding=1),
          torch::nn_batch_norm2d(outChannels),
          if(useLeaky == TRUE){
            torch::nn_relu(inplace=TRUE)
          }else{
            torch::nn_leaky_relu(inplace=TRUE)
          },
          torch::nn_conv2d(outChannels, outChannels, kernel_size=c(3,3), stride=1, padding=1),
          torch::nn_batch_norm2d(outChannels),
          if(useLeaky == TRUE){
            torch::nn_relu(inplace=TRUE)
          }else{
            torch::nn_leaky_relu(inplace=TRUE)
          }
        )
      )
    }

    self$encoder1 <- self$double_conv(nChn, encoderChn[1])

    self$encoder2 <- torch::nn_sequential(
      torch::nn_max_pool2d(kernel_size=c(2,2), stride=2),
      self$double_conv(encoderChn[1], encoderChn[2],
                         useLeaky=useLeaky, negative_slope=negative_slope)
    )

    self$encoder3 <- torch::nn_sequential(
      torch::nn_max_pool2d(kernel_size=c(2,2), stride=2),
      self$double_conv(encoderChn[2], encoderChn[3],
                         useLeaky=useLeaky, negative_slope=negative_slope)
    )

    self$encoder4 <- torch::nn_sequential(
      torch::nn_max_pool2d(kernel_size=c(2,2), stride=2),
      self$double_conv(encoderChn[3], encoderChn[4],
                         useLeaky=useLeaky, negative_slope=negative_slope)
    )

    self$bottleneck <- torch::nn_sequential(
      torch::nn_max_pool2d(kernel_size=c(2,2), stride=2),
     self$double_conv(encoderChn[4], botChn,
                         useLeaky=useLeaky, negative_slope=negative_slope)
    )

    self$decoder1up <- self$up_conv(botChn, botChn, useLeaky=useLeaky)
    self$decoder1 <- self$double_conv(encoderChn[4] + botChn, decoderChn[1],
                                        useLeaky=useLeaky,
                                        negative_slope=negative_slope)

    self$decoder2up <- self$up_conv(decoderChn[1], decoderChn[1],
                                      useLeaky=useLeaky, negative_slope=negative_slope)
    self$decoder2 <- self$double_conv(encoderChn[3] + decoderChn[1], decoderChn[2],
                                        useLeaky=useLeaky, negative_slope=negative_slope)

    self$decoder3up <- self$up_conv(decoderChn[2], decoderChn[2], useLeaky=useLeaky,
                                      negative_slope=negative_slope)
    self$decoder3 <- self$double_conv(encoderChn[2] + decoderChn[2], decoderChn[3],
                                        useLeaky=useLeaky, negative_slope=negative_slope)

    self$decoder4up <- self$up_conv(decoderChn[3], decoderChn[3],
                                      useLeaky=useLeaky, negative_slope=negative_slope)
    self$decoder4 <- self$double_conv(encoderChn[1] + decoderChn[3], decoderChn[4],
                                        useLeaky=useLeaky, negative_slope=negative_slope)

    self$classifier <- torch::nn_conv2d(decoderChn[4], nCls, kernel_size=c(1,1))
  },
  forward = function(x) {
    e1 <- self$encoder1(x)
    e2 <- self$encoder2(e1)
    e3 <- self$encoder3(e2)
    e4 <- self$encoder4(e3)

    x <- self$bottleneck(e4)

    x <- self$decoder1up(x)
    x <- torch::torch_cat(list(x, e4), dim=2)
    x <- self$decoder1(x)

    x <- self$decoder2up(x)
    x <- torch::torch_cat(list(x, e3), dim=2)
    x <- self$decoder2(x)

    x <- self$decoder3up(x)
    x <- torch::torch_cat(list(x, e2), dim=2)
    x <- self$decoder3(x)

    x <- self$decoder4up(x)
    x <- torch::torch_cat(list(x, e1), dim=2)
    x <- self$decoder4(x)

    x <- self$classifier(x)

    return(x)
  }
)


#dilatedUNet
#attnDSUNet
#resUNet
#vggUNet
#mobileUNet
