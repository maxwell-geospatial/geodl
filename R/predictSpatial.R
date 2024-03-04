#' predictSpatial
#'
#' Apply a trained semantic segmentation model to predict back to geospatial raster data
#'
#' This function generates a pixel-by-pixel prediction using input data and a
#' trained semantic segmentation model. Can return either hard classifications, logits, or
#' rescaled logits with a sigmoid or softmax activation applied. Result is written
#' to disk and provided as a spatRaster object.
#'
#' @param imgIn Input image to classify. Can be a file path (full or relative to
#' current working directory) or a spatRaster object. Should have the same number
#' of bands as the data used to train the model. Bands must also be in the same
#' order.
#' @param model Trained model to use to infer to new data.
#' @param predOut Name of output prediction with full path or path relative to
#' the working directory. Must also include the file extension (e.g., ".tif).
#' @param mode Either "multiclass" or "binary". Default is "multiclass". If model
#' returns a single logit for the positive case, should be "binary". If two or more
#' class logits are returned, this should be "multiclass".
#' @param predType "class", "logit", or "prob". Default is "class". Whether to generate a "hard"
#' classification ("class"), logit(s) ("logit"), or rescaled logit(s) ("prob") with a
#' sigmoid or softmax activation applied for the positive class or each predicted class.
#' If "class", a single-band raster of class indices is returned. If "logit" or "prob",
#' the positive class probability is returned as a single-band raster grid for a binary
#' classification. If "logit" or "prob" for a multiclass problem, a multiband raster grid
#' is returned with a channel for each class.
#' @param biThresh When mode = "binary" and predType = "class", threshold to use to indicate
#' the presence class. This threshold is applied to the rescaled logits after a sigmoid
#' activation is applied. Default is 0.5. If rescaled logit is greater than or equal
#' to this threshold, pixel will be mapped to the positive case. Otherwise, it will be
#' mapped to the negative or background class.
#' @param useCUDA TRUE or FALSE. Whether or not to perform the inference on a GPU.
#' If TRUE, the GPU is used. If FALSE, the CPU is used. Must have access to a CUDA-
#' enabled graphics card. Default is FALSE. Note that using a GPU significantly
#' speeds up inference.
#' @param nCls Number of classes being differentiated. Should be 1 for a binary
#' classification problem and the number of classes for a multiclass classification
#' problem.
#' @param chpSize Size of image chips that will be fed through the prediction process.
#' We recommend using the size of the image chips used to train the model. However,
#' this is not strictly necessary.
#' @param stride_x Stride in the x direction. We recommend using a 50% overlap.
#' @param stride_y Stride in the y direction. We recommend using a 50% overlap.
#' @param crop Number of rows and columns to crop from each side of the image chip
#' to reduce edge effects. We recommend at least 20.
#' @param nChn Number of input channels. Default is 3.
#' @param normalize TRUE or FALSE. Whether to apply normalization. If FALSE,
#' bMns and bSDs is ignored. Default is FALSE. If TRUE, you must provide bMns
#' and bSDs. This should match the setting used in defineSegDataSet().
#' @param bMns Vector of band means. Length should be the same as the number of bands.
#' Normalization is applied before any rescaling within the function. This should
#' match the setting used in defineSegDataSet() when model was trained.
#' @param bSDs Vector of band standard deviations. Length should be the same
#' as the number of bands. Normalization is applied before any rescaling within
#' the function. This should match the setting used in defineSegDataSet().
#' @param rescaleFactor A rescaling factor to rescale the bands to 0 to 1. For
#' example, this could be set to 255 to rescale 8-bit data. Default is 1 or no
#' rescaling. This should match the setting used in defineSegDataSet().
#' @param usedDS TRUE or FALSE. If model is configured to use deep supervision,
#' this must be set to TRUE. Default is FALSE, or it is assumed that deep supervision
#' is not used.
#' @return A spatRast object and a raster grid saved to disk of predicted class
#' indices (predType = "class"), logits (predType = "logit"), or rescaled logits
#' (predType = "prob").

#' @export
predictSpatial <- function(imgIn,
                           model,
                           predOut,
                           mode="multiclass",
                           predType="class",
                           biThresh = 0.5,
                           useCUDA=FALSE,
                           nCls,
                           chpSize,
                           stride_x,
                           stride_y,
                           crop,
                           nChn=3,
                           normalize=FALSE,
                           bMns,
                           bSDs,
                           rescaleFactor=1,
                           useDS=FALSE){


  image <- terra::rast(imgIn)

  p_arr <- image

  if(mode == "multiclass" & predType %in% c("logit", "prob")){
    p_arr <- terra::subset(image, 1)
    p_arr <- rep(p_arr, nCls)
    names(p_arr) <- paste0("Class_", rep(1:nCls))
    p_arr[] <- 0
    outGrd <- p_arr
    p_arr <- torch::torch_tensor(terra::as.array(p_arr))
    p_arr <- p_arr$permute(c(3,2,1))
  }else{
    p_arr <- terra::subset(image, 1)
    p_arr[] <- 0
    names(p_arr) <- "Class"
    outGrd <- p_arr
    p_arr <- torch::torch_tensor(terra::as.array(p_arr))
    p_arr <- p_arr$permute(c(3,2,1))
  }

  if(useCUDA == TRUE){
   p_arr <- p_arr$to(device = "cuda")
  }

  if(predType %in% c("logit", "prob")){
    outBands <- nCls
  }else{
    outBands <- 1
  }

  image <- terra::as.array(image)
  image <- torch::torch_tensor(image)

  if(useCUDA == TRUE){
    image <- image$to(device = "cuda")
  }

  image <- image$permute(c(3,2,1))

  if(normalize == TRUE){
    image <- torchvision::transform_normalize(image,
                                              bMns,
                                              bSDs,
                                              inplace = FALSE)
  }

  image <- image/rescaleFactor
  image <- torch::torch_tensor(image, dtype=torch::torch_float32())

  size = chpSize
  stride_x = stride_x
  stride_y = stride_y
  crop = crop
  n_channels = nChn

  cropStart <- crop+1

  across_cnt = image$shape[3]
  down_cnt = image$shape[2]
  tile_size_across = size
  tile_size_down = size
  overlap_across = stride_x
  overlap_down = stride_y
  across = ceiling(across_cnt/overlap_across)
  down = ceiling(down_cnt/overlap_down)
  across_seq = seq(1, across, 1)
  down_seq = seq(1, down, 1)
  across_seq2 = (across_seq*overlap_across)+1
  across_seq2 <- c(1, across_seq2[1:(length(across_seq2)-1)])
  down_seq2 = (down_seq*overlap_down)+1
  down_seq2 <- c(1, down_seq2[1:(length(down_seq2)-1)])

  columnCount <- length(across_seq2)
  rowCount <- length(down_seq2)
  print(paste0("Processing ",
               as.character(columnCount),
               " columns by ",
               as.character(rowCount),
               " rows."))

  #Loop through row/column combinations to make predictions for entire image
  for(c in across_seq2){
    for(r in down_seq2){
      c1 <- c
      r1 <- r
      c2 <- c + (size-1)
      r2 <- r + (size-1)
      #Default
      if(c2 <= across_cnt & r2 <= down_cnt){
        r1b <- r1
        r2b <- r2
        c1b <- c1
        c2b <- c2
      }else if(c2 > across_cnt & r2 <= down_cnt){#Last column
        r1b <- r1
        r2b <- r2
        c1b <- across_cnt - size + 1
        c2b <- across_cnt
      }else if(c2 <= across_cnt & r2 > down_cnt){#Last row
        r1b <- down_cnt - size + 1
        r2b <- down_cnt
        c1b <- c1
        c2b <- c2
      }else{ #Last row, last column
        c1b <- across_cnt - size + 1
        c2b <- across_cnt
        r1b <- down_cnt - size + 1
        r2b <- down_cnt
      }

      ten1 = image[1:n_channels, r1b:r2b, c1b:c2b]
      ten1 <- torch::torch_unsqueeze(ten1, 1)

      preds <- predict(model, ten1)

      if(useDS==TRUE){
        preds <- preds[1]
      }

      if(mode == "multiclass"){
        if(predType == "prob"){
          preds <- torch::nnf_softmax(preds, dim=2)
        }else if(predType == "logit"){
          preds <- preds
        }else{
          preds = torch::torch_argmax(preds, dim=2)
        }
      }else{
        if(predType == "prob"){
          preds <- torch::nnf_sigmoid(preds)
        }else if(predType == "logit"){
          preds <- preds
        }else{
          preds <- torch::nnf_sigmoid(preds)
          preds <- preds >= biThresh
        }
      }

      if(length(dim(preds))==4){
        preds <- torch::torch_squeeze(preds, dim=1)
      }

      if(r1b == 1 & c1b == 1){ #Write first row, first column

        #print("Worked for first row, first column")

        p_arr[1:outBands, r1b:(r2b-crop), c1b:(c2b-crop)] <- preds[1:outBands, 1:(size-crop), 1:(size-crop)]

      }else if(r1b == 1 & c2b == across_cnt){ #Write first row, last column

        #print("Worked for first row, last column")

        p_arr[1:outBands,r1b:(r2b-crop), (c1b+crop):c2b] = preds[1:outBands,1:(size-crop), cropStart:(size)]



      }else if(r2b == down_cnt & c1b == 1){#Write last row, first column

        #print("Worked for last row, first column")

        p_arr[1:outBands,(r1b+crop):r2b, c1b:(c2b-crop)] = preds[1:outBands, cropStart:size, 1:(size-crop)]



      }else if(r2b == down_cnt & c2b == across_cnt){#Write last row, last column

        #print("Worked for last row, last column")

        p_arr[1:outBands,(r1b+crop):r2b, (c1b+crop):c2b] = preds[1:outBands, cropStart:size, cropStart:size]



      }else if((r1b == 1 & c1b != 1) | (r1b == 1 & c2b != across_cnt)){#Write first row

        #print("Worked for first row")

        p_arr[1:outBands,r1b:(r2b-crop), (c1b+crop):(c2b-crop)] = preds[1:outBands,1:(size-crop), cropStart:(size-crop)]



      }else if((r2b == down_cnt & c1b != 1) | (r2b == down_cnt & c2b != across_cnt)){# Write last row

        #print("Worked for last row, last column")

        p_arr[1:outBands,(r1b+crop):r2b, (c1b+crop):(c2b-crop)] = preds[1:outBands, cropStart:size, cropStart:(size-crop)]



      }else if((c1b == 1 & r1b !=1) | (c1b == 1 & r2b != down_cnt)){#Write first column

        #print("Worked for first column")

        p_arr[1:outBands,(r1b+crop):(r2b-crop), c1b:(c2b-crop)] = preds[1:outBands, cropStart:(size-crop), 1:(size-crop)]



      }else if((c2b == across_cnt & r1b != 1) | (c2b == across_cnt & r2b != down_cnt)){# write last column

        #print("Worked for last column")

        p_arr[1:outBands,(r1b+crop):(r2b-crop), (c1b+crop):c2b] = preds[1:outBands, cropStart:(size-crop), cropStart:(size)]



      }else{#Write middle chips

        #print("Worked for middle")

        p_arr[1:outBands,(r1b+crop):(r2b-crop), (c1b+crop):(c2b-crop)] = preds[1:outBands, cropStart:(size-crop), cropStart:(size-crop)]

      }
    }

  }

  p_arr2 <- p_arr$permute(c(2,3,1))$to(device="cpu")

  p_arrA <- torch::as_array(p_arr2)
  outGrd[] <- p_arrA
  terra::writeRaster(outGrd, predOut, overwrite=TRUE)

  print("Prediction completed!")
  return(outGrd)
}
