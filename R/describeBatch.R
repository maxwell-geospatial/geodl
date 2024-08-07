#' describeBatch
#'
#' Generate summary information for a batch of image chips and masks.
#'
#' The goal of this function is to provide a check of a mini-batch of image chips and associated
#' masks generated by a DataLoader instance using defineSegDataSet(). Summary information includes the mini-batch size
#' (batchSize); image chip data type (imageDataType); mask data type (maskDataType); the shape of the mini-batch of images
#' or predictor variables as mini-batch size, number of channels, width pixel count, and height pixel count (imageShape); the mask shape
#' (maskShape); image band means (bndMns); image band standard deviations (bndSDs);
#' count of pixels in each class in the mini-batch (maskCnts); and minimum (minIndex) and maximum (maxIndex) class indices present in the mini-batch.
#'
#' @param dataLoader Instantiated instance of a DataLoader created using torch::dataloader().
#' @param zeroStart TRUE or FALSE. If class indices start at 0, set this to TRUE. If they start at 1,
#' set this to FALSE. Default is FALSE.
#' @return List object summarizing a mini-batch of image chips and masks.
#' @examples
#' \dontrun{
#' trainStats <- describeBatch(trainDL,
#' tzeroStart=TRUE,
#' tusedDS=FALSE)
#' }
#' @export
describeBatch <- function(dataLoader, zeroStart=FALSE){

  batch1 <- dataLoader$.iter()$.next()

  imgDims <- batch1$image$shape |> as.character()
  mskDims <- batch1$mask$shape |> as.character()
  imgDT <- batch1$image[1]$dtype |> as.character()
  mskDT <- batch1$mask[1]$dtype |> as.character()

  mask <- batch1$mask

  if(zeroStart==TRUE){
    maskFreq <- mask+1
  }else{
    maskFreq <- mask
  }
  freqTble <- torch::torch_bincount(torch::torch_tensor(maskFreq, dtype=torch::torch_long())$flatten())$cpu() |>
    torch::as_array() |>
    as.vector()

  maxIndex <- torch::torch_max(maskFreq)$cpu() |>
    torch::as_array() |>
    as.vector()
  minIndex <- torch::torch_min(maskFreq)$cpu() |>
    torch::as_array() |>
    as.vector()

  bndMns <- torch::torch_mean(batch1$image, dim=c(1,3,4))$cpu() |>
    torch::as_array() |>
    as.vector()

  bndSDs <- torch::torch_std(batch1$image, dim=c(1,3,4))$cpu() |>
    torch::as_array() |>
    as.vector()

  return(list(batchSize = batch1$image$shape[1],
              imageDataType = imgDT,
              maskDataType = mskDT,
              imageShape=imgDims,
              maskShape=mskDims,
              bndMns = bndMns,
              bandSDs = bndSDs,
              maskCount=freqTble,
              minIndex = minIndex,
              maxIndex = maxIndex
              ))
}
