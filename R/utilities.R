#' countParams
#'
#' Count the number of trainable parameters in a model.
#'
#' Count the number of trainable parameters in an instantiated model subclassed from nn_module().
#'
#' @param model instantiated model object as nn_module subclass as opposed to luz fitted object.
#' @param test_data Example tensor of the correct shape for the model being explored.
#' @returns Counts of model parameters.
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
#'                     useDS = FALSE,
#'                     enChn = c(16,32,64,128),
#'                     dcChn = c(128,64,32,16),
#'                     btnChn = 256,
#'                     dilRates=c(6,12,18),
#'                     dilChn=c(256,256,256,256),
#'                     negative_slope = 0.01,
#'                     seRatio=8)
#'  t1 <- torch::torch_rand(c(12,4,128,128))
#'  params <- countParams(model, t1)
#'  }
#' @export
countParams <- function(model, test_data) {
  # Extract the named parameters from the model
  params <- model$named_parameters()
  param_names <- names(params)

  # Build a data frame with each parameter's name, element count, and training flag.
  df <- data.frame(
    param = param_names,
    numel = sapply(params, function(p) prod(p$size())),
    trainable = sapply(params, function(p) p$requires_grad),
    stringsAsFactors = FALSE
  )

  # Assume parameter names follow "layer.something" and extract the layer name.
  df$layer <- sapply(df$param, function(x) strsplit(x, "\\.")[[1]][1])

  # Aggregate counts by layer.
  layer_summary <- df |>
    dplyr::group_by(layer) |>
    dplyr::summarise(
      trainable = sum(numel[trainable]),
      non_trainable = sum(numel[!trainable])
    ) |>
    as.data.frame()

  # Compute total parameter counts.
  total_trainable     <- sum(df$numel[df$trainable])
  total_non_trainable <- sum(df$numel[!df$trainable])

  # Run the model on test_data to get output dimensions.
  output <- model(test_data)

  # Extract input and output sizes as integer vectors (e.g. c(batch, channel, height, width)).
  input_size  <- as.integer(test_data$size())
  output_size <- as.integer(output$size())

  # Return results as a list.
  list(
    total_trainable = total_trainable,
    total_non_trainable = total_non_trainable,
    layer_params = layer_summary,
    input_size = input_size,
    output_size = output_size
  )
}

#' callback_save_model_state_dict
#'
#' Save the model state dictionary to disk at the end of each epoch as a .pt file.
#' For use within luz training loop.
#'
#' @param save_dir path and directory to save state dictionary files
#' @param prefix prefix for file name. Default is "model_epoch". The suffix
#' is the epoch number.
#' @returns No R object is returned. The state dictionary is saved to disk as a .pt file.
#' @export
callback_save_model_state_dict <- luz::luz_callback(
  name = "save_model_state_dict",

  initialize = function(save_dir = ".", prefix = "model_epoch") {
    dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
    self$save_dir <- save_dir
    self$prefix <- prefix
  },

  on_epoch_end = function() {
    epoch <- self$ctx$epoch
    model_state <- self$ctx$model$state_dict()
    file_path <- file.path(
      self$save_dir,
      sprintf("%s_%03d.pt", self$prefix, epoch)
    )
    torch_save(model_state, file_path)
    cat(sprintf("Saved model state dict to: %s\n", file_path))
  }
)


cropTensor <- function(inT, crpFactor){
  startCR <- crpFactor+1
  endCR <- inT$size(3) - crpFactor
  outT <- inT[, , startCR:endCR, startCR:endCR]
  return(outT)
}

