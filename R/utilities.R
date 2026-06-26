#' countParams
#'
#' Count the number of trainable parameters in a model.
#'
#' Count the number of trainable parameters in an instantiated model subclassed from nn_module().
#'
#' @param model instantiated model object as nn_module subclass as opposed to luz fitted object.
#' @param test_data Example tensor of the correct shape for the model being explored.
#' @returns Counts of model parameters.
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
#' @importFrom luz luz_callback
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
    torch::torch_save(model_state, file_path)
    cat(sprintf("Saved model state dict to: %s\n", file_path))
  }
)


#' callback_per_stage_optimizer
#'
#' Replace the luz optimizer with per-stage parameter groups at the start of training.
#'
#' A luz callback that calls \code{model$get_param_groups()} on the instantiated
#' model at the start of \code{fit()} and replaces the default flat-parameter
#' optimizer with one built from the per-stage parameter groups. This allows
#' different learning rates to be assigned to different stages of the network
#' when using models that support \code{stageLRs} (e.g. \code{defineUNet},
#' \code{defineMobileUNet}, \code{defineUNet3p}, \code{defineEfficientUNetB2}).
#'
#' This callback must be placed \strong{before} any learning rate scheduler
#' callback in the \code{callbacks} list so that the scheduler is initialised
#' against the per-stage optimizer rather than the original flat one.
#'
#' When using \code{torch::lr_one_cycle} alongside this callback, pass
#' \code{max_lr}, \code{max_momentum}, and \code{base_momentum} as vectors of
#' length equal to the number of stages so that the scheduler can assign
#' per-group values. A proportionally scaled \code{max_lr} vector can be
#' constructed as \code{stage_lrs / max(stage_lrs) * global_max_lr}.
#'
#' @param optimizer_fn Optimizer constructor function. Defaults to
#'   \code{torch::optim_adamw}. Any torch optimizer that accepts a list of
#'   parameter groups as its first argument can be used.
#' @param ... Additional arguments forwarded to \code{optimizer_fn}, such as
#'   \code{weight_decay}.
#' @returns A luz callback object.
#' @importFrom luz luz_callback
#' @export
callback_per_stage_optimizer <- luz::luz_callback(
  name = "per_stage_optimizer",

  initialize = function(optimizer_fn = torch::optim_adamw, ...) {
    self$optimizer_fn <- optimizer_fn
    self$opt_args     <- list(...)
  },

  on_fit_begin = function() {
    param_groups <- self$ctx$model$get_param_groups()
    self$ctx$optimizers[[1L]] <- do.call(
      self$optimizer_fn,
      c(list(param_groups), self$opt_args)
    )
  }
)


#' callback_freeze_stages
#'
#' Progressively freeze model stage parameters during training.
#'
#' A luz callback that sets \code{requires_grad = FALSE} on the parameters of
#' specified model components at defined epoch boundaries. Multiple freeze
#' actions at different epochs are supported via the \code{schedule} argument.
#' Once frozen, a component's parameters receive no gradient and are not
#' updated by the optimizer for the remainder of training.
#'
#' Component names must match the submodule field names used inside the model.
#' For \code{defineUNet} and \code{defineUNet3p} the stage names are
#' \code{"e1"}, \code{"e2"}, \code{"e3"}, \code{"e4"}, \code{"btn"},
#' \code{"d1"}, \code{"d2"}, \code{"d3"}, \code{"d4"}. For
#' \code{defineMobileUNet} and \code{defineEfficientUNetB2} there are five
#' encoder stages (\code{"e1"} through \code{"e5"}) and five decoder stages
#' (\code{"d1"} through \code{"d5"}). Attention gate and squeeze-and-excitation
#' submodules (\code{"ag1"}, \code{"se1"}, etc.) can also be named.
#'
#' Freezing takes effect at the \strong{end} of the specified epoch, so frozen
#' components are inactive from the following epoch onward. A message is printed
#' each time a component is frozen, reporting the component name and parameter
#' count.
#'
#' @param schedule A list of named lists, each specifying one freeze action.
#'   Each entry must contain:
#'   \describe{
#'     \item{\code{epoch}}{Single integer. The training epoch after which the
#'       listed components are frozen.}
#'     \item{\code{components}}{Character vector of submodule names to freeze
#'       at this epoch.}
#'   }
#' @returns A luz callback object.
#' @examples
#' \dontrun{
#' # Freeze early encoder stages after epoch 5,
#' # then the remaining encoder and bottleneck after epoch 10
#' callback_freeze_stages(
#'   schedule = list(
#'     list(epoch = 5,  components = c("e1", "e2")),
#'     list(epoch = 10, components = c("e3", "e4", "btn"))
#'   )
#' )
#' }
#' @importFrom luz luz_callback
#' @export
callback_freeze_stages <- luz::luz_callback(
  name = "freeze_stages",

  initialize = function(schedule) {
    if (!is.list(schedule))
      stop("schedule must be a list of named lists with 'epoch' and 'components' elements")
    for (i in seq_along(schedule)) {
      item <- schedule[[i]]
      if (!is.list(item) || is.null(item$epoch) || is.null(item$components))
        stop(sprintf(
          "schedule[[%d]] must be a list with 'epoch' and 'components' elements", i
        ))
      if (!is.numeric(item$epoch) || length(item$epoch) != 1L)
        stop(sprintf("schedule[[%d]]$epoch must be a single integer", i))
      if (!is.character(item$components) || length(item$components) == 0L)
        stop(sprintf(
          "schedule[[%d]]$components must be a non-empty character vector", i
        ))
    }
    self$schedule <- schedule
    self$frozen   <- character(0L)
  },

  on_epoch_end = function() {
    epoch <- self$ctx$epoch
    for (item in self$schedule) {
      if (epoch != as.integer(item$epoch)) next
      for (comp_name in item$components) {
        if (comp_name %in% self$frozen) next
        comp <- self$ctx$model[[comp_name]]
        if (is.null(comp)) {
          warning(sprintf(
            "callback_freeze_stages: '%s' not found in model — skipping.",
            comp_name
          ))
          next
        }
        for (p in comp$parameters) p$requires_grad_(FALSE)
        n_params <- sum(vapply(comp$parameters, function(p) as.numeric(p$numel()), numeric(1L)))
        self$frozen <- c(self$frozen, comp_name)
        cat(sprintf(
          "Epoch %d: froze '%s' (%s parameters)\n",
          epoch, comp_name, format(n_params, big.mark = ",")
        ))
      }
    }
  }
)


#' callback_unfreeze_stages
#'
#' Progressively unfreeze previously frozen model stage parameters during training.
#'
#' A luz callback that sets \code{requires_grad = TRUE} on the parameters of
#' specified model components at defined epoch boundaries, reversing an earlier
#' freeze applied by \code{callback_freeze_stages} or by pre-training setup
#' (e.g. \code{freezeEncoder = TRUE}). Multiple unfreeze actions at different
#' epochs are supported via the \code{schedule} argument.
#'
#' Component names must match the submodule field names used inside the model.
#' For \code{defineUNet} and \code{defineUNet3p} the stage names are
#' \code{"e1"}, \code{"e2"}, \code{"e3"}, \code{"e4"}, \code{"btn"},
#' \code{"d1"}, \code{"d2"}, \code{"d3"}, \code{"d4"}. For
#' \code{defineMobileUNet} and \code{defineEfficientUNetB2} there are five
#' encoder stages (\code{"e1"} through \code{"e5"}) and five decoder stages
#' (\code{"d1"} through \code{"d5"}).
#'
#' Unfreezing takes effect at the \strong{end} of the specified epoch, so
#' unfrozen components begin receiving gradient updates from the following
#' epoch onward. A message is printed each time a component is unfrozen,
#' reporting the component name and parameter count. The optimizer state for
#' the newly unfrozen parameters is not reset, so any accumulated momentum
#' carries over.
#'
#' \code{callback_unfreeze_stages} and \code{callback_freeze_stages} can be
#' used together to implement warm-up schedules where the encoder is initially
#' frozen while the decoder trains, then progressively unfrozen.
#'
#' @param schedule A list of named lists, each specifying one unfreeze action.
#'   Each entry must contain:
#'   \describe{
#'     \item{\code{epoch}}{Single integer. The training epoch after which the
#'       listed components are unfrozen.}
#'     \item{\code{components}}{Character vector of submodule names to unfreeze
#'       at this epoch.}
#'   }
#' @returns A luz callback object.
#' @examples
#' \dontrun{
#' # Train with encoder frozen for 5 epochs, then unfreeze progressively
#' callback_unfreeze_stages(
#'   schedule = list(
#'     list(epoch = 5,  components = c("e4", "e5")),
#'     list(epoch = 10, components = c("e2", "e3")),
#'     list(epoch = 15, components = c("e1"))
#'   )
#' )
#' }
#' @importFrom luz luz_callback
#' @export
callback_unfreeze_stages <- luz::luz_callback(
  name = "unfreeze_stages",

  initialize = function(schedule) {
    if (!is.list(schedule))
      stop("schedule must be a list of named lists with 'epoch' and 'components' elements")
    for (i in seq_along(schedule)) {
      item <- schedule[[i]]
      if (!is.list(item) || is.null(item$epoch) || is.null(item$components))
        stop(sprintf(
          "schedule[[%d]] must be a list with 'epoch' and 'components' elements", i
        ))
      if (!is.numeric(item$epoch) || length(item$epoch) != 1L)
        stop(sprintf("schedule[[%d]]$epoch must be a single integer", i))
      if (!is.character(item$components) || length(item$components) == 0L)
        stop(sprintf(
          "schedule[[%d]]$components must be a non-empty character vector", i
        ))
    }
    self$schedule  <- schedule
    self$unfrozen  <- character(0L)
  },

  on_epoch_end = function() {
    epoch <- self$ctx$epoch
    for (item in self$schedule) {
      if (epoch != as.integer(item$epoch)) next
      for (comp_name in item$components) {
        if (comp_name %in% self$unfrozen) next
        comp <- self$ctx$model[[comp_name]]
        if (is.null(comp)) {
          warning(sprintf(
            "callback_unfreeze_stages: '%s' not found in model — skipping.",
            comp_name
          ))
          next
        }
        for (p in comp$parameters) p$requires_grad_(TRUE)
        n_params <- sum(vapply(comp$parameters, function(p) as.numeric(p$numel()), numeric(1L)))
        self$unfrozen <- c(self$unfrozen, comp_name)
        cat(sprintf(
          "Epoch %d: unfroze '%s' (%s parameters)\n",
          epoch, comp_name, format(n_params, big.mark = ",")
        ))
      }
    }
  }
)


#' callback_ema
#'
#' Maintain an exponential moving average (EMA) of model weights during training.
#'
#' A luz callback that keeps a shadow copy of all model parameters and updates
#' them as an exponential moving average of the training weights after each
#' optimizer step. The EMA weights are swapped into the model during validation
#' so that validation metrics reflect EMA performance, then the training weights
#' are restored so training continues normally.
#'
#' When \code{save_dir} is supplied, a complete model state dictionary with the
#' EMA parameter values merged in is written to disk at the end of each epoch.
#' The saved file is fully compatible with
#' \code{model$load_state_dict(torch::torch_load(path))} and can be used
#' directly for inference without any additional steps. Non-parameter buffers
#' such as batch normalisation running statistics are taken from the live model
#' and are not averaged, which is the standard EMA practice.
#'
#' The update rule applied after each training batch is:
#' \deqn{\hat{\theta} \leftarrow \mathrm{decay} \times \hat{\theta} +
#'   (1 - \mathrm{decay}) \times \theta}
#' where \eqn{\hat{\theta}} are the shadow (EMA) weights and \eqn{\theta} are
#' the current training weights.
#'
#' During \code{warmup_steps} the shadow weights are set equal to the current
#' training weights at every batch rather than applying the EMA rule. This
#' prevents early random-initialisation values from being mixed into the
#' average.
#'
#' @param decay Numeric scalar in \code{(0, 1)} controlling the smoothing
#'   strength. Higher values (e.g. \code{0.999}) produce a slower-moving
#'   average; lower values (e.g. \code{0.99}) track the training weights more
#'   closely. Default \code{0.999}.
#' @param warmup_steps Non-negative integer giving the number of optimizer
#'   steps (batches) at the start of training during which the shadow weights
#'   track the training weights exactly before EMA decay begins. Default
#'   \code{0L}.
#' @param save_dir Optional character string path to a directory. When
#'   supplied, the EMA state dictionary is saved here at the end of every
#'   epoch as a \code{.pt} file. Default \code{NULL} (no saving).
#' @param prefix File-name prefix used when \code{save_dir} is set. Files are
#'   named \code{<prefix>_NNN.pt} where \code{NNN} is the zero-padded epoch
#'   number. Default \code{"ema_epoch"}.
#' @returns A luz callback object.
#' @importFrom luz luz_callback
#' @export
callback_ema <- luz::luz_callback(
  name = "ema",

  initialize = function(decay        = 0.999,
                        warmup_steps = 0L,
                        save_dir     = NULL,
                        prefix       = "ema_epoch") {
    if (!is.numeric(decay) || decay <= 0 || decay >= 1)
      stop("decay must be a numeric value strictly between 0 and 1")
    warmup_steps <- as.integer(warmup_steps)
    if (warmup_steps < 0L)
      stop("warmup_steps must be a non-negative integer")
    self$decay        <- decay
    self$warmup_steps <- warmup_steps
    self$save_dir     <- save_dir
    self$prefix       <- prefix
    self$shadow       <- NULL
    self$backup       <- NULL
    self$step         <- 0L
    if (!is.null(save_dir))
      dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
  },

  on_train_start = function() {
    params <- self$ctx$model$named_parameters()
    self$shadow <- lapply(params, function(p) p$detach()$clone())
  },

  on_train_batch_end = function() {
    self$step  <- self$step + 1L
    params     <- self$ctx$model$named_parameters()
    if (self$step <= self$warmup_steps) {
      for (nm in names(self$shadow))
        self$shadow[[nm]]$copy_(params[[nm]]$detach())
      return(invisible(NULL))
    }
    d <- self$decay
    for (nm in names(self$shadow))
      self$shadow[[nm]] <- self$shadow[[nm]] * d + params[[nm]]$detach() * (1 - d)
  },

  on_valid_epoch_begin = function() {
    params      <- self$ctx$model$named_parameters()
    self$backup <- lapply(params, function(p) p$detach()$clone())
    for (nm in names(self$shadow))
      params[[nm]]$copy_(self$shadow[[nm]])
  },

  on_valid_epoch_end = function() {
    params <- self$ctx$model$named_parameters()
    for (nm in names(self$backup))
      params[[nm]]$copy_(self$backup[[nm]])
    self$backup <- NULL
  },

  on_epoch_end = function() {
    if (is.null(self$save_dir)) return(invisible(NULL))
    epoch <- self$ctx$epoch
    state <- self$ctx$model$state_dict()
    for (nm in names(self$shadow))
      state[[nm]] <- self$shadow[[nm]]$clone()
    file_path <- file.path(
      self$save_dir,
      sprintf("%s_%03d.pt", self$prefix, epoch)
    )
    torch::torch_save(state, file_path)
    cat(sprintf("Saved EMA state dict to: %s\n", file_path))
  }
)


#' make_ds_weights
#'
#' Create a shared mutable weight container for deep supervision loss weighting.
#'
#' Creates an R environment that holds the current per-output loss weights for
#' deep supervision training. Pass the returned object to both
#' \code{defineUnifiedFocalLossDS} (via the \code{weight_env} argument) and
#' \code{callback_ds_weight_decay} so the callback can update the weights the
#' loss function reads each epoch.
#'
#' @param weights Numeric vector of initial loss weights. The first element is
#'   the primary (full-resolution) output; subsequent elements are ancillary
#'   outputs in coarsest-to-finest order. All values must be non-negative.
#' @returns An R environment with a single element \code{$values} containing
#'   the current weight vector. Passed by reference, so updates made by
#'   \code{callback_ds_weight_decay} are immediately visible to \code{ds_loss}.
#' @export
make_ds_weights <- function(weights) {
  if (!is.numeric(weights) || any(weights < 0)) {
    stop("weights must be a non-negative numeric vector")
  }
  e <- new.env(parent = emptyenv())
  e$values <- as.numeric(weights)
  e
}

#' callback_ds_weight_decay
#'
#' Decay ancillary deep supervision loss weights during luz training.
#'
#' A luz callback that reduces the weights of ancillary outputs (all elements
#' of \code{weight_env$values} except the first) toward zero over the decaying
#' portion of training, leaving the primary output weight unchanged. The
#' fraction of epochs reserved for primary-loss-only training is controlled by
#' \code{primary_only_frac} (default \code{0.1}, i.e. the final 10\% of
#' epochs).
#'
#' Two decay schedules are supported:
#' \describe{
#'   \item{\code{"linear"}}{Ancillary weights decrease linearly from their
#'     initial values to zero. Matches the nnU-Net strategy.}
#'   \item{\code{"cosine"}}{Ancillary weights follow a cosine curve, decaying
#'     slowly at first and more rapidly through mid-training.}
#' }
#'
#' @param weight_env A weight environment created by \code{make_ds_weights} and
#'   shared with \code{ds_loss}.
#' @param decay Character string, either \code{"linear"} (default) or
#'   \code{"cosine"}.
#' @param primary_only_frac Numeric value in \code{(0, 1)} specifying the
#'   fraction of total epochs at the end of training during which only the
#'   primary loss is used (ancillary weights are zero). Defaults to \code{0.1}.
#' @returns A luz callback object.
#' @importFrom luz luz_callback
#' @export
callback_ds_weight_decay <- luz::luz_callback(
  name = "ds_weight_decay",

  initialize = function(weight_env, decay = "linear", primary_only_frac = 0.1) {
    if (!decay %in% c("linear", "cosine")) {
      stop('decay must be "linear" or "cosine"')
    }
    if (!is.numeric(primary_only_frac) || primary_only_frac <= 0 || primary_only_frac >= 1) {
      stop("primary_only_frac must be a numeric value strictly between 0 and 1")
    }
    self$env              <- weight_env
    self$init             <- weight_env$values
    self$decay            <- decay
    self$primary_only_frac <- primary_only_frac
  },

  on_epoch_end = function() {
    t  <- self$ctx$epoch
    T  <- self$ctx$max_epochs * (1 - self$primary_only_frac)
    scale <- if (self$decay == "linear") {
      max(0, 1 - t / T)
    } else {
      max(0, 0.5 * (1 + cos(pi * t / T)))
    }
    self$env$values[-1] <- self$init[-1] * scale
  }
)

#' callback_training_log
#'
#' Log per-epoch learning rates and deep supervision weights to a CSV file.
#'
#' A luz callback that writes one row per training epoch recording the current
#' optimizer learning rate(s) and, optionally, the deep supervision loss
#' weights. Column structure is determined automatically on the first epoch:
#' \itemize{
#'   \item If the model uses per-stage learning rates via \code{get_param_groups},
#'     each stage is logged in its own column (\code{lr_1}, \code{lr_2}, ...).
#'   \item If a single learning rate is used, a single \code{lr} column is written.
#'   \item If \code{weight_env} is supplied, columns \code{ds_primary},
#'     \code{ds_aux1}, \code{ds_aux2}, ... are appended for each weight.
#' }
#'
#' @param path File path for the output CSV. The parent directory is created if
#'   it does not exist. Any existing file at this path is overwritten.
#' @param weight_env Optional. A weight environment created by
#'   \code{make_ds_weights}. Supply when using deep supervision so that the
#'   current weights are logged each epoch. Omit or pass \code{NULL} when not
#'   using deep supervision.
#' @returns A luz callback object.
#' @importFrom luz luz_callback
#' @export
callback_training_log <- luz::luz_callback(
  name = "training_log",

  initialize = function(path, weight_env = NULL) {
    self$path       <- path
    self$weight_env <- weight_env
    self$col_names  <- NULL
  },

  on_epoch_end = function() {
    opt    <- self$ctx$optimizers[[1L]]
    if (is.null(opt)) opt <- self$ctx$optimizer
    groups <- opt$param_groups
    lrs    <- vapply(groups, function(g) as.numeric(g$lr), numeric(1L))
    epoch  <- self$ctx$epoch
    vals   <- c(epoch, lrs)

    if (!is.null(self$weight_env)) vals <- c(vals, self$weight_env$values)

    if (is.null(self$col_names)) {
      lr_cols <- if (length(lrs) == 1L) "lr" else paste0("lr_", seq_along(lrs))
      ds_cols <- if (!is.null(self$weight_env)) {
        n_w <- length(self$weight_env$values)
        c("ds_primary", if (n_w > 1L) paste0("ds_aux", seq_len(n_w - 1L)))
      } else character(0L)
      self$col_names <- c("epoch", lr_cols, ds_cols)
      dir.create(dirname(self$path), showWarnings = FALSE, recursive = TRUE)
      cat(paste(self$col_names, collapse = ","), "\n", file = self$path, sep = "")
    }

    cat(paste(vals, collapse = ","), "\n", file = self$path, append = TRUE, sep = "")
  }
)

#' plotTrainingSchedule
#'
#' Visualise per-stage base learning rates, an optional LR scheduler trace,
#' and deep supervision weight decay without running a training loop.
#'
#' Produces up to three panels depending on inputs. The left panel always shows
#' a bar chart of per-stage base learning rates. When \code{scheduler} is
#' supplied a centre panel shows the effective learning rate curve over all
#' epochs. When \code{initialWeights} is supplied a right panel shows how each
#' deep supervision weight evolves over epochs, replicating the exact decay
#' applied by \code{callback_ds_weight_decay}.
#'
#' The \code{scheduler} list accepts three types, each mirroring a torch R
#' scheduler:
#' \describe{
#'   \item{\code{"cosine_annealing"}}{Mirrors \code{lr_cosine_annealing}.
#'     Named elements: \code{base_lr} (starting LR; defaults to
#'     \code{max(stageLRs)}), \code{T_max} (epochs per half-cycle; defaults to
#'     \code{epochs}), \code{eta_min} (minimum LR, default \code{0}). The LR
#'     oscillates between \code{base_lr} and \code{eta_min} with period
#'     \code{2 * T_max}.}
#'   \item{\code{"one_cycle"}}{Mirrors \code{lr_one_cycle}.
#'     Named elements: \code{max_lr} (peak LR; defaults to
#'     \code{max(stageLRs)}), \code{pct_start} (warmup fraction, default
#'     \code{0.3}), \code{div_factor} (initial_lr = max_lr / div_factor,
#'     default \code{25}), \code{final_div_factor} (min_lr = initial_lr /
#'     final_div_factor, default \code{1e4}), \code{anneal_strategy}
#'     (\code{"cos"} or \code{"linear"}, default \code{"cos"}).}
#'   \item{\code{"step"}}{Mirrors \code{lr_step}.
#'     Named elements: \code{base_lr} (starting LR; defaults to
#'     \code{max(stageLRs)}), \code{step_size} (epochs between LR drops;
#'     defaults to \code{max(1, floor(epochs / 10))}), \code{gamma}
#'     (multiplicative decay factor per drop, default \code{0.1}).}
#' }
#'
#' @param stageLRs Numeric vector of per-stage base learning rates matching the
#'   \code{stageLRs} argument passed to the model. A single value produces a
#'   one-bar chart.
#' @param epochs Total number of training epochs.
#' @param initialWeights Optional numeric vector of initial deep supervision
#'   loss weights as passed to \code{make_ds_weights}. When \code{NULL} the DS
#'   weight panel is omitted.
#' @param decay Character string, either \code{"linear"} (default) or
#'   \code{"cosine"}, matching the \code{decay} argument passed to
#'   \code{callback_ds_weight_decay}.
#' @param primary_only_frac Numeric value in \code{(0, 1)} specifying the
#'   fraction of total epochs reserved for primary-loss-only training at the
#'   end of the run, matching the \code{primary_only_frac} argument passed to
#'   \code{callback_ds_weight_decay}. Defaults to \code{0.1}.
#' @param stageNames Optional character vector of stage labels for the learning
#'   rate chart. Defaults to \code{"Stage 1"}, \code{"Stage 2"}, ...
#' @param scheduler Optional named list specifying an LR scheduler to
#'   visualise. Must contain at least \code{type} (\code{"cosine_annealing"},
#'   \code{"one_cycle"}, or \code{"step"}). See Details for per-type parameters.
#' @returns Invisibly returns a named list with element \code{lr} (a data frame
#'   of stage names and learning rates), \code{lr_trace} (a data frame of
#'   epoch-by-LR values when \code{scheduler} is supplied), and \code{weights}
#'   (a data frame of epoch-by-weight values when \code{initialWeights} is
#'   supplied).
#' @export
plotTrainingSchedule <- function(stageLRs,
                                 epochs,
                                 initialWeights    = NULL,
                                 decay             = "linear",
                                 primary_only_frac = 0.1,
                                 stageNames        = NULL,
                                 scheduler         = NULL) {

  if (!is.numeric(stageLRs) || any(stageLRs <= 0))
    stop("stageLRs must be a positive numeric vector")
  epochs <- as.integer(epochs)
  if (epochs < 1L)
    stop("epochs must be a positive integer")
  if (!decay %in% c("linear", "cosine"))
    stop('decay must be "linear" or "cosine"')
  if (!is.numeric(primary_only_frac) || primary_only_frac <= 0 || primary_only_frac >= 1)
    stop("primary_only_frac must be a numeric value strictly between 0 and 1")

  n_stages <- length(stageLRs)
  if (is.null(stageNames))
    stageNames <- paste0("Stage ", seq_len(n_stages))
  if (length(stageNames) != n_stages)
    stop("stageNames length must match stageLRs length")

  has_ds   <- !is.null(initialWeights)
  has_sched <- !is.null(scheduler)

  if (has_ds) {
    if (!is.numeric(initialWeights) || any(initialWeights < 0))
      stop("initialWeights must be a non-negative numeric vector")
  }

  # --- Compute scheduler LR trace --------------------------------------------
  if (has_sched) {
    if (!is.list(scheduler) || is.null(scheduler$type))
      stop('scheduler must be a named list with at least a "type" element')
    sched_type <- scheduler$type
    if (!sched_type %in% c("cosine_annealing", "one_cycle", "step"))
      stop('scheduler$type must be "cosine_annealing", "one_cycle", or "step"')

    if (sched_type == "cosine_annealing") {
      base_lr_s <- if (!is.null(scheduler$base_lr)) scheduler$base_lr           else max(stageLRs)
      T_max     <- if (!is.null(scheduler$T_max))   as.integer(scheduler$T_max) else epochs
      eta_min   <- if (!is.null(scheduler$eta_min)) scheduler$eta_min           else 0

      if (T_max < 1L)
        stop("scheduler$T_max must be a positive integer")
      if (eta_min < 0)
        stop("scheduler$eta_min must be non-negative")

      lr_trace <- numeric(epochs)
      for (t in seq_len(epochs)) {
        lr_trace[t] <- eta_min + (base_lr_s - eta_min) * (1 + cos(pi * t / T_max)) / 2
      }
      sched_title <- paste0("Cosine Annealing (T_max=", T_max, ")")

    } else if (sched_type == "one_cycle") {
      max_lr_s         <- if (!is.null(scheduler$max_lr))           scheduler$max_lr          else max(stageLRs)
      pct_start        <- if (!is.null(scheduler$pct_start))        scheduler$pct_start       else 0.3
      div_factor       <- if (!is.null(scheduler$div_factor))       scheduler$div_factor      else 25
      final_div_factor <- if (!is.null(scheduler$final_div_factor)) scheduler$final_div_factor else 1e4
      anneal_strategy  <- if (!is.null(scheduler$anneal_strategy))  scheduler$anneal_strategy else "cos"

      if (!anneal_strategy %in% c("cos", "linear"))
        stop('scheduler$anneal_strategy must be "cos" or "linear"')

      initial_lr <- max_lr_s / div_factor
      min_lr_s   <- initial_lr / final_div_factor
      t_warmup   <- round(pct_start * epochs)
      t_anneal   <- epochs - t_warmup

      lr_trace <- numeric(epochs)
      for (t in seq_len(epochs)) {
        t0 <- t - 1L
        if (t0 < t_warmup) {
          pct <- if (t_warmup > 0L) t0 / t_warmup else 1
          lr_trace[t] <- if (anneal_strategy == "cos")
            initial_lr + (max_lr_s - initial_lr) * (1 - cos(pi * pct)) / 2
          else
            initial_lr + (max_lr_s - initial_lr) * pct
        } else {
          pct <- if (t_anneal > 0L) (t0 - t_warmup) / t_anneal else 1
          lr_trace[t] <- if (anneal_strategy == "cos")
            min_lr_s + (max_lr_s - min_lr_s) * (1 + cos(pi * pct)) / 2
          else
            max_lr_s - (max_lr_s - min_lr_s) * pct
        }
      }
      sched_title <- paste0("One Cycle LR (", anneal_strategy, ")")

    } else {  # step
      base_lr_s <- if (!is.null(scheduler$base_lr))   scheduler$base_lr              else max(stageLRs)
      step_size <- if (!is.null(scheduler$step_size)) as.integer(scheduler$step_size) else max(1L, floor(epochs / 10L))
      gamma     <- if (!is.null(scheduler$gamma))     scheduler$gamma                else 0.1

      if (step_size < 1L)
        stop("scheduler$step_size must be a positive integer")

      lr_trace <- numeric(epochs)
      for (t in seq_len(epochs)) {
        lr_trace[t] <- base_lr_s * gamma^floor(t / step_size)
      }
      sched_title <- paste0("Step LR (step=", step_size, ", gamma=", gamma, ")")
    }
  }

  # --- Layout ----------------------------------------------------------------
  n_panels <- 1L + has_sched + has_ds

  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par))

  if (n_panels > 1L)
    graphics::layout(matrix(seq_len(n_panels), nrow = 1L))
  graphics::par(mar = c(7, 4, 4, 2))

  # --- Panel 1: per-stage learning rates -------------------------------------
  use_log <- (max(stageLRs) / min(stageLRs)) > 10
  lr_cols <- grDevices::colorRampPalette(c("#2E86AB", "#A23B72"))(n_stages)

  bp <- graphics::barplot(
    stageLRs,
    names.arg = stageNames,
    las       = 2,
    col       = lr_cols,
    border    = NA,
    ylab      = if (use_log) "Learning Rate (log scale)" else "Learning Rate",
    main      = "Per-Stage Base Learning Rates",
    log       = if (use_log) "y" else ""
  )
  graphics::text(
    x      = bp,
    y      = stageLRs,
    labels = formatC(stageLRs, format = "e", digits = 1L),
    pos    = 3L,
    cex    = 0.72,
    xpd    = TRUE
  )

  out <- list(lr = data.frame(stage = stageNames, lr = stageLRs,
                               stringsAsFactors = FALSE))

  # --- Panel 2 (optional): LR scheduler trace --------------------------------
  if (has_sched) {
    graphics::par(mar = c(5, 4, 4, 2))

    lr_min_pos  <- min(lr_trace[lr_trace > 0])
    use_log_s   <- (max(lr_trace) / lr_min_pos) > 100

    graphics::plot(
      x    = seq_len(epochs),
      y    = lr_trace,
      type = "l",
      lwd  = 2L,
      col  = "#2E86AB",
      xlab = "Epoch",
      ylab = if (use_log_s) "Learning Rate (log scale)" else "Learning Rate",
      main = sched_title,
      log  = if (use_log_s) "y" else ""
    )

    # Dashed vertical lines at stage boundaries (assumes equal epochs per stage)
    if (n_stages > 1L) {
      eps_per_stage <- epochs / n_stages
      bounds <- eps_per_stage * seq_len(n_stages - 1L)
      graphics::abline(v = bounds, lty = 2L, col = "grey70")
      y_top <- if (use_log_s) exp(log(max(lr_trace)) * 0.97) else max(lr_trace) * 0.97
      graphics::text(
        x      = bounds - eps_per_stage / 2,
        y      = y_top,
        labels = stageNames[-n_stages],
        cex    = 0.72,
        col    = "grey40"
      )
    }

    out$lr_trace <- data.frame(epoch = seq_len(epochs), lr = lr_trace)
  }

  # --- Panel 3 (optional): deep supervision weight decay ---------------------
  if (has_ds) {
    graphics::par(mar = c(7, 4, 4, 2))

    n_w     <- length(initialWeights)
    T_decay <- epochs * (1 - primary_only_frac)

    weight_mat <- matrix(0, nrow = epochs, ncol = n_w)
    for (t in seq_len(epochs)) {
      scale <- if (decay == "linear") {
        max(0, 1 - t / T_decay)
      } else {
        max(0, 0.5 * (1 + cos(pi * t / T_decay)))
      }
      weight_mat[t, 1L]  <- initialWeights[1L]
      weight_mat[t, -1L] <- initialWeights[-1L] * scale
    }

    w_labels <- c("Primary", paste0("Aux ", seq_len(n_w - 1L)))
    w_cols   <- c("#2E86AB", "#E84855", "#F4A261", "#3BB273")[seq_len(n_w)]

    graphics::matplot(
      x    = seq_len(epochs),
      y    = weight_mat,
      type = "l",
      lty  = 1L,
      lwd  = 2L,
      col  = w_cols,
      xlab = "Epoch",
      ylab = "Weight",
      main = paste0("DS Weight Decay (", decay, ")"),
      ylim = c(0, max(initialWeights) * 1.08)
    )
    graphics::abline(v = T_decay, lty = 2L, col = "grey60")
    graphics::text(
      x      = T_decay,
      y      = max(initialWeights) * 1.04,
      labels = paste0("Aux = 0\n(epoch ", round(T_decay), ")"),
      pos    = 2L,
      cex    = 0.75,
      col    = "grey40"
    )
    graphics::legend("topright", legend = w_labels, col = w_cols,
                     lwd = 2L, bty = "n", cex = 0.85)

    colnames(weight_mat) <- w_labels
    out$weights <- as.data.frame(cbind(epoch = seq_len(epochs), weight_mat))
  }

  invisible(out)
}

cropTensor <- function(inT, crpFactor){
  startCR <- crpFactor+1
  endCR <- inT$size(3) - crpFactor
  outT <- inT[, , startCR:endCR, startCR:endCR]
  return(outT)
}

torch_is_available <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    return(FALSE)
  }

  out <- tryCatch({
    torch::torch_tensor(1)
    TRUE
  }, error = function(e) {
    FALSE
  })

  isTRUE(out)
}
