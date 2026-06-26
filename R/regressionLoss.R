#' defineRegressionLoss
#'
#' Define a regression loss function for use in a luz training loop.
#'
#' Returns a \code{torch::nn_module} that computes one of four common
#' regression losses between a prediction tensor and a target tensor. The
#' returned object is passed directly to the \code{loss} argument of
#' \code{luz::setup()}.
#'
#' The four supported loss types are:
#' \describe{
#'   \item{\code{"mse"}}{Mean Squared Error.
#'     \deqn{\frac{1}{n}\sum_{i}(\hat{y}_i - y_i)^2}
#'     Penalises large errors quadratically. Sensitive to outliers because the
#'     squared term magnifies large residuals.}
#'   \item{\code{"mae"}}{Mean Absolute Error.
#'     \deqn{\frac{1}{n}\sum_{i}|\hat{y}_i - y_i|}
#'     More robust to outliers than MSE. The gradient is constant in magnitude
#'     for all non-zero residuals, which can slow convergence near the optimum.}
#'   \item{\code{"huber"}}{Huber loss.
#'     \deqn{L_\delta(r) = \begin{cases}
#'       \tfrac{1}{2}r^2 & |r| \le \delta \\
#'       \delta\bigl(|r| - \tfrac{1}{2}\delta\bigr) & |r| > \delta
#'     \end{cases}}
#'     where \eqn{r = \hat{y} - y}. Quadratic for small residuals (like MSE)
#'     and linear for large residuals (like MAE), combining outlier robustness
#'     with smooth gradients near zero. The transition point is set by
#'     \code{delta}.}
#'   \item{\code{"log_cosh"}}{Log-Cosh loss.
#'     \deqn{\frac{1}{n}\sum_{i}\log\!\bigl(\cosh(\hat{y}_i - y_i)\bigr)}
#'     Approximates \eqn{\tfrac{1}{2}r^2} for small residuals and
#'     \eqn{|r| - \log 2} for large residuals. Doubly differentiable
#'     everywhere, which benefits second-order optimisers and produces very
#'     smooth loss surfaces. Computed in a numerically stable form:
#'     \eqn{|r| + \log(1 + e^{-2|r|}) - \log 2}.}
#' }
#'
#' @param type Character string specifying the loss type. One of \code{"mse"}
#'   (default), \code{"mae"}, \code{"huber"}, or \code{"log_cosh"}.
#' @param delta Positive numeric scalar controlling the quadratic-to-linear
#'   transition in the Huber loss. Residuals with absolute value below
#'   \code{delta} are penalised quadratically; those above are penalised
#'   linearly. Ignored for all other loss types. Default is \code{1.0}.
#' @param reduction Character string controlling how element-wise losses are
#'   aggregated across the batch. One of \code{"mean"} (default),
#'   \code{"sum"}, or \code{"none"} (returns the full element-wise loss
#'   tensor without reduction).
#' @returns An instantiated \code{torch::nn_module} whose \code{forward}
#'   method accepts \code{(input, target)} tensors of any matching shape and
#'   returns the scalar (or element-wise when \code{reduction = "none"}) loss.
#' @examples
#' \dontrun{
#' # MSE loss (default)
#' loss_fn <- defineRegressionLoss()
#'
#' # MAE loss
#' loss_fn <- defineRegressionLoss(type = "mae")
#'
#' # Huber loss with a tighter transition point
#' loss_fn <- defineRegressionLoss(type = "huber", delta = 0.5)
#'
#' # Log-Cosh loss
#' loss_fn <- defineRegressionLoss(type = "log_cosh")
#'
#' # Use in a luz training loop
#' fitted <- myModel |>
#'   luz::setup(
#'     loss      = defineRegressionLoss(type = "huber", delta = 1.0),
#'     optimizer = torch::optim_adamw
#'   ) |>
#'   luz::fit(data = trainDL, epochs = 30, valid_data = valDL)
#' }
#' @export
defineRegressionLoss <- torch::nn_module(

  initialize = function(type      = "mse",
                        delta     = 1.0,
                        reduction = "mean") {

    type      <- match.arg(type,      c("mse", "mae", "huber", "log_cosh"))
    reduction <- match.arg(reduction, c("mean", "sum", "none"))

    if (!is.numeric(delta) || length(delta) != 1L || delta <= 0)
      stop("delta must be a single positive number")

    self$type      <- type
    self$delta     <- delta
    self$reduction <- reduction
  },

  forward = function(input, target) {

    if (self$type == "mse") {
      return(torch::nnf_mse_loss(input, target, reduction = self$reduction))
    }

    if (self$type == "mae") {
      return(torch::nnf_l1_loss(input, target, reduction = self$reduction))
    }

    diff     <- input - target
    abs_diff <- torch::torch_abs(diff)

    if (self$type == "huber") {
      elementwise <- torch::torch_where(
        abs_diff <= self$delta,
        0.5 * diff * diff,
        self$delta * (abs_diff - 0.5 * self$delta)
      )
    } else {
      # log_cosh: numerically stable form avoids cosh overflow for large residuals
      # log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
      elementwise <- abs_diff +
        torch::torch_log1p(torch::torch_exp(-2 * abs_diff)) -
        log(2)
    }

    if (self$reduction == "mean") return(torch::torch_mean(elementwise))
    if (self$reduction == "sum")  return(torch::torch_sum(elementwise))
    elementwise
  }
)
