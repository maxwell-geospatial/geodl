# defineRegressionLoss

Define a regression loss function for use in a luz training loop.

## Usage

``` r
defineRegressionLoss(type = "mse", delta = 1, reduction = "mean")
```

## Arguments

- type:

  Character string specifying the loss type. One of `"mse"` (default),
  `"mae"`, `"huber"`, or `"log_cosh"`.

- delta:

  Positive numeric scalar controlling the quadratic-to-linear transition
  in the Huber loss. Residuals with absolute value below `delta` are
  penalised quadratically; those above are penalised linearly. Ignored
  for all other loss types. Default is `1.0`.

- reduction:

  Character string controlling how element-wise losses are aggregated
  across the batch. One of `"mean"` (default), `"sum"`, or `"none"`
  (returns the full element-wise loss tensor without reduction).

## Value

An instantiated
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
whose `forward` method accepts `(input, target)` tensors of any matching
shape and returns the scalar (or element-wise when `reduction = "none"`)
loss.

## Details

Returns a
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
that computes one of four common regression losses between a prediction
tensor and a target tensor. The returned object is passed directly to
the `loss` argument of
[`luz::setup()`](https://mlverse.github.io/luz/reference/setup.html).

The four supported loss types are:

- `"mse"`:

  Mean Squared Error. \$\$\frac{1}{n}\sum\_{i}(\hat{y}\_i - y_i)^2\$\$
  Penalises large errors quadratically. Sensitive to outliers because
  the squared term magnifies large residuals.

- `"mae"`:

  Mean Absolute Error. \$\$\frac{1}{n}\sum\_{i}\|\hat{y}\_i - y_i\|\$\$
  More robust to outliers than MSE. The gradient is constant in
  magnitude for all non-zero residuals, which can slow convergence near
  the optimum.

- `"huber"`:

  Huber loss. \$\$L\_\delta(r) = \begin{cases} \tfrac{1}{2}r^2 & \|r\|
  \le \delta \\ \delta\bigl(\|r\| - \tfrac{1}{2}\delta\bigr) & \|r\| \>
  \delta \end{cases}\$\$ where \\r = \hat{y} - y\\. Quadratic for small
  residuals (like MSE) and linear for large residuals (like MAE),
  combining outlier robustness with smooth gradients near zero. The
  transition point is set by `delta`.

- `"log_cosh"`:

  Log-Cosh loss. \$\$\frac{1}{n}\sum\_{i}\log\\\bigl(\cosh(\hat{y}\_i -
  y_i)\bigr)\$\$ Approximates \\\tfrac{1}{2}r^2\\ for small residuals
  and \\\|r\| - \log 2\\ for large residuals. Doubly differentiable
  everywhere, which benefits second-order optimisers and produces very
  smooth loss surfaces. Computed in a numerically stable form: \\\|r\| +
  \log(1 + e^{-2\|r\|}) - \log 2\\.

## Examples

``` r
if (FALSE) { # \dontrun{
# MSE loss (default)
loss_fn <- defineRegressionLoss()

# MAE loss
loss_fn <- defineRegressionLoss(type = "mae")

# Huber loss with a tighter transition point
loss_fn <- defineRegressionLoss(type = "huber", delta = 0.5)

# Log-Cosh loss
loss_fn <- defineRegressionLoss(type = "log_cosh")

# Use in a luz training loop
fitted <- myModel |>
  luz::setup(
    loss      = defineRegressionLoss(type = "huber", delta = 1.0),
    optimizer = torch::optim_adamw
  ) |>
  luz::fit(data = trainDL, epochs = 30, valid_data = valDL)
} # }
```
