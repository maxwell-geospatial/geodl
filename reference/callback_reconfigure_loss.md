# callback_reconfigure_loss

Reconfigure the unified focal loss hyperparameters at specified epochs
during training.

## Usage

``` r
callback_reconfigure_loss(loss, schedule)
```

## Arguments

- loss:

  An instantiated `defineUnifiedFocalLoss` or `defineUnifiedFocalLossDS`
  object. Must be the same object passed as `loss` to
  [`luz::setup()`](https://mlverse.github.io/luz/reference/setup.html).

- schedule:

  A list of named lists, each specifying one reconfiguration event. Each
  entry must contain:

  `epoch`

  :   Single integer. The training epoch at the end of which the
      parameters are updated. The new values take effect from the
      following epoch onward.

  `params`

  :   Named list of parameter name–value pairs to update. Names must
      match those listed in the Details section above.

## Value

A luz callback object.

## Details

A luz callback that updates one or more parameters of an instantiated
`defineUnifiedFocalLoss` or `defineUnifiedFocalLossDS` loss object at
defined epoch boundaries. Because torch `nn_module` objects are
reference types, the same object passed here is used by the luz training
loop, so changes applied by the callback take effect immediately for the
following epoch.

For `defineUnifiedFocalLossDS`, parameters are routed to the inner
`defineUnifiedFocalLoss` sub-object automatically.

Parameters that have pre-computed tensor counterparts (see Details) have
both the scalar field and the corresponding tensor updated atomically so
that the forward pass always reads consistent values.

The following parameters may be reconfigured:

- `lambda`:

  Relative weight of distribution vs. region loss.

- `gamma`:

  Focal weighting strength.

- `delta`:

  False positive / false negative balance (scalar or per-class vector).

- `clsWghtsDist`:

  Class weights for the distribution loss (scalar or per-class vector).

- `clsWghtsReg`:

  Class weights for the region loss (scalar or per-class vector).

- `smooth`:

  Smoothing constant for numeric stability.

- `distThreshold`:

  Distance threshold for edge weighting.

- `minWght`:

  Minimum per-pixel edge weight.

- `maxWght`:

  Maximum per-pixel edge weight.

- `useLogCosH`:

  Toggle log-cosh transform on the region loss.

- `labelSmoothing`:

  Label smoothing factor for the distribution loss.

- `classAvgDist`:

  Switch between macro-averaged and pixel-level distribution loss.

- `prioritizeEdge`:

  Flip which pixels receive the higher edge weight.

- `edgeMode`:

  Edge weighting mode: `"threshold"`, `"linear"`, or `"gaussian"`.

- `applyDistWght`:

  Which loss component receives edge weights: `"distribution"`,
  `"region"`, or `"both"`.

- `cropFactorMsk`:

  Mask crop factor.

- `cropFactorPred`:

  Prediction crop factor.

- `zeroStart`:

  Whether class indices start at 0.

## Examples

``` r
if (FALSE) { # \dontrun{
loss_fn <- defineUnifiedFocalLoss(nCls = 3, lambda = 0.5, gamma = 0.5, delta = 0.6)

reconfigure_cb <- callback_reconfigure_loss(
  loss = loss_fn,
  schedule = list(
    list(epoch = 5,  params = list(lambda = 0.7, gamma = 0.3)),
    list(epoch = 10, params = list(lambda = 0.9, clsWghtsDist = c(1, 2, 3))),
    list(epoch = 15, params = list(labelSmoothing = 0.1))
  )
)

fitted <- model |>
  luz::setup(loss = loss_fn, optimizer = torch::optim_adamw) |>
  luz::fit(train_dl, epochs = 20,
            callbacks = list(reconfigure_cb))
} # }
```
