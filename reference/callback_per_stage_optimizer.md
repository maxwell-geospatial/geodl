# callback_per_stage_optimizer

Replace the luz optimizer with per-stage parameter groups at the start
of training.

## Usage

``` r
callback_per_stage_optimizer(optimizer_fn = torch::optim_adamw, ...)
```

## Arguments

- optimizer_fn:

  Optimizer constructor function. Defaults to
  [`torch::optim_adamw`](https://torch.mlverse.org/docs/reference/optim_adamw.html).
  Any torch optimizer that accepts a list of parameter groups as its
  first argument can be used.

- ...:

  Additional arguments forwarded to `optimizer_fn`, such as
  `weight_decay`.

## Value

A luz callback object.

## Details

A luz callback that calls `model$get_param_groups()` on the instantiated
model at the start of `fit()` and replaces the default flat-parameter
optimizer with one built from the per-stage parameter groups. This
allows different learning rates to be assigned to different stages of
the network when using models that support `stageLRs` (e.g.
`defineUNet`, `defineMobileUNet`, `defineUNet3p`,
`defineEfficientUNetB2`).

This callback must be placed **before** any learning rate scheduler
callback in the `callbacks` list so that the scheduler is initialised
against the per-stage optimizer rather than the original flat one.

When using
[`torch::lr_one_cycle`](https://torch.mlverse.org/docs/reference/lr_one_cycle.html)
alongside this callback, pass `max_lr`, `max_momentum`, and
`base_momentum` as vectors of length equal to the number of stages so
that the scheduler can assign per-group values. A proportionally scaled
`max_lr` vector can be constructed as
`stage_lrs / max(stage_lrs) * global_max_lr`.
