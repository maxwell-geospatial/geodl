# make_ds_weights

Create a shared mutable weight container for deep supervision loss
weighting.

## Usage

``` r
make_ds_weights(weights)
```

## Arguments

- weights:

  Numeric vector of initial loss weights. The first element is the
  primary (full-resolution) output; subsequent elements are ancillary
  outputs in coarsest-to-finest order. All values must be non-negative.

## Value

An R environment with a single element `$values` containing the current
weight vector. Passed by reference, so updates made by
`callback_ds_weight_decay` are immediately visible to `ds_loss`.

## Details

Creates an R environment that holds the current per-output loss weights
for deep supervision training. Pass the returned object to both
`defineUnifiedFocalLossDS` (via the `weight_env` argument) and
`callback_ds_weight_decay` so the callback can update the weights the
loss function reads each epoch.
