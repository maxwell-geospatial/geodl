# callback_ds_weight_decay

Decay ancillary deep supervision loss weights during luz training.

## Usage

``` r
callback_ds_weight_decay(weight_env, decay = "linear", primary_only_frac = 0.1)
```

## Arguments

- weight_env:

  A weight environment created by `make_ds_weights` and shared with
  `ds_loss`.

- decay:

  Character string, either `"linear"` (default) or `"cosine"`.

- primary_only_frac:

  Numeric value in `(0, 1)` specifying the fraction of total epochs at
  the end of training during which only the primary loss is used
  (ancillary weights are zero). Defaults to `0.1`.

## Value

A luz callback object.

## Details

A luz callback that reduces the weights of ancillary outputs (all
elements of `weight_env$values` except the first) toward zero over the
decaying portion of training, leaving the primary output weight
unchanged. The fraction of epochs reserved for primary-loss-only
training is controlled by `primary_only_frac` (default `0.1`, i.e. the
final 10\\ epochs).

Two decay schedules are supported:

- `"linear"`:

  Ancillary weights decrease linearly from their initial values to zero.
  Matches the nnU-Net strategy.

- `"cosine"`:

  Ancillary weights follow a cosine curve, decaying slowly at first and
  more rapidly through mid-training.
