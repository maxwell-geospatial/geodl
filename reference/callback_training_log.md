# callback_training_log

Log per-epoch learning rates and deep supervision weights to a CSV file.

## Usage

``` r
callback_training_log(path, weight_env = NULL)
```

## Arguments

- path:

  File path for the output CSV. The parent directory is created if it
  does not exist. Any existing file at this path is overwritten.

- weight_env:

  Optional. A weight environment created by `make_ds_weights`. Supply
  when using deep supervision so that the current weights are logged
  each epoch. Omit or pass `NULL` when not using deep supervision.

## Value

A luz callback object.

## Details

A luz callback that writes one row per training epoch recording the
current optimizer learning rate(s) and, optionally, the deep supervision
loss weights. Column structure is determined automatically on the first
epoch:

- If the model uses per-stage learning rates via `get_param_groups`,
  each stage is logged in its own column (`lr_1`, `lr_2`, ...).

- If a single learning rate is used, a single `lr` column is written.

- If `weight_env` is supplied, columns `ds_primary`, `ds_aux1`,
  `ds_aux2`, ... are appended for each weight.
