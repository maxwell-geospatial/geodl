# callback_ema

Maintain an exponential moving average (EMA) of model weights during
training.

## Usage

``` r
callback_ema(
  decay = 0.999,
  warmup_steps = 0L,
  save_dir = NULL,
  prefix = "ema_epoch"
)
```

## Arguments

- decay:

  Numeric scalar in `(0, 1)` controlling the smoothing strength. Higher
  values (e.g. `0.999`) produce a slower-moving average; lower values
  (e.g. `0.99`) track the training weights more closely. Default
  `0.999`.

- warmup_steps:

  Non-negative integer giving the number of optimizer steps (batches) at
  the start of training during which the shadow weights track the
  training weights exactly before EMA decay begins. Default `0L`.

- save_dir:

  Optional character string path to a directory. When supplied, the EMA
  state dictionary is saved here at the end of every epoch as a `.pt`
  file. Default `NULL` (no saving).

- prefix:

  File-name prefix used when `save_dir` is set. Files are named
  `<prefix>_NNN.pt` where `NNN` is the zero-padded epoch number. Default
  `"ema_epoch"`.

## Value

A luz callback object.

## Details

A luz callback that keeps a shadow copy of all model parameters and
updates them as an exponential moving average of the training weights
after each optimizer step. The EMA weights are swapped into the model
during validation so that validation metrics reflect EMA performance,
then the training weights are restored so training continues normally.

When `save_dir` is supplied, a complete model state dictionary with the
EMA parameter values merged in is written to disk at the end of each
epoch. The saved file is fully compatible with
`model$load_state_dict(torch::torch_load(path))` and can be used
directly for inference without any additional steps. Non-parameter
buffers such as batch normalisation running statistics are taken from
the live model and are not averaged, which is the standard EMA practice.

The update rule applied after each training batch is: \$\$\hat{\theta}
\leftarrow \mathrm{decay} \times \hat{\theta} + (1 - \mathrm{decay})
\times \theta\$\$ where \\\hat{\theta}\\ are the shadow (EMA) weights
and \\\theta\\ are the current training weights.

During `warmup_steps` the shadow weights are set equal to the current
training weights at every batch rather than applying the EMA rule. This
prevents early random-initialisation values from being mixed into the
average.
