# plotTrainingSchedule

Visualise per-stage base learning rates, an optional LR scheduler trace,
and deep supervision weight decay without running a training loop.

## Usage

``` r
plotTrainingSchedule(
  stageLRs,
  epochs,
  initialWeights = NULL,
  decay = "linear",
  primary_only_frac = 0.1,
  stageNames = NULL,
  scheduler = NULL
)
```

## Arguments

- stageLRs:

  Numeric vector of per-stage base learning rates matching the
  `stageLRs` argument passed to the model. A single value produces a
  one-bar chart.

- epochs:

  Total number of training epochs.

- initialWeights:

  Optional numeric vector of initial deep supervision loss weights as
  passed to `make_ds_weights`. When `NULL` the DS weight panel is
  omitted.

- decay:

  Character string, either `"linear"` (default) or `"cosine"`, matching
  the `decay` argument passed to `callback_ds_weight_decay`.

- primary_only_frac:

  Numeric value in `(0, 1)` specifying the fraction of total epochs
  reserved for primary-loss-only training at the end of the run,
  matching the `primary_only_frac` argument passed to
  `callback_ds_weight_decay`. Defaults to `0.1`.

- stageNames:

  Optional character vector of stage labels for the learning rate chart.
  Defaults to `"Stage 1"`, `"Stage 2"`, ...

- scheduler:

  Optional named list specifying an LR scheduler to visualise. Must
  contain at least `type` (`"cosine_annealing"`, `"one_cycle"`, or
  `"step"`). See Details for per-type parameters.

## Value

Invisibly returns a named list with element `lr` (a data frame of stage
names and learning rates), `lr_trace` (a data frame of epoch-by-LR
values when `scheduler` is supplied), and `weights` (a data frame of
epoch-by-weight values when `initialWeights` is supplied).

## Details

Produces up to three panels depending on inputs. The left panel always
shows a bar chart of per-stage base learning rates. When `scheduler` is
supplied a centre panel shows the effective learning rate curve over all
epochs. When `initialWeights` is supplied a right panel shows how each
deep supervision weight evolves over epochs, replicating the exact decay
applied by `callback_ds_weight_decay`.

The `scheduler` list accepts three types, each mirroring a torch R
scheduler:

- `"cosine_annealing"`:

  Mirrors `lr_cosine_annealing`. Named elements: `base_lr` (starting LR;
  defaults to `max(stageLRs)`), `T_max` (epochs per half-cycle; defaults
  to `epochs`), `eta_min` (minimum LR, default `0`). The LR oscillates
  between `base_lr` and `eta_min` with period `2 * T_max`.

- `"one_cycle"`:

  Mirrors `lr_one_cycle`. Named elements: `max_lr` (peak LR; defaults to
  `max(stageLRs)`), `pct_start` (warmup fraction, default `0.3`),
  `div_factor` (initial_lr = max_lr / div_factor, default `25`),
  `final_div_factor` (min_lr = initial_lr / final_div_factor, default
  `1e4`), `anneal_strategy` (`"cos"` or `"linear"`, default `"cos"`).

- `"step"`:

  Mirrors `lr_step`. Named elements: `base_lr` (starting LR; defaults to
  `max(stageLRs)`), `step_size` (epochs between LR drops; defaults to
  `max(1, floor(epochs / 10))`), `gamma` (multiplicative decay factor
  per drop, default `0.1`).
