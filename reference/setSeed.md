# setSeed

Set a global random seed for reproducible geodl workflows.

## Usage

``` r
setSeed(seed)
```

## Arguments

- seed:

  Single integer random seed.

## Value

Invisibly returns `seed` (as an integer).

## Details

A geodl/torch training pipeline draws from two independent random number
generator (RNG) streams: R's base RNG and torch's own RNG. `setSeed()`
seeds both from a single value so that dataset shuffling, data
augmentation, model weight initialization, and the training loop all
produce identical results across runs.

The following sources of randomness are covered:

- Dataset shuffling:

  `torch::dataloader(..., shuffle = TRUE)` and any
  [`dplyr::sample_n()`](https://dplyr.tidyverse.org/reference/sample_n.html)-based
  shuffling (e.g. `makeChipsDF(shuffle = TRUE)`) read from R's base RNG.

- Data augmentation:

  [`defineSegDataSet()`](https://maxwell-geospatial.github.io/geodl/reference/defineSegDataSet.md)
  and
  [`defineDynamicSegDataSet()`](https://maxwell-geospatial.github.io/geodl/reference/defineDynamicSegDataSet.md)
  choose which augmentations to apply and their magnitudes with base R
  functions ([`runif()`](https://rdrr.io/r/stats/Uniform.html),
  [`sample()`](https://rdrr.io/r/base/sample.html),
  [`dplyr::sample_n()`](https://dplyr.tidyverse.org/reference/sample_n.html)),
  which read from R's base RNG.

- Model weight initialization:

  All `defineXXX` model constructors (e.g. `defineUNet`, `defineDLab`)
  rely on torch's default layer initializers, which draw from torch's
  RNG.

- Training loop:

  Any stochastic operations inside
  [`luz::fit()`](https://generics.r-lib.org/reference/fit.html) (e.g.
  dropout, if present in a given architecture) draw from torch's RNG.

**Call this before** creating the dataset, dataloader, and model, and
before calling
[`luz::fit()`](https://generics.r-lib.org/reference/fit.html) — any
random draws that occur before seeding will not be reproducible.

When
[`torch::dataloader()`](https://torch.mlverse.org/docs/reference/dataloader.html)
is used with `num_workers > 0`, torch automatically reseeds each worker
subprocess (with both [`set.seed()`](https://rdrr.io/r/base/Random.html)
and
[`torch::torch_manual_seed()`](https://torch.mlverse.org/docs/reference/torch_manual_seed.html))
from the parent process's RNG state at the time the workers are spawned.
As long as `setSeed()` is called once before the dataloader is created,
and `num_workers` is held constant between runs, worker seeding is
deterministic and requires no further action.

Full determinism on GPU is still subject to caveats outside this
function's control: some cuDNN convolution algorithms are inherently
non-deterministic regardless of RNG seeding, and torch's R bindings do
not currently expose a "deterministic algorithms" toggle. For strict
bit-for-bit reproducibility, training on CPU remains the most reliable
option.
