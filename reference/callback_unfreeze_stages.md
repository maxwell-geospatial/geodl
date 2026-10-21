# callback_unfreeze_stages

Progressively unfreeze previously frozen model stage parameters during
training.

## Usage

``` r
callback_unfreeze_stages(schedule)
```

## Arguments

- schedule:

  A list of named lists, each specifying one unfreeze action. Each entry
  must contain:

  `epoch`

  :   Single integer. The training epoch after which the listed
      components are unfrozen.

  `components`

  :   Character vector of submodule names to unfreeze at this epoch.

## Value

A luz callback object.

## Details

A luz callback that sets `requires_grad = TRUE` on the parameters of
specified model components at defined epoch boundaries, reversing an
earlier freeze applied by `callback_freeze_stages` or by pre-training
setup (e.g. `freezeEncoder = TRUE`). Multiple unfreeze actions at
different epochs are supported via the `schedule` argument.

Component names must match the submodule field names used inside the
model. For `defineUNet` and `defineUNet3p` the stage names are `"e1"`,
`"e2"`, `"e3"`, `"e4"`, `"btn"`, `"d1"`, `"d2"`, `"d3"`, `"d4"`. For
`defineMobileUNet` and `defineEfficientUNetB2` there are five encoder
stages (`"e1"` through `"e5"`) and five decoder stages (`"d1"` through
`"d5"`).

Unfreezing takes effect at the **end** of the specified epoch, so
unfrozen components begin receiving gradient updates from the following
epoch onward. A message is printed each time a component is unfrozen,
reporting the component name and parameter count. The optimizer state
for the newly unfrozen parameters is not reset, so any accumulated
momentum carries over.

`callback_unfreeze_stages` and `callback_freeze_stages` can be used
together to implement warm-up schedules where the encoder is initially
frozen while the decoder trains, then progressively unfrozen.

## Examples

``` r
if (FALSE) { # \dontrun{
# Train with encoder frozen for 5 epochs, then unfreeze progressively
callback_unfreeze_stages(
  schedule = list(
    list(epoch = 5,  components = c("e4", "e5")),
    list(epoch = 10, components = c("e2", "e3")),
    list(epoch = 15, components = c("e1"))
  )
)
} # }
```
