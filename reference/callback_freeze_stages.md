# callback_freeze_stages

Progressively freeze model stage parameters during training.

## Usage

``` r
callback_freeze_stages(schedule)
```

## Arguments

- schedule:

  A list of named lists, each specifying one freeze action. Each entry
  must contain:

  `epoch`

  :   Single integer. The training epoch after which the listed
      components are frozen.

  `components`

  :   Character vector of submodule names to freeze at this epoch.

## Value

A luz callback object.

## Details

A luz callback that sets `requires_grad = FALSE` on the parameters of
specified model components at defined epoch boundaries. Multiple freeze
actions at different epochs are supported via the `schedule` argument.
Once frozen, a component's parameters receive no gradient and are not
updated by the optimizer for the remainder of training.

Component names must match the submodule field names used inside the
model. For `defineUNet` and `defineUNet3p` the stage names are `"e1"`,
`"e2"`, `"e3"`, `"e4"`, `"btn"`, `"d1"`, `"d2"`, `"d3"`, `"d4"`. For
`defineMobileUNet` and `defineEfficientUNetB2` there are five encoder
stages (`"e1"` through `"e5"`) and five decoder stages (`"d1"` through
`"d5"`). Attention gate and squeeze-and-excitation submodules (`"ag1"`,
`"se1"`, etc.) can also be named.

Freezing takes effect at the **end** of the specified epoch, so frozen
components are inactive from the following epoch onward. A message is
printed each time a component is frozen, reporting the component name
and parameter count.

## Examples

``` r
if (FALSE) { # \dontrun{
# Freeze early encoder stages after epoch 5,
# then the remaining encoder and bottleneck after epoch 10
callback_freeze_stages(
  schedule = list(
    list(epoch = 5,  components = c("e1", "e2")),
    list(epoch = 10, components = c("e3", "e4", "btn"))
  )
)
} # }
```
