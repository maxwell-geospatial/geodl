# countParams

Count the number of trainable parameters in a model.

## Usage

``` r
countParams(model, test_data)
```

## Arguments

- model:

  instantiated model object as nn_module subclass as opposed to luz
  fitted object.

- test_data:

  Example tensor of the correct shape for the model being explored.

## Value

Counts of model parameters.

## Details

Count the number of trainable parameters in an instantiated model
subclassed from nn_module().
