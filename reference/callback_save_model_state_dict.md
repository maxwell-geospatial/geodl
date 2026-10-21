# callback_save_model_state_dict

Save the model state dictionary to disk at the end of each epoch as a
.pt file. For use within luz training loop.

## Usage

``` r
callback_save_model_state_dict(save_dir = ".", prefix = "model_epoch")
```

## Arguments

- save_dir:

  path and directory to save state dictionary files

- prefix:

  prefix for file name. Default is "model_epoch". The suffix is the
  epoch number.

## Value

No R object is returned. The state dictionary is saved to disk as a .pt
file.
