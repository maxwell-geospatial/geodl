# viewBatch

Generate image grid of mini-batch of image chips and associated masks
created by a DataLoader.

## Usage

``` r
viewBatch(
  dataLoader,
  nCols = 3,
  r = 1,
  g = 2,
  b = 3,
  cCodes,
  cNames,
  cColors,
  padding = 10
)
```

## Arguments

- dataLoader:

  Instantiated instance of a DataLoader created using
  torch::dataloader().

- nCols:

  Number of columns in the image grid. Default is 3.

- r:

  Index of channel to assign to red channel. Default is 1 or first
  channel. For gray scale or single-band images, assign the same index
  to all three bands.

- g:

  Index of channel to assign to green channel. Default is 2 or the
  second channel. For gray scale or single-band images, assign the same
  index to all three bands.

- b:

  Index of channel to assign to blue channel. Default is 3 or the third
  channel. For gray scale or single-band images, assign the same index
  to all three bands.

- cCodes:

  class indices as a vector of integer values equal in length to the
  number of classes.

- cNames:

  Vector of class names. Must be the same length as number of classes.

- cColors:

  Vector of color values to use to display the masks. Colors are applied
  based on the order of class indices. Length of vector must be the same
  as the number of classes.

- padding:

  to place between chips in plot. Default is 10.

## Value

Image grids of example chips and masks loaded from a mini-batch produced
by the DataLoader.

## Details

The goal of this function is to provide a visual check of a mini-batch
of image chips and associated masks generated from a DataLoader.

## Examples

``` r
if (FALSE) { # \dontrun{
viewBatch(dataLoader=trainDL,
          nCols = 5,
          r = 1,
          g = 2,
          b = 3,
          cCodes=c(1,2),
          cNames=c("Background", "Mine"),
          cColors=c("gray", "darksalmon"))
} # }
```
