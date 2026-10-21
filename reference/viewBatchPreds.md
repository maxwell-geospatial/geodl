# viewBatchPreds

Generate image grid of mini-batch of image chips, masks, and predictions
for all samples in a DataLoader mini-batch.

## Usage

``` r
viewBatchPreds(
  dataLoader,
  model,
  mode = "multiclass",
  nCols = 4,
  padding = 10,
  r = 1,
  g = 2,
  b = 3,
  cCodes,
  cNames,
  cColors,
  useCUDA = TRUE,
  probs = FALSE,
  usedDS = FALSE
)
```

## Arguments

- dataLoader:

  Instantiated instance of a DataLoader created using
  torch::dataloader().

- model:

  Fitted model used to predict mini-batch.

- mode:

  "multiclass" or "binary". If the prediction returns the positive case
  logit for a binary classification problem, use "binary". If 2 or more
  class logits are returned, use "multiclass". This package treats all
  cases as multiclass.

- nCols:

  Number of columns in the image grid. Default is 3.

- padding:

  Number of cells of padding to add around each example in resulting
  image.

- r:

  Index of channel to assign to red channel. Default is 1 or the first
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

  Integer codes assigned to each class. Should be in the same order as
  cNames.

- cNames:

  Vector of class names. Must be the same length as number of classes.

- cColors:

  Vector of color values to use to display the masks. Colors are applied
  based on the order of class indices. Length of vector must be the same
  as the number of classes.

- useCUDA:

  TRUE or FALSE. Default is FALSE. If TRUE, GPU will be used to predict
  the data mini-batch. If FALSE, predictions will be made on the CPU. We
  recommend using a GPU.

- probs:

  TRUE or FALSE. Default is FALSE. If TRUE, rescaled logits will be
  shown as opposed to the hard classification. If FALSE, hard
  classification will be shown. For a binary problem where only the
  positive case logit is returned, the logit is transformed using a
  sigmoid function. When 2 or more classes are predicted, softmax is
  used to rescale the logits.

- usedDS:

  TRUE or FALSE. Must be set to TRUE when using deep supervision.
  Default is FALSE, or it is assumed that deep supervision is not used.

## Value

Image grids of example chips, reference masks, and predictions loaded
from a mini-batch provided by the DataLoader.

## Details

The goal of this function is to provide a visual check of predictions
for a mini-batch of data.

## Examples

``` r
if (FALSE) { # \dontrun{
viewBatchPreds(dataLoader=testDL,
               model=model,
               mode="multiclass",
               nCols =5,
               r = 1,
               g = 2,
               b = 3,
               cCodes=c(1,2),
               cNames=c("Not Mine", "Mine"),
               cColors=c("gray", "darksalmon"),
               useCUDA=TRUE,
               probs=FALSE,
               usedDS=FALSE)
} # }
```
