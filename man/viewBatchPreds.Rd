% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/viewBatchPreds.R
\name{viewBatchPreds}
\alias{viewBatchPreds}
\title{viewBatchPreds}
\usage{
viewBatchPreds(
  dataLoader,
  model,
  mode = "multiclass",
  nCols = 4,
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
}
\arguments{
\item{dataLoader}{Instantiated instance of a DataLoader created using torch::dataloader().}

\item{model}{Fitted model used to predict mini-batch.}

\item{mode}{"multiclass" or "binary". If the prediction returns the positive case logit
for a binary classification problem, use "binary". If 2 or more class logits are returned,
use "multiclass". This package treats all cases as multiclass.}

\item{nCols}{Number of columns in the image grid. Default is 3.}

\item{r}{Index of channel to assign to red channel. Default is 1 or the first channel.
For gray scale or single-band images, assign the same index to all three bands.}

\item{g}{Index of channel to assign to green channel. Default is 2 or the second channel.
For gray scale or single-band images, assign the same index to all three bands.}

\item{b}{Index of channel to assign to blue channel. Default is 3 or the third channel.
For gray scale or single-band images, assign the same index to all three bands.}

\item{cCodes}{Integer codes assigned to each class. Should be in the same order as cNames.}

\item{cNames}{Vector of class names. Must be the same length as number of classes.}

\item{cColors}{Vector of color values to use to display the masks. Colors are applied based on the
order of class indices. Length of vector must be the same as the number of classes.}

\item{useCUDA}{TRUE or FALSE. Default is FALSE. If TRUE, GPU will be used to predict
the data mini-batch. If FALSE, predictions will be made on the CPU. We recommend using a GPU.}

\item{probs}{TRUE or FALSE. Default is FALSE. If TRUE, rescaled logits will be
shown as opposed to the hard classification. If FALSE, hard classification will be
shown. For a binary problem where only the positive case logit is returned, the logit
is transformed using a sigmoid function. When 2 or more classes are predicted, softmax
is used to rescale the logits.}

\item{usedDS}{TRUE or FALSE. Must be set to TRUE when using deep supervision. Default is FALSE,
or it is assumed that deep supervision is not used.}
}
\value{
Image grids of example chips, reference masks, and predictions loaded from a mini-batch provided by the DataLoader.
}
\description{
Generate image grid of mini-batch of image chips, masks, and predictions for all samples in a DataLoader mini-batch.
}
\details{
The goal of this function is to provide a visual check of predictions for a mini-batch of data.
}
\examples{
\dontrun{
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
}
}
