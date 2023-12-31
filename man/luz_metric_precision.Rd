% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/luzMetrics.R
\name{luz_metric_precision}
\alias{luz_metric_precision}
\title{luz_metric_precision}
\usage{
luz_metric_precision(
  nCls = 1,
  smooth = 1,
  mode = "multiclass",
  average = "macro",
  zeroStart = TRUE,
  chnDim = TRUE,
  usedDS = FALSE
)
}
\arguments{
\item{smooth}{a smoothing factor to avoid divide by zero errors. Default is 1.}

\item{mode}{Either "binary" or "multiclass". If "binary", only the logit for
positive class prediction should be provided. If both the positive and negative
or background class probability is provided for a binary classification, use
the "multiclass" mode.}

\item{average}{Either "micro" or "macro". Whether to use micro- or macro-averaging
for multiclass metric calculation. Ignored when mode is "binary". Default is
"macro". Macro averaging consists of calculating the metric separately for each class
and averaging the results such that all classes are equally weighted. Micro-averaging calculates the
metric for all classes collectively, and classes with a larger number of samples will have a larger
weight in the final metric.}

\item{zeroStart}{TRUE or FALSE. If class indices start at 0 as opposed to 1, this should be set to
TRUE. This is required  to implement one-hot encoding since R starts indexing at 1. Default is TRUE.}

\item{chnDim}{TRUE or FALSE. Whether the channel dimension is included in the target tensor:
\link{Batch, Channel, Height, Width} as opposed to \link{Batch, Channel, Height, Width}. If the channel dimension
is included, this should be set to TRUE. If it is not, this should be set to FALSE. Default is TRUE.}

\item{usedDS}{TRUE or FALSE. If deep supervision was implemented and masks are produced at varying scales using
the defineSegDataSetDS() function, this should be set to TRUE. Only the original resolution is used
to calculate assessment metrics. Default is FALSE.}

\item{preds}{Tensor of class predicted probabilities with shape
\link{Batch, Class Logits, Height, Width} for a multiclass classification. For a
binary classification, you can provide logits for the positive class as
\link{Batch, Positive Class Logit, Height, Width} or \link{Batch, Height, Width}.}

\item{target}{Tensor of target class labels with shape
\link{Batch, Class Indices, Height, Width} for a multiclass classification. For a
binary classification, you can provide targets as
\link{Batch, Positive Class Index, Height, Width} or \link{Batch, Height, Width}. For
binary classification, the class index must be 1 for the positive class and
0 for the background case.}

\item{nCLs}{number of classes being differentiated. Should be 1 for a binary classification
where only the postive case logit is returned. Default is 1.}
}
\value{
Calculated metric return as a base-R vector as opposed to tensor.
}
\description{
luz_metric function to calculate precision
}
\details{
Calculates precision based on luz_metric() for use within training and validation
loops.
}
