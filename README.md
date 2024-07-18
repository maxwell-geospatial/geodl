---
output: github_document
---

<!-- README.md is generated from README.Rmd. Please edit that file -->

# geodl <img src="vignettes/figure/geodlHex.png" align="right" width="132" />

<!-- badges: start -->
<!-- badges: end -->

This package provides tools for semantic segmentation of geospatial data using convolutional neural network-based deep learning. Utility functions allow for creating masks, image chips, data frames listing image chips in a directory, and DataSets for use within DataLoaders. Additional functions are provided to serve as checks during the data preparation and training process. 

Full details about the package are documented in a [preprint article](https://doi.org/10.31223/X53M6T) available on EarthArXiv.

A UNet architecture can be defined with 4 blocks in the encoder, a bottleneck block, and 4 blocks in the decoder. The UNet can accept a variable number of input channels, and the user can define the number of feature maps produced in each encoder and decoder block and the bottleneck. Users can also choose to (1) replace all ReLU activation functions with leaky ReLU or swish, (2) implement attention gates along the skip connections, (3) implement squeeze and excitation modules within the encoder blocks, (4) add residual connections within all blocks, (5) replace the bottleneck with a modified atrous spatial pyramid pooling (ASPP) module, and/or (6) implement deep supervision using predictions generated at each stage in the decoder. A second UNet architecture is implemented with a MobileNet-v2 backbone. 

A unified focal loss framework is implemented after:

Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal loss: Generalising dice and cross entropy-based losses to handle class imbalanced medical image segmentation. *Computerized Medical Imaging and Graphics*, 95, p.102026.

We have also implemented assessment metrics using the luz package including overall accuracy, F1-score, recall, and precision. 

Trained models can be used to predict to spatial data without the need to generate chips from larger spatial extents. Functions are available for performing accuracy assessment. 

The package relies on torch for implementing deep learning, which does not require the installation of a Python environment. Raster geospatial data are handled with terra. Models can be trained using a CUDA-enabled GPU; however, multi-GPU training is not supported by torch in R. Both binary and multiclass models can be trained. 

This package is still experimental and is a work-in-progress. 

## Installation

You can install the development version of geodl from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("maxwell-geospatial/geodl")
```

## Example

Please see the preprint article and articles/vignettes for details and example workflows and explanations. 
