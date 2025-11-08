---
output: github_document
---

<!-- README.md is generated from README.Rmd. Please edit that file -->

# geodl <img src="geodlHex.png" align="right" width="132" />

<!-- badges: start -->
<!-- badges: end -->

# Updates with July 2025 GitHub Release 

Note: This release has yet to be submitted to CRAN. 

1. Support for three model architectures: UNet, UNet with MobileNetv2 encoder, and UNet3+
2. UNet with MobileNetv2 encoder is no longer limited to three input predictor variables
3. Assessment and prediction functions now expect a nn_module object as opposed to a luz fitted object
4. Dynamically generate chips during training process as opposed to saving them to disk beforehand (still experimental)
5. Ignore outer rows and columns of cells when calculating loss or assessment metrics if desired
5. Use R torch to calculate several different land surface parameters (LSPs) from a digital terrain model: slope, hillshade, aspect, northwardness, eastwardness, transformed solar radiation aspect index (TRASP), site exposure index (SEI), topographic position index (TPI), and surface curvatures (mean, profile, and planform)
7. Calculate three-band terrain visualization raster grid from a DTM using torch or terra
8. New specialized model for extracting geomorphic features from digital terrain models (DTMs)
9. New function to count the number of trainable parameters in a model
10. Fixed issue with chip generation pipeline that caused some chips with NA cells to be written
11. Updated atrous spatial pyramid pooling (ASPP) module to align with the version used within DeepLabv3+


This package provides tools for semantic segmentation of geospatial data using convolutional neural network-based deep learning. Utility functions allow for creating masks, image chips, data frames listing image chips in a directory, and DataSets for use within DataLoaders. Additional functions are provided to serve as checks during the data preparation and training process. Training can also be conducted by dynamically generated chips (still experimental). The package relies on torch for implementing deep learning, which does not require the installation of a Python environment. Raster geospatial data are handled with terra. Models can be trained using a CUDA-enabled GPU; however, multi-GPU training is not supported by torch in R. Both binary and multiclass models can be trained. 

Full details about the package are documented in a [PLOS One article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0315127):

Maxwell, A.E., Farhadpour, S., Das, S. and Yang, Y., 2024. geodl: An R package for geospatial deep learning semantic segmentation using torch and terra. *PLoS One*, 19(12), p.e0315127.

A UNet architecture can be defined with 4 blocks in the encoder, a bottleneck block, and 4 blocks in the decoder. The UNet can accept a variable number of input channels, and the user can define the number of feature maps produced in each encoder and decoder block and the bottleneck. Users can also choose to (1) replace all ReLU activation functions with leaky ReLU or swish, (2) implement attention gates along the skip connections, (3) implement squeeze and excitation modules within the encoder blocks, (4) add residual connections within all blocks, (5) replace the bottleneck with a modified atrous spatial pyramid pooling (ASPP) module, and/or (6) implement deep supervision using predictions generated at each stage in the decoder. 

A second UNet architecture is implemented with a MobileNet-v2 backbone. This model can be initialized using ImageNet weights for the encoder. The encoder can be frozen or trained during the training loop. If the number of input predictor variables or channels is not three, ImageNet weights are averaged for all input channels in the first layer. If three channels or predictor variables are provided, then the user can choose to use the ImageNet weights or average the weights in the first layer. 

Two additional models are provided: UNet3+ and a modified version of HRNet. 

A unified focal loss framework is implemented after:

Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal loss: Generalising dice and cross entropy-based losses to handle class imbalanced medical image segmentation. *Computerized Medical Imaging and Graphics*, 95, p.102026.

We have also implemented assessment metrics using the luz package including overall accuracy, F1-score, recall, and precision. 

Trained models can be used to predict to spatial data without the need to generate chips from larger spatial extents. Functions are available for performing accuracy assessment. 

Utility functions are provided to generate a variety of land surface parameters (LSPs) from a digital terrain model (DTM).

This package is still experimental and is a work-in-progress. We are interested in finding additional contributors/collaborators. 

## Installation

You can install the development version of geodl from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("maxwell-geospatial/geodl")
```
## Example

[Chapter 15](https://wvview.org/gslr/C15geodl_P1.html) and [Chapter 16](https://wvview.org/gslr/c16geodl_P2.html) in the free and openly available online text [*Geospatial Supervised Learning using R*](https://wvview.org/gslr/index.html) serve as the documentation for this package. 
