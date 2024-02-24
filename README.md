
<!-- README.md is generated from README.Rmd. Please edit that file -->

# geodl <img src="geodlHex.png" align="right" width="132" />

<!-- badges: start -->
<!-- badges: end -->

This package provides tools for semantic segmentation of geospatial data using convolutional neural network-based deep learning. Utility functions allow for creating masks, image chips, data frames listing image chips in a directory, and DataSets for use within DataLoaders. Additional functions are provided to serve as checks during the data preparation and training process. 

A UNet architecture can be defined with 4 blocks in the encoder, a bottleneck block, and 4 blocks in the decoder. The UNet can accept a variable number of input channels, and the user can define the number of feature maps produced in each encoder and decoder block and the bottleneck. Users can also choose to (1) replace all ReLU activation functions with leaky ReLU or swish, (2) implement attention gates along the skip connections, (3) implement squeeze and excitation modules within the encoder blocks, (4) add residual connections within all blocks, (5) replace the bottleneck with a modified atrous spatial pyramid pooling (ASPP) module, and/or (6) implement deep supervision using predictions generated at each stage in the decoder. 

A unified focal loss framework is implemented after:

Yeung, M., Sala, E., Schönlieb, C.B. and Rundo, L., 2022. Unified focal loss: Generalising dice and cross entropy-based losses to handle class imbalanced medical image segmentation. *Computerized Medical Imaging and Graphics*, 95, p.102026.

We have also implemented assessment metrics using the luz package including F1-score, recall, and precision. 

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
## Data Preparation Examples

```r
library(geodl)
setwd("Your Working Directory")

# Make terrain derivatives ------------------------------------------------

inDTM <- terra::rast("data/elev/dtm.tif")
terrOut <- makeTerrainDerivatives(dtm=inDTM,
                                  res=2,
                                  filename="data/elev/tstack.tif")
terra::plotRGB(terrOut*255)

# Make masks from vector data ---------------------------------------------

makeMasks(image = "data/toChipBinary/image/KY_Saxton_709705_1970_24000_geo.tif",
          features = "data/toChipBinary/msks/KY_Saxton_709705_1970_24000_geo.shp",
          crop = TRUE,
          extent = "data/toChipBinary/extent/KY_Saxton_709705_1970_24000_geo.shp",
          field = "classvalue",
          background = 0,
          outImage = "data/toChipBinary/output/topoOut.tif",
          outMask = "data/toChipBinary/output/mskOut.tif",
          mode = "Both")

terra::plotRGB(terra::rast("data/toChipBinary/output/topoOut.tif"))
terra::plot(terra::rast("data/toChipBinary/output/mskOut.tif"))

# Make, describe, and view chips (binary) ---------------------------------

makeChips(image = "data/toChipBinary/output/topoOut.tif",
          mask = "data/toChipBinary/output/mskOut.tif",
          n_channels = 3,
          size = 256,
          stride_x = 256,
          stride_y = 256,
          outDir = "data/toChipBinary/chips/",
          mode = "Positive",
          useExistingDir=FALSE)

chpDF <- makeChipsDF(folder = "data/toChipBinary/chips/",
                     outCSV = "data/toChipBinary/chips/chipsDF.csv",
                     extension = ".tif",
                     mode="Positive",
                     shuffle=FALSE,
                     saveCSV=TRUE)
head(chpDF)

chpDescript <- describeChips(folder= "data/toChipBinary/chips/",
                             extension = ".tif",
                             mode = "Positive",
                             subSample = TRUE,
                             numChips = 100,
                             numChipsBack = 100,
                             subSamplePix = TRUE,
                             sampsPerChip = 400)

print(chpDescript)

viewChips(chpDF=chpDF,
          folder= "data/toChipBinary/chips/",
          nSamps = 16,
          mode = "both",
          justPositive = FALSE,
          cCnt = 4,
          rCnt = 4,
          r = 1,
          g = 2,
          b = 3,
          rescale = FALSE,
          rescaleVal = 1,
          cNames=c("Background", "Mine"),
          cColor=c("#D4C2AD","#BA8E7A"),
          useSeed = FALSE,
          seed = 42)


# Make, describe, and view chips (multiclass) -----------------------------

makeChipsMultiClass(image = "data/toChipMultiClass/multiclassLCAI.tif",
                    mask = "data/toChipMultiClass/multiclass_reference.tif",
                    n_channels = 3,
                    size = 256,
                    stride_x = 512,
                    stride_y = 512,
                    outDir = "data/toChipMultiClass/chips/",
                    useExistingDir=FALSE)

chpDF <- makeChipsDF(folder = "data/toChipMultiClass/chips/",
                     outCSV = "data/toChipMultiClass/chips/chipsDF.csv",
                     extension = ".tif",
                     mode="All",
                     shuffle=FALSE,
                     saveCSV=TRUE)
head(chpDF)

chpDescript <- describeChips(folder= "data/toChipMultiClass/chips/",
                             extension = ".tif",
                             mode = "All",
                             subSample = TRUE,
                             numChips = 50,
                             numChipsBack = 100,
                             subSamplePix = TRUE,
                             sampsPerChip = 400)

print(chpDescript)

viewChips(chpDF=chpDF,
          folder= "data/toChipMultiClass/chips/",
          nSamps = 16,
          mode = "both",
          justPositive = FALSE,
          cCnt = 4,
          rCnt = 4,
          r = 1,
          g = 2,
          b = 3,
          rescale = FALSE,
          rescaleVal = 1,
          cNames=c("Background",
                   "Building",
                   "Woodland",
                   "Water",
                   "Road"),
          cColor=c("gray",
                   "darksalmon",
                   "forestgreen",
                   "lightblue",
                   "black"),
          useSeed = FALSE,
          seed = 42)

```

## Accuracy Assessment Examples

```r
library(geodl)
setwd("Your Working Directory")

# Accuracy assessment examples --------------------------------------------

#Example 1: table that already has the reference and predicted labels for a multiclass classification
mcIn <- readr::read_csv("data/tables/multiClassExample.csv")
myMetrics <- assessPnts(reference=mcIn$ref,
                        predicted=mcIn$pred,
                        multiclass=TRUE)
print(myMetrics)

#Example 2: table that already has the reference and predicted labels for a binary classification
bIn <- readr::read_csv("data/tables/binaryExample.csv")
myMetrics <- assessPnts(reference=bIn$ref,
                        predicted=bIn$pred,
                        multiclass=FALSE,
                        positive_case = "Mine")
print(myMetrics)

#Example 3: Read in point layer and intersect with raster output
pntsIn <- terra::vect("data/topoResult/topoPnts.shp")
refG <- terra::rast("data/topoResult/topoRef.tif")
predG <- terra::rast("data/topoResult/topoPred.tif")
pntsIn2 <- terra::project(pntsIn, terra::crs(refG))
refIsect <- terra::extract(refG, pntsIn2)
predIsect <- terra::extract(predG, pntsIn2)
resultsIn <- data.frame(ref=as.factor(refIsect$topoRef),
                        pred=as.factor(predIsect$topoPred))
myMetrics <- assessPnts(reference=bIn$ref,
                        predicted=bIn$pred,
                        multiclass=FALSE,
                        mappings=c("Mine", "Not Mine"),
                        positive_case = "Mine")
print(myMetrics)

#Assess using raster grids
refG <- terra::rast("data/topoResult/topoRef.tif")
predG <- terra::rast("data/topoResult/topoPred.tif")
refG2 <- terra::crop(terra::project(refG, predG), predG)
myMetrics <- assessRaster(reference = refG2,
                          predicted = predG,
                          multiclass = FALSE,
                          mappings = c("Not Mine", "Mine"),
                          positive_case = "Mine")
print(myMetrics)

```

## DataSets and DataLoaders Examples

``` r
library(geodl)
setwd("Your Working Directory")

# Binary Example ----------------------------------------------------------

#Get chips dataframe
chpDF <- makeChipsDF(folder = "data/toChipBinary/chips/",
                     outCSV = "data/toChipBinary/chips/chipsDF.csv",
                     extension = ".tif",
                     mode="Positive",
                     shuffle=FALSE,
                     saveCSV=FALSE)
head(chpDF)

#Get chips description
chpDescript <- describeChips(folder= "data/toChipBinary/chips/",
                             extension = ".tif",
                             mode = "Positive",
                             subSample = TRUE,
                             numChips = 100,
                             numChipsBack = 100,
                             subSamplePix = TRUE,
                             sampsPerChip = 400)

#Create dataset with transforms
myDS <- defineSegDataSet(chpDF,
                         folder="data/toChipBinary/chips/",
                         normalize = TRUE,
                         rescaleFactor = 255,
                         mskRescale=1,
                         mskAdd=0,
                         bands = c(1,2,3),
                         bMns=c(214,206,163),
                         bSDs=c(33,50,51),
                         mskLong = TRUE,
                         chnDim = TRUE,
                         doAugs = TRUE,
                         maxAugs = 2,
                         probVFlip = .3,
                         probHFlip = .3,
                         probBrightness = .1,
                         probContrast = 0,
                         probGamma = 0,
                         probHue = 0,
                         probSaturation = ,1,
                         brightFactor = c(.8,1.2),
                         contrastFactor = c(.8,1.2),
                         gammaFactor = c(.8, 1.2, 1),
                         hueFactor = c(-.2, .2),
                         saturationFactor = c(.8, 1.2))

#Instantiate dataloader
myDL <- torch::dataloader(myDS,
                          batch_size=4,
                          shuffle=TRUE,
                          drop_last = TRUE)

#View a batch as a check
viewBatch(dataLoader=myDL,
          chnDim = TRUE,
          mskLong = TRUE,
          nRows = 2,
          r = 1,
          g = 2,
          b = 3,
          cNames=c("Background", "Mine"),
          cColor=c("#D4C2AD","#BA8E7A"),
          usedDS=FALSE)

#Get batch stats as a check
myBatchStats <- describeBatch(myDL, usedDS=FALSE)
print(myBatchStats)

# Multiclass Example ------------------------------------------------------

#Get chips dataframe
chpDF <- makeChipsDF(folder = "data/toChipMultiClass/chips/",
                     outCSV = "data/toChipMultiClass/chips/chipsDF.csv",
                     extension = ".tif",
                     mode="All",
                     shuffle=FALSE,
                     saveCSV=FALSE)
head(chpDF)

#Get chips description
chpDescript <- describeChips(folder= "data/toChipMultiClass/chips/",
                             extension = ".tif",
                             mode = "All",
                             subSample = TRUE,
                             numChips = 50,
                             numChipsBack = 100,
                             subSamplePix = TRUE,
                             sampsPerChip = 400)
print(chpDescript)

#Create dataset with transforms
myDS <- defineSegDataSet(chpDF,
                         folder="data/toChipMultiClass/chips/",
                         normalize = TRUE,
                         rescaleFactor = 255,
                         mskRescale=1,
                         mskAdd=0,
                         bands = c(1,2,3),
                         bMns=c(90,92,90),
                         bSDs=c(50,41,31),
                         mskLong = TRUE,
                         chnDim = TRUE,
                         doAugs = TRUE,
                         maxAugs = 2,
                         probVFlip = .3,
                         probHFlip = .3,
                         probBrightness = .1,
                         probContrast = 0,
                         probGamma = 0,
                         probHue = 0,
                         probSaturation = ,1,
                         brightFactor = c(.8,1.2),
                         contrastFactor = c(.8,1.2),
                         gammaFactor = c(.8, 1.2, 1),
                         hueFactor = c(-.2, .2),
                         saturationFactor = c(.8, 1.2))

#Instantiate dataloader
myDL <- torch::dataloader(myDS,
                          batch_size=6,
                          shuffle=TRUE,
                          drop_last = TRUE)

#View a batch as a check
viewBatch(dataLoader=myDL,
          chnDim = TRUE,
          mskLong = TRUE,
          nRows = 2,
          r = 1,
          g = 2,
          b = 3,
          cNames=c("Background",
                   "Building",
                   "Woodland",
                   "Water",
                   "Road"),
          cColor=c("gray",
                   "darksalmon",
                   "forestgreen",
                   "lightblue",
                   "black"),
          usedDS=FALSE)

#Get batch stats as a check
myBatchStats <- describeBatch(myDL, usedDS=FALSE)
print(myBatchStats)
```

## luz Metrics Examples


``` r
library(geodl)
setwd("Your Working Directory")

# Binary examples ---------------------------------------------------------

refC <- terra::rast("data/metricCheck/binary_reference.tif")
predL <- terra::rast("data/metricCheck/binary_logits.tif")
predC <- terra::rast("data/metricCheck/binary_prediction.tif")
names(refC) <- "reference"
names(predC) <- "prediction"
cm <- terra::crosstab(c(predC, refC))
yardstick::f_meas(cm, estimator="binary", event_level="second")
yardstick::accuracy(cm, estimator="binary", event_level="second")
yardstick::recall(cm, estimator="binary", event_level="second")
yardstick::precision(cm, estimator="binary", event_level="second")

predL <- terra::as.array(predL)
refC <- terra::as.array(refC)

target <- torch::torch_tensor(refC, dtype=torch::torch_long())
pred <- torch::torch_tensor(predL, dtype=torch::torch_float32())
target <- target$permute(c(3,1,2))
pred <- pred$permute(c(3,1,2))

target <- target$unsqueeze(1)
pred <- pred$unsqueeze(1)

target <- torch::torch_cat(list(target, target), dim=1)
pred <- torch::torch_cat(list(pred, pred), dim=1)

metric<-luz_metric_f1score(nCls=1,
                           smooth=1e-8,
                           mode = "binary",
                           average="macro",
                           zeroStart=TRUE,
                           chnDim=TRUE,
                           usedDS=FALSE)
metric<-metric$new()
metric$update(pred,target)
metric$compute()

metric<-luz_metric_recall(nCls=1,
                          smooth=1e-8,
                          mode = "binary",
                          average="macro",
                          zeroStart=TRUE,
                          chnDim=TRUE,
                          usedDS=FALSE)
metric<-metric$new()
metric$update(pred,target)
metric$compute()

metric<-luz_metric_precision(nCls=1,
                             smooth=1e-8,
                             mode = "binary",
                             average="macro",
                             zeroStart=TRUE,
                             chnDim=TRUE,
                             usedDS=FALSE)
metric<-metric$new()
metric$update(pred,target)
metric$compute()

# Multiclass examples -----------------------------------------------------

refC <- terra::rast("data/metricCheck/multiclass_reference.tif")
predL <- terra::rast("data/metricCheck/multiclass_logits.tif")
predC <- terra::rast("data/metricCheck/multiclass_prediction.tif")
names(refC) <- "reference"
names(predC) <- "prediction"
cm <- terra::crosstab(c(predC, refC))
yardstick::f_meas(cm, estimator="macro")
yardstick::accuracy(cm, estimator="micro")
yardstick::recall(cm, estimator="macro")
yardstick::precision(cm, estimator="macro")

predL <- terra::as.array(predL)
refC <- terra::as.array(refC)

target <- torch::torch_tensor(refC, dtype=torch::torch_long())
pred <- torch::torch_tensor(predL, dtype=torch::torch_float32())
target <- target$permute(c(3,1,2))
pred <- pred$permute(c(3,1,2))

target <- target$unsqueeze(1)
pred <- pred$unsqueeze(1)

target <- torch::torch_cat(list(target, target), dim=1)
pred <- torch::torch_cat(list(pred, pred), dim=1)

#F1-Score
metric<-luz_metric_f1score(nCls=5,
                           smooth=1e-8,
                           mode = "multiclass",
                           average="macro",
                           zeroStart=TRUE,
                           chnDim=TRUE,
                           usedDS=FALSE)
metric<-metric$new()
metric$update(pred,target)
metric$compute()

#Recall
metric<-luz_metric_recall(nCls=5,
                          smooth=1e-8,
                          mode = "multiclass",
                          average="macro",
                          zeroStart=TRUE,
                          chnDim=TRUE,
                          usedDS=FALSE)
metric<-metric$new()
metric$update(pred,target)
metric$compute()

#Precision
metric<-luz_metric_precision(nCls=5,
                             smooth=1e-8,
                             mode = "multiclass",
                             average="macro",
                             zeroStart=TRUE,
                             chnDim=TRUE,
                             usedDS=FALSE)
metric<-metric$new()
metric$update(pred,target)
metric$compute()
```
## Unified Focal Loss Examples (still working on binary classification version)

``` r
library(geodl)
setwd("C:/Users/vidcg/Dropbox/code_dev/geodl/")

# Multiclass examples -----------------------------------------------------

refC <- terra::rast("C:/Users/vidcg/Dropbox/code_dev/geodl/data/metricCheck/multiclass_reference.tif")
predL <- terra::rast("C:/Users/vidcg/Dropbox/code_dev/geodl/data/metricCheck/multiclass_logits.tif")

predL <- terra::as.array(predL)
refC <- terra::as.array(refC)

#Dice loss example
target <- torch::torch_tensor(refC, dtype=torch::torch_long())
pred <- torch::torch_tensor(predL, dtype=torch::torch_float32())
target <- target$permute(c(3,1,2))
pred <- pred$permute(c(3,1,2))

target <- target$unsqueeze(1)
pred <- pred$unsqueeze(1)

target <- torch::torch_cat(list(target, target), dim=1)
pred <- torch::torch_cat(list(pred, pred), dim=1)

define_unified_focal_loss(pred=pred,
                          target=target,
                          nCls=5,
                          lambda=0, #Only use region-based loss
                          gamma= 1,
                          delta= 0.5, #Equal weights for FP and FN
                          smooth = 1e-8,
                          chnDim=TRUE,
                          zeroStart=TRUE,
                          clsWghtsDist=1,
                          clsWghtsReg=1,
                          useLogCosH =FALSE)


#Tversky loss example
target <- torch::torch_tensor(refC, dtype=torch::torch_long())
pred <- torch::torch_tensor(predL, dtype=torch::torch_float32())
target <- target$permute(c(3,1,2))
pred <- pred$permute(c(3,1,2))

target <- target$unsqueeze(1)
pred <- pred$unsqueeze(1)
target <- torch::torch_cat(list(target, target), dim=1)
pred <- torch::torch_cat(list(pred, pred), dim=1)

define_unified_focal_loss(pred=pred,
                          target=target,
                          nCls=5,
                          lambda=0, #Only use region-based loss
                          gamma= 1,
                          delta= 0.6, #FP and FN not equally weighted
                          smooth = 1e-8,
                          chnDim=TRUE,
                          zeroStart=TRUE,
                          clsWghtsDist=1,
                          clsWghtsReg=1,
                          useLogCosH =FALSE)

#CE loss example
target <- torch::torch_tensor(refC, dtype=torch::torch_long())
pred <- torch::torch_tensor(predL, dtype=torch::torch_float32())
target <- target$permute(c(3,1,2))
pred <- pred$permute(c(3,1,2))

target <- target$unsqueeze(1)
pred <- pred$unsqueeze(1)
target <- torch::torch_cat(list(target, target), dim=1)
pred <- torch::torch_cat(list(pred, pred), dim=1)

define_unified_focal_loss(pred=pred,
                          target=target,
                          nCls=5,
                          lambda=1, #Only used distribution-based loss
                          gamma= 1,
                          delta= 0.5,
                          smooth = 1e-8,
                          chnDim=TRUE,
                          zeroStart=TRUE,
                          clsWghtsDist=1,
                          clsWghtsReg=1,
                          useLogCosH =FALSE)

#Combo loss example
target <- torch::torch_tensor(refC, dtype=torch::torch_long())
pred <- torch::torch_tensor(predL, dtype=torch::torch_float32())
target <- target$permute(c(3,1,2))
pred <- pred$permute(c(3,1,2))

target <- target$unsqueeze(1)
pred <- pred$unsqueeze(1)
target <- torch::torch_cat(list(target, target), dim=1)
pred <- torch::torch_cat(list(pred, pred), dim=1)

define_unified_focal_loss(pred=pred,
                          target=target,
                          nCls=5,
                          lambda=.5, #Use both distribution and region-based losses
                          gamma= 1,
                          delta= 0.6,
                          smooth = 1e-8,
                          chnDim=TRUE,
                          zeroStart=TRUE,
                          clsWghtsDist=1,
                          clsWghtsReg=1,
                          useLogCosH =FALSE)
```


## Semantic Segmentation Workflow (Needs updated to reflect most recent version)

``` r
#=================Preparation===================================================

#Load packages
library(torch)
library(luz)
library(geodl)
library(dplyr)
library(ggplot2)


#Set device to GPU
device <- torch_device("cuda")

#=================Data==========================================================

#Create training chips Data Frame
trainDF <- makeChipsDF(folder="PATH/chips/train/",
                        extension=".png",
                        mode="Divided",
                        shuffle=TRUE,
                        saveCSV=FALSE)

#Create validation chips Data Frame
valDF <- makeChipsDF(folder="PATH/chips/val/",
                       extension=".png",
                       mode="Divided",
                       shuffle=TRUE,
                       saveCSV=FALSE)

#Subset chips
trainDF <- trainDF |> filter(division == "Positive")
valDF <- valDF |> sample_frac(.4)

#View subset of created chips as check
viewChips(chpDF=trainDF,
          folder="PATH
          /chips/train/",
          nSamps = 25,
          mode = "both",
          justPositive = TRUE,
          cCnt = 5,
          rCnt = 5,
          r = 1,
          g = 2,
          b = 3,
          rescale = FALSE,
          rescaleVal = 1,
          cNames=c("Background", "Mines"),
          cColors=c("gray", "red"),
          useSeed = TRUE,
          seed = 42)

#Define training dataset and augmentations
trainDS <- defineSegDataSet(
                    chpDF=trainDF,
                    folder="PATH/chips/train/",
                    normalize = FALSE,
                    rescaleFactor = 255,
                    mskRescale=255,
                    bands = c(1,2,3),
                    mskLong = FALSE,
                    chnDim = TRUE,
                    doAugs = TRUE,
                    maxAugs = 1,
                    probVFlip = .5,
                    probHFlip = .5,
                    probBrightness = 0,
                    probContrast = 0,
                    probGamma = 0,
                    probHue = 0,
                    probSaturation = 0,
                    brightFactor = c(.9,1.1),
                    contrastFactor = c(.9,1.1),
                    gammaFactor = c(.9, 1.1, 1),
                    hueFactor = c(-.1, .1),
                    saturationFactor = c(.9, 1.1))

#Define validation dataset
valDS <- defineSegDataSet(
                    chpDF=valDF,
                    folder="PATH/chips/val/",
                    normalize = FALSE,
                    rescaleFactor = 255,
                    mskRescale=255,
                    bands = c(1,2,3),
                    mskLong = FALSE,
                    chnDim = TRUE,
                    doAugs = FALSE,
                    maxAugs = 0,
                    probVFlip = 0,
                    probHFlip = 0)

#Check dataset lengths
length(trainDS)
length(valDS)

#Create dataloaders
trainDL <- torch::dataloader(trainDS,
                             batch_size=24,
                             shuffle=TRUE,
                             drop_last = TRUE)

valDL <- torch::dataloader(valDS,
                           batch_size=24,
                           shuffle=TRUE,
                           drop_last = TRUE)

#View a batch as a check
viewBatch(dataLoader=trainDL,
          chnDim = TRUE,
          mskLong = TRUE,
          nRows = 4,
          r = 1,
          g = 2,
          b = 3,
          cNames=c("Background", "Mines"),
          cColors=c("gray", "red"))

viewBatch(dataLoader=valDL,
          chnDim=TRUE,
          mskLong = TRUE,
          nRows = 4,
          r = 1,
          g = 2,
          b = 3,
          cNames=c("Background", "Mines"),
          cColors=c("gray", "red"))

#Get batch stats as a check
trainStats <- describeBatch(trainDL)
valStats <- describeBatch(valDL)
trainStats
valStats

#================Training=======================================================

#Train model
fitted <- baseUNet %>%
  luz::setup(
    loss = defineDiceLossFamily(nCls=1,
                                smooth=1,
                                mode="binary",
                                alpha=.6,
                                beta=.4,
                                tversky=TRUE,
                                mskLong=FALSE),
    optimizer = optim_adamw,
    metrics = list(
      luz_metric_binary_accuracy_with_logits(),
      luz_metric_f1score(mode="binary", smooth=1),
      luz_metric_recall(mode="binary", smooth=1),
      luz_metric_precision(mode="binary", smooth=1)
    )
  ) %>%
  set_hparams(nChn = 3,
              nCls=1,
              encoderChn = c(16,32,64,128),
              decoderChn = c(128,64,32,16),
              botChn = 256,
              useLeaky = TRUE,
              negative_slope = 0.01) %>%
  set_opt_hparams(lr = 0.01) %>%
  fit(data=trainDL,
      valid_data=valDL,
      epochs = 25,
      callbacks = list(luz_callback_csv_logger("PATH/trainLogs.csv"),
                       luz_callback_early_stopping(monitor="valid_loss",
                                                   patience=4,
                                                   mode='min'),
                       luz_callback_lr_scheduler(lr_one_cycle,
                                                 max_lr = 0.01,
                                                 epochs = 25,
                                                 steps_per_epoch = length(trainDL),
                                                 call_on = "on_batch_end"),
                       luz_callback_model_checkpoint(path="PATH/models/",
                                                     monitor="valid_loss",
                                                     save_best_only=TRUE,
                                                     mode="min",
                                                     )),
      accelerator = accelerator(device_placement = TRUE,
                                cpu = FALSE,
                                cuda_index = torch::cuda_current_device()),
      verbose=TRUE)

#======================Load a trained model from disk===========================

fitted <- baseUNet %>%
  luz::setup(
    loss = defineDiceLossFamily(nCls=1,
                                smooth=1,
                                mode="binary",
                                alpha=.6,
                                beta=.4,
                                tversky=TRUE,
                                mskLong=FALSE),
    optimizer = optim_adamw,
    metrics = list(
      luz_metric_binary_accuracy_with_logits(),
      luz_metric_f1score(mode="binary", smooth=1),
      luz_metric_recall(mode="binary", smooth=1),
      luz_metric_precision(mode="binary", smooth=1)
    )
  ) %>%
  set_hparams(nChn = 3,
              nCls=1,
              encoderChn = c(16,32,64,128),
              decoderChn = c(128,64,32,16),
              botChn = 256,
              useLeaky = TRUE,
              negative_slope = 0.01) %>%
  set_opt_hparams(lr = 0.01) %>%
  fit(data=trainDL,
    valid_data=valDL,
    epochs = 0,
    callbacks = list(luz_callback_csv_logger("PATH/trainLogs.csv"),
                     luz_callback_early_stopping(monitor="valid_loss",
                                                 patience=4,
                                                 mode='min'),
                     luz_callback_lr_scheduler(lr_one_cycle,
                                               max_lr = 0.01,
                                               epochs = 25,
                                               steps_per_epoch = length(trainDL),
                                               call_on = "on_batch_end"),
                     luz_callback_model_checkpoint(path="PATH/models/",
                                                   monitor="valid_loss",
                                                   save_best_only=TRUE,
                                                   mode="min",
                     )),
    accelerator = accelerator(device_placement = TRUE,
                              cpu = FALSE,
                              cuda_index = torch::cuda_current_device()),
    verbose=TRUE)


luz_load_checkpoint(fitted, "PATH/chips/models/epoch-22-valid_loss-0.107.pt")



#==================Assessment===================================================

#Plot losses/metrics
allMets <- read.csv("PATH/chips/models/trainLogs.csv")
ggplot(allMets, aes(x=epoch, y=loss, color=set))+
  geom_line(lwd=1)+
  labs(x="Epoch", y="Loss", color="Set")

ggplot(allMets, aes(x=epoch, y=f1score, color=set))+
  geom_line(lwd=1)+
  labs(x="Epoch", y="F1-Score", color="Set")

ggplot(allMets, aes(x=epoch, y=recall, color=set))+
  geom_line(lwd=1)+
  labs(x="Epoch", y="Recall", color="Set")

ggplot(allMets, aes(x=epoch, y=precision, color=set))+
  geom_line(lwd=1)+
  labs(x="Epoch", y="Precision", color="Set")

#View batch of predictions
viewBatchPreds(dataLoader=trainDL,
               model = fitted,
               mode="binary",
               chnDim=TRUE,
               mskLong=FALSE,
               nRows=6,
               r=1,
               g=2,
               b=3,
               cNames=c("Background", "Mine"),
               cColors=c("gray", "red"),
               useCUDA=TRUE,
               probs=FALSE)

viewBatchPreds(dataLoader=trainDL,
               model = fitted,
               mode="binary",
               chnDim=TRUE,
               mskLong=FALSE,
               nRows=6,
               r=1,
               g=2,
               b=3,
               cNames=c("Background", "Mine"),
               cColors=c("gray", "red"),
               useCUDA=TRUE,
               probs=TRUE)

#Assess with test set
testDF <- makeChipsDF(folder="PATH/chips/test/",
                     extension=".png",
                     mode="Divided",
                     shuffle=TRUE,
                     saveCSV=FALSE)

testDS <- defineSegDataSet(
  chpDF=testDF,
  folder="PATH/chips/test/",
  normalize = FALSE,
  rescaleFactor = 255,
  mskRescale=255,
  bands = c(1,2,3),
  mskLong = FALSE,
  chnDim = TRUE,
  doAugs = FALSE,
  maxAugs = 0,
  probVFlip = 0,
  probHFlip = 0)

testDL <- torch::dataloader(testDS,
                           batch_size=24,
                           shuffle=TRUE,
                           drop_last = TRUE)

testEval <- fitted %>% evaluate(data=testDL)
assMets <- get_metrics(testEval)
print(assMets)

#==========================Inference============================================
#Use model to predict to spatial data
predCls <- predictSpatial(imgIn="PATH/inputData.tif",
                          model=fitted,
                          predOut="PATH/predictedClasses.tif",
                          mode="binary",
                          probs=FALSE,
                          useCUDA=TRUE,
                          nCls=1,
                          chpSize=256,
                          stride_x=128,
                          stride_y=128,
                          crop=50,
                          nChn=3,
                          rescaleFactor=255)

predProbs <- predictSpatial(imgIn="PATH/inputData.tif",
                        model=fitted,
                        predOut="PATH/predictedProbs.tif",
                        mode="binary",
                        probs=TRUE,
                        useCUDA=TRUE,
                        nCls=1,
                        chpSize=256,
                        stride_x=128,
                        stride_y=128,
                        crop=50,
                        nChn=3,
                        rescaleFactor=255)

terra::plotRGB(terra::rast("PATH/inputData.tif"))
terra::plot(predCls, type="classes",axes=FALSE, levels=c("Background", "Mine"), col=c("Gray", "Red"))
terra::plot(predProbs, type="continuous",axes=FALSE, col=terra::map.pal("grey"))
```
