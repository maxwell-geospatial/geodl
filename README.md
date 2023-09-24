
<!-- README.md is generated from README.Rmd. Please edit that file -->

# geodl <img src="geodlHex.png" align="right" width="132" />

<!-- badges: start -->
<!-- badges: end -->

This package provides utilities and functions for semantic segmentation of geospatial data using convolutional neural network-based deep learning. Functions allow for creating masks, image chips, data frames listing image chips in a directory, and datasets for use within DataLoaders. A basic UNet architecture is provided, and more UNet-like architectures will be made available in later releases. Dice and Dice-like loss metrics are also available along with F1-score, recall, and precision assessment metrics implemented with luz. Trained models can be used to predict to spatial data without the need to generate chips from larger spatial extents. The package relies on torch for implementing deep learning, which does not require the installation of a Python environment. Raster geospatial data are handled with terra. Models can be trained using a CUDA-enabled GPU; however, multi-GPU training is not supported by torch. Both binary and multiclass models can be trained. 

This package is still experimental and is a work-in-progress. We hope to add additional semantic segmentation architectures in future releases. 

## Installation

You can install the development version of geodl from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("maxwell-geospatial/geodl")
```

## Some considerations

1. Input data should generally be scaled from 0 to 1. defineSegDataSet() provides options for rescaling and normalizing data. 
2. A binary classification problem can be framed to predict a single, positive case logit or to predict both a positive and negative class logit. When both positive and negative case logits are predicted, you should use multiclass assessment metrics and losses. 
3. The model will return logits as opposed to probabilities for both binary classification and multiclass classification. The loss and assessment metrics provided as part of the package expect logits as opposed to probabilities. If you use binary cross entropy as a loss metric for a binary classification problem in which only the positive-case logit is returned, you should use torch::nn_bce_with_logits_loss() as opposed to torch:: nn_bce_loss(). For multiclass classification, you should use torch::nn_cross_entropy_loss(). 
4. If you want to use the dice-like loss metrics provided as part of this package (Dice, Focal Dice, Tversky, and Focal Tversky), please consult the documentation for the correct configuration. 
5. If using the overall accuracy metrics provided by luz for a binary classification where only the positive-case logit is returned, use luz::luz_metric_binary_accuracy_with_logits(). For multiclass classification, use luz::luz_metric_accuracy(). 
6. When using our Dice-based losses and/or the F1-score, recall, and precision assessment metrics in macro-averaging mode, you can assign class weights to control the relative impact of each class in the calculations. Weights can be set to 0 to ignore a specific class, such as when labels are sparse or incomplete. 
7. If a Dice-like loss is used as a means to combat class imbalance in a multiclass classification, we recommend a macro-averaging as opposed to micro-averaging method.
8. Our chip generation routines will only return chips in which all cells contain values. Any chips with NULL or NoData cells are not produced. So, when extents are not rectangular, the full spatial extent will not be chipped. 
9. Chips can be generated such that presence and background-only chips are written to separate folders. 
10. It is a good idea to check your data using the provided utility functions: decribeBatch(), describeChips(), viewBatch(), and viewChips().
11. The predictSpatial() function allows for returning both predicted class indices and predicted class probabilities following the application of a sigmoid or softmax activation. 
12. Care should be taken when generating target masks. Our loss and assessment metrics assume that targets are created using a torch_long datatype as opposed to a torch_float32 datatype. However, some other loss metrics may expected a torch_float32 data type. The mskLong parameter, provided for several functions, can be used to deal with this issue. 

## Data Preparation Examples

```r
library(geodl)
#Create terrain derivatives from DTM
inDTM <- terra::rast("data/elev/dtm.tif")
terrOut <- makeTerrainDerivatives(dtm=inDTM, res=2, filename="data/elev/stack.tif")
terra::plotRGB(terrOut*255)

#Make mask
makeMasks(image = "data/toChip/image/KY_Saxton_709705_1970_24000_geo.tif",
          features = "data/toChip/msks/KY_Saxton_709705_1970_24000_geo.shp",
          crop = TRUE,
          extent = "data/toChip/extent/KY_Saxton_709705_1970_24000_geo.shp",
          field = "classvalue",
          background = 0,
          outImage = "data/toChip/output/topoOut.tif",
          outMask = "data/toChip/output/mskOut.tif",
          mode = "Both")

terra::plotRGB(terra::rast("data/toChip/output/topoOut.tif"))
terra::plot(terra::rast("data/toChip/output/mskOut.tif"))

#Make chips
makeChips(image = "data/toChip/output/topoOut.tif",
          mask = "data/toChip/output/mskOut.tif",
          n_channels = 3,
          size = 256,
          stride_x = 256,
          stride_y = 256,
          outDir = "data/toChip/chips/",
          mode = "Positive")

makeChipsMultiClass(image = "data/toChip/output/topoOut.tif",
                    mask = "data/toChip/output/mskOut.tif",
                    n_channels = 3,
                    hasZero = TRUE,
                    size = 256,
                    stride_x = 256,
                    stride_y = 256,
                    outDir = "data/toChip/chips/")

#Describe chips
lstOut <- describeChips(folder = "data/toChip/chips/",
              extension = ".tif",
              mode="Positive",
              subSample = TRUE,
              numChips=200,
              subSamplePix=TRUE,
              sampsPerChip=100)

print(lstOut$ImageStats)
print(lstOut$mskStats)

#Describe chips
chpDF <- makeChipsDF(folder = "data/toChip/chips/",
                    outCSV = "data/toChip/chips/chipsDF.csv",
                    extension = ".tif",
                    mode="Positive",
                    shuffle=FALSE,
                    saveCSV=TRUE)

#Assess using point locations

#Example 1: table that already has the reference and predicted labels for a multiclass classification
mcIn <- readr::read_csv("data/tables/multiClassExample.csv")
myMetrics <- assessPnts(reference=mcIn$ref,
                       predicted=mcIn$pred,
                       multiclass=TRUE)

#Example 2: table that already has the reference and predicted labels for a binary classification
bIn <- readr::read_csv("data/tables/binaryExample.csv")
myMetrics <- assessPnts(reference=bIn$ref,
                       predicted=bIn$pred,
                       multiclass=FALSE,
                       positive_case = "Mine")

#Example 3: Read in point layer and intersect with rater output
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

#Assess using raster grids
refG <- terra::rast("data/topoResult/topoRef.tif")
predG <- terra::rast("data/topoResult/topoPred.tif")
refG2 <- crop(project(refG, predG), predG)
myMetrics <- assessRaster(reference = refG2,
                          predicted = predG,
                          multiclass = FALSE,
                          mappings = c("Not Mine", "Mine"),
                          positive_case = "Mine")
```

## Semantic Segmentation Workflow

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
