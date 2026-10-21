# Package index

## Terrain Derivatives

Compute land surface parameters (LSPs) from a digital terrain model and
build multi-band terrain visualizations, using either terra or torch.

- [`makeSlope()`](https://maxwell-geospatial.github.io/geodl/reference/makeSlope.md)
  : makeSlope
- [`makeAspect()`](https://maxwell-geospatial.github.io/geodl/reference/makeAspect.md)
  : makeAspect
- [`makeCrv()`](https://maxwell-geospatial.github.io/geodl/reference/makeCrv.md)
  : makeCrv
- [`makeTPI()`](https://maxwell-geospatial.github.io/geodl/reference/makeTPI.md)
  : makeTPI
- [`makeTRI()`](https://maxwell-geospatial.github.io/geodl/reference/makeTRI.md)
  : makeTRI
- [`makeHillshade()`](https://maxwell-geospatial.github.io/geodl/reference/makeHillshade.md)
  : makeAspect
- [`makeTerrainVisTerra()`](https://maxwell-geospatial.github.io/geodl/reference/makeTerrainVisTerra.md)
  : makeTerrainVisTerra
- [`makeTerrainVisTorch()`](https://maxwell-geospatial.github.io/geodl/reference/makeTerrainVisTorch.md)
  : makeTerrainVisTerra

## Chip Generation and Inspection

Generate masks and image/mask chips from larger rasters, list and
describe chips, and visually check chips prior to training.

- [`makeMasks()`](https://maxwell-geospatial.github.io/geodl/reference/makeMasks.md)
  : makeMasks
- [`makeChips()`](https://maxwell-geospatial.github.io/geodl/reference/makeChips.md)
  : makeChips
- [`makeChipsMultiClass()`](https://maxwell-geospatial.github.io/geodl/reference/makeChipsMultiClass.md)
  : makeChipsMultiClass
- [`makeChipsDF()`](https://maxwell-geospatial.github.io/geodl/reference/makeChipsDF.md)
  : makeChipsDF
- [`makeBoundaryDistance()`](https://maxwell-geospatial.github.io/geodl/reference/makeBoundaryDistance.md)
  : makeBoundaryDistance
- [`describeChips()`](https://maxwell-geospatial.github.io/geodl/reference/describeChips.md)
  : describeChips
- [`viewChips()`](https://maxwell-geospatial.github.io/geodl/reference/viewChips.md)
  : viewChips
- [`addCoordNoise()`](https://maxwell-geospatial.github.io/geodl/reference/addCoordNoise.md)
  : addCoordNoise

## Dynamic Chip Generation

Generate chips on-the-fly from points and larger rasters during training
instead of writing chips to disk beforehand.

- [`makeDynamicChipsSF()`](https://maxwell-geospatial.github.io/geodl/reference/makeDynamicChipsSF.md)
  : makeDynamicChipsSF
- [`saveDynamicChips()`](https://maxwell-geospatial.github.io/geodl/reference/saveDynamicChips.md)
  : saveDynamicChips
- [`checkDynamicChips()`](https://maxwell-geospatial.github.io/geodl/reference/checkDynamicChips.md)
  : checkDymamicChips

## Datasets and DataLoaders

torch dataset subclasses for geospatial semantic segmentation and
functions for inspecting a mini-batch.

- [`defineSegDataSet()`](https://maxwell-geospatial.github.io/geodl/reference/defineSegDataSet.md)
  : defineSegDataSet
- [`defineDynamicSegDataSet()`](https://maxwell-geospatial.github.io/geodl/reference/defineDynamicSegDataSet.md)
  [`defineDynmamicSegDataSet()`](https://maxwell-geospatial.github.io/geodl/reference/defineDynamicSegDataSet.md)
  : defineDynamicSegDataSet
- [`describeBatch()`](https://maxwell-geospatial.github.io/geodl/reference/describeBatch.md)
  : describeBatch
- [`viewBatch()`](https://maxwell-geospatial.github.io/geodl/reference/viewBatch.md)
  : viewBatch
- [`viewBatchPreds()`](https://maxwell-geospatial.github.io/geodl/reference/viewBatchPreds.md)
  : viewBatchPreds
- [`viewDist()`](https://maxwell-geospatial.github.io/geodl/reference/viewDist.md)
  : viewDist

## Model Architectures

Configurable UNet-family semantic segmentation architectures as
torch::nn_module() subclasses.

- [`defineUNet()`](https://maxwell-geospatial.github.io/geodl/reference/defineUNet.md)
  : defineUNet
- [`defineUNet3p()`](https://maxwell-geospatial.github.io/geodl/reference/defineUNet3p.md)
  : defineUNet3p
- [`defineMobileUNet()`](https://maxwell-geospatial.github.io/geodl/reference/defineMobileUNet.md)
  : defineMobileUNet
- [`defineResNetUNet()`](https://maxwell-geospatial.github.io/geodl/reference/defineResNetUNet.md)
  : defineResNetUNet
- [`defineEfficientUNetB2()`](https://maxwell-geospatial.github.io/geodl/reference/defineEfficientUNetB2.md)
  : defineEfficientUNetB2
- [`defineDLab()`](https://maxwell-geospatial.github.io/geodl/reference/defineDLab.md)
  : defineDLab
- [`defineTerrainSeg()`](https://maxwell-geospatial.github.io/geodl/reference/defineTerrainSeg.md)
  : defineTerrainSeg
- [`countParams()`](https://maxwell-geospatial.github.io/geodl/reference/countParams.md)
  : countParams

## Loss Functions

Unified focal loss and regression loss implementations for luz.

- [`defineUnifiedFocalLoss()`](https://maxwell-geospatial.github.io/geodl/reference/defineUnifiedFocalLoss.md)
  : defineUnifiedFocalLoss
- [`defineUnifiedFocalLossDS()`](https://maxwell-geospatial.github.io/geodl/reference/defineUnifiedFocalLossDS.md)
  : defineUnifiedFocalLossDS
- [`defineRegressionLoss()`](https://maxwell-geospatial.github.io/geodl/reference/defineRegressionLoss.md)
  : defineRegressionLoss
- [`make_ds_weights()`](https://maxwell-geospatial.github.io/geodl/reference/make_ds_weights.md)
  : make_ds_weights

## Training Metrics

luz_metric() implementations for semantic segmentation.

- [`luz_metric_overall_accuracy()`](https://maxwell-geospatial.github.io/geodl/reference/luz_metric_overall_accuracy.md)
  : luz_metric_overall_accuracy
- [`luz_metric_f1score()`](https://maxwell-geospatial.github.io/geodl/reference/luz_metric_f1score.md)
  : luz_metric_f1score
- [`luz_metric_recall()`](https://maxwell-geospatial.github.io/geodl/reference/luz_metric_recall.md)
  : luz_metric_recall
- [`luz_metric_precision()`](https://maxwell-geospatial.github.io/geodl/reference/luz_metric_precision.md)
  : luz_metric_precision

## Training Callbacks

luz_callback() implementations for staged training, EMA, and logging.

- [`callback_save_model_state_dict()`](https://maxwell-geospatial.github.io/geodl/reference/callback_save_model_state_dict.md)
  : callback_save_model_state_dict
- [`callback_per_stage_optimizer()`](https://maxwell-geospatial.github.io/geodl/reference/callback_per_stage_optimizer.md)
  : callback_per_stage_optimizer
- [`callback_freeze_stages()`](https://maxwell-geospatial.github.io/geodl/reference/callback_freeze_stages.md)
  : callback_freeze_stages
- [`callback_unfreeze_stages()`](https://maxwell-geospatial.github.io/geodl/reference/callback_unfreeze_stages.md)
  : callback_unfreeze_stages
- [`callback_ema()`](https://maxwell-geospatial.github.io/geodl/reference/callback_ema.md)
  : callback_ema
- [`callback_ds_weight_decay()`](https://maxwell-geospatial.github.io/geodl/reference/callback_ds_weight_decay.md)
  : callback_ds_weight_decay
- [`callback_reconfigure_loss()`](https://maxwell-geospatial.github.io/geodl/reference/callback_reconfigure_loss.md)
  : callback_reconfigure_loss
- [`callback_training_log()`](https://maxwell-geospatial.github.io/geodl/reference/callback_training_log.md)
  : callback_training_log

## Reproducibility

Seed R and torch RNG streams for reproducible workflows.

- [`setSeed()`](https://maxwell-geospatial.github.io/geodl/reference/setSeed.md)
  : setSeed

## Prediction and Accuracy Assessment

Generate spatial predictions from a trained model and assess accuracy
from rasters, dataloaders, or tabular reference/predicted values.

- [`predictSpatial()`](https://maxwell-geospatial.github.io/geodl/reference/predictSpatial.md)
  : predictSpatial
- [`assessDL()`](https://maxwell-geospatial.github.io/geodl/reference/assessDL.md)
  : assessDL
- [`assessRaster()`](https://maxwell-geospatial.github.io/geodl/reference/assessRaster.md)
  : assessRaster
- [`assessPnts()`](https://maxwell-geospatial.github.io/geodl/reference/assessPnts.md)
  : assessPnts

## Training Diagnostics

- [`plotTrainingSchedule()`](https://maxwell-geospatial.github.io/geodl/reference/plotTrainingSchedule.md)
  : plotTrainingSchedule
