# geodl 0.3.0

* Support for three model architectures: UNet, UNet with MobileNetv2 encoder, and UNet3+
* UNet with MobileNetv2 encoder is no longer limited to three input predictor variables
* Assessment and prediction functions now expect a nn_module object as opposed to a luz fitted object
* Dynamically generate chips during training process as opposed to saving them to disk beforehand (still experimental)
* Ignore outer rows and columns of cells when calculating loss or assessment metrics if desired
* Use R torch to calculate several different land surface parameters (LSPs) from a digital terrain model: slope, hillshade, aspect, northwardness, eastwardness, transformed solar radiation aspect index (TRASP), site exposure index (SEI), topographic position index (TPI), and surface curvatures (mean, profile, and planform)
* Calculate three-band terrain visualization raster grid from a DTM using torch or terra
* New specialized model for extracting geomorphic features from digital terrain models (DTMs)
* New function to count the number of trainable parameters in a model
* Fixed issue with chip generation pipeline that caused some chips with NA cells to be written
* Updated atrous spatial pyramid pooling (ASPP) module to align with the version used within DeepLabv3+

# geodl 0.1.0

* Initial CRAN submission.
