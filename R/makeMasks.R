#' makeMasks
#'
#' Make raster mask from input vector data
#'
#' This function creates a raster mask from input vector data. The cell value is
#' indicated by the field parameter. A unique numeric code should be provided for
#' each class. In the case of a binary classification, 0 should indicate background
#' and 1 should indicate positive. For a multiclass problem, values should be sequential
#' from 0 to n-1, where n is the number of classes, or 1 to n. We recommend using
#' 0 to n-1. If no cropping is applied, the generated raster mask should have the
#' same spatial extent, number of rows of pixels, number of columns of pixels,
#' cell size, and coordinate reference system as the input image.
#'
#' @param image File name and full path or path relative to working directory for image.
#' Image is converted to a spatRaster internally.
#' @param features File name and full path or path relative to working directory
#' for vector mask or label data. A field should be provided that differentiates
#' classes using unique numeric codes as explained above. If the input features
#' use a different coordinate reference system then the input image, the features
#' will be reprojected to match the image. Vector data are converted to a
#' SpatVector object internally.
#' @param crop TRUE or FALSE. Whether or not to crop the input image data relative
#' to a defined vector extent. The default is FALSE.
#' @param extent File name and full path or path relative to working directory for
#' vector extent data. If the extent uses a different coordinate reference system
#' then the input image, the features will be reprojected to match the image.
#' Vector data are converted to a SpatVector object internally.
#' @param field The name of the field in the feature vector data that differentiate
#' classes using a unique numeric code with an integer data type. Field name should
#' be provided as a string.
#' @param background The numeric value to assign to the background class. The default
#' is 0. If the full spatial extent has labels in the input feature data, no background
#' value will be applied. For binary classification problems, the background should be
#' coded to 0 and the positive case should be coded to 1. It is not necessary to
#' include the background class in the vector feature data.
#' @param outImage Image output name in TIFF format (".tif") with full path or path
#' relative to working directory for image. This output will only be generated if
#' the mode is set to "Both".
#' @param outMask Mask output name in TIFF format (".tif") with full path or path
#' relative to working directory for image. Output will be a single-band raster
#' grid of class numeric codes.
#' @param mode Either "Both" or "Mask". If "Both", a copy of the image will be made
#' along with the generated raster mask. If "Mask", only the mask is produced. If
#' you are experiencing issues with alignment between the image and associated mask,
#' setting the mode to "Both" can alleviate this issue. However, this will result in
#' more data being written to disk.
#' @return Single-band raster mask written to disk in TIFF format and, optionally,
#' a copy of the image written to disk. Cropping may be applied as specified.
#' No R objects are returned.
#' @export
makeMasks <- function(image,
                      features,
                      crop = FALSE,
                      extent,
                      field,
                      background = 0,
                      outImage,
                      outMask,
                      mode = "Both"){
  imgData <- terra::rast(image)
  featData <- terra::vect(features)
  if(terra::crs(imgData) != terra::crs(featData)){
    featData <- terra::project(featData, imgData)
  }
  if(crop==TRUE){
    extData <- terra::vect(extent)
    if(terra::crs(imgData) != terra::crs(extData)){
      extData <- terra::project(extData, imgData)
    }
    imgData <- terra::crop(imgData, extData)
  }
  maskR <- terra::rasterize(featData, imgData, field=field, background=background)
  if(mode=="Both"){
    terra::writeRaster(imgData, outImage)
    terra::writeRaster(maskR, outMask)
  } else if(mode=="Mask"){
    terra::writeRaster(maskR, outMask)
  } else{
    print("Invalid Mode.")
  }
}
