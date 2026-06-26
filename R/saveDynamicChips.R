#' saveDynamicChips
#'
#' Save chips meant to be generated dynamically to disk.
#'
#' Save chips defined by makeDynamicChipsSF() to disk. It is not required to save
#' dynamically generated chips to disk. This is primarily a utility function.
#'
#' @param chipsSF output from makeDynamicChipsSF().
#' @param chipSize size of chips to generate. Default is 512 (512-by-512 cells).
#' @param cellSize cell size of input and output data. Default is 1 m.
#' @param outDir full or relative path to output directory. Must include final forward slash in path.
#' @param mode either "All", "Positive", or "Divided". If "All", all chips and masks are saved.
#' If "Positive", only chips and masks containing positive cells are maintained. If "Divided",
#' background-only and positive-containing chips are saved but written to separate directories.
#' For multiclass, use "All". Default is "All".
#' @param useExistingDir TRUE or FALSE. Whether or not to use a directory that already
#' contains chips. Default is FALSE.
#' @return chips and masks written to disk. No R object is returned.
#' @export
saveDynamicChips <- function(chipsSF,
                             chipSize=512,
                             cellSize=1,
                             outDir,
                             mode = "All",
                             useExistingDir = FALSE){

    if(mode == "All"){
      if(useExistingDir == FALSE){
        dir.create(paste0(outDir, "/images"))
        dir.create(paste0(outDir, "/masks"))
      }

      for(i in 1:nrow(chipsSF)){
        c1 <- makeDynamicChip(chipIn=chipsSF[i,],
                              chipSize=chipSize,
                              cellSize=cellSize)

        naCntImg <- terra::global(is.na(c1$image), fun = "sum")[1,1]
        naCntMsk <- terra::global(is.na(c1$mask), fun = "sum")[1,1]

        if(naCntImg == 0 & naCntMsk == 0){
          terra::writeRaster(c1$image,
                             paste0(outDir,
                                    "/images/",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))

          terra::writeRaster(c1$mask,
                             paste0(outDir,
                                    "/masks/",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))
        }
      }

    }else if(mode == "Positive"){
      if(useExistingDir == FALSE){
        dir.create(paste0(outDir, "/images"))
        dir.create(paste0(outDir, "/masks"))
      }

      for(i in 1:nrow(chipsSF)){
        c1 <- makeDynamicChip(chipIn=chipsSF[i,],
                              chipSize=chipSize,
                              cellSize=cellSize)

        naCntImg <- terra::global(is.na(c1$image), fun = "sum")[1,1]
        naCntMsk <- terra::global(is.na(c1$mask), fun = "sum")[1,1]
        max_value_df <- terra::global(c1$mask, fun = max, na.rm = TRUE)[1,1]

        if(naCntImg == 0 & naCntMsk == 0 & max_value_df > 0){
          terra::writeRaster(c1$image,
                             paste0(outDir,
                                    "/images/",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))

          terra::writeRaster(c1$mask,
                             paste0(outDir,
                                    "/masks/",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))
        }
      }

    }else if(mode == "Divided"){
      if(useExistingDir == FALSE){
        dir.create(paste0(outDir, "/images"))
        dir.create(paste0(outDir, "/masks"))
        dir.create(paste0(outDir, "/images/positive"))
        dir.create(paste0(outDir, "/images/background"))
        dir.create(paste0(outDir, "/masks/positive"))
        dir.create(paste0(outDir, "/masks/background"))
      }

      for(i in 1:nrow(chipsSF)){
        c1 <- makeDynamicChip(chipIn=chipsSF[i,],
                              chipSize=chipSize,
                              cellSize=cellSize)

        naCntImg <- terra::global(is.na(c1$image), fun = "sum")[1,1]
        naCntMsk <- terra::global(is.na(c1$mask), fun = "sum")[1,1]
        max_value_df <- terra::global(c1$mask, fun = max, na.rm = TRUE)[1,1]

        if(naCntImg == 0 & naCntMsk == 0 & max_value_df > 0){
          terra::writeRaster(c1$image,
                             paste0(outDir,
                                    "/images/positive/",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))
          terra::writeRaster(c1$mask,
                             paste0(outDir,
                                    "/masks/positive/",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))
        }else if(naCntImg == 0 & naCntMsk == 0){
          terra::writeRaster(c1$image,
                             paste0(outDir,
                                    "/images/background/",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))
          terra::writeRaster(c1$mask,
                             paste0(outDir,
                                    "/masks/background/",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))
        }
      }

    }else{
      message("Invalid Mode Provided.")
    }
}
