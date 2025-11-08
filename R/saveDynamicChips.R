#' saveDynamicChips
#'
#' Save chips meant to be generated dynamically to disk.
#'
#' Save chips defined by defineDynamicChips() to disk. It is not required to save
#' dynamically generated chips to disk. This is primarily a utility function.
#'
#' @param chipsSF output from defineDynmaicChips().
#' @param chipSize size of chips to generate. Default is 512 (512-by512 cells).
#' @param cellSize cell size of input and output data. Default is 1 m.
#' @param outDir full or relative path to output directory. Must include final forward slash in path.
#' @param mode either "All", "Positive", or "Divided". If "All", all chips and masks are saved.
#' If "Positive", only chips and masks containing positive cells are maintained. if "Divided",
#' background-only and positive-containing chips are saved but written to separate directories.
#' For multiclass, use "All". Default is "All".
#' @param useExistingDir TRUE or FALSE. Whether or not to use a directory that that already
#' contains chips. Default is FALSE.
#' @param doJitter whether or not to add random jitter to center coordinate of chips. Default is FALSE.
#' @param jitterSD standard deviation to use when applying jitter. Default is 15.
#' @param useSeed whether or not to use a random seed when incorporating jitter. Default is TRUE but only
#' applies if using jitter.
#' @param seed random seed value. Default is 42.
#' @return chips and masks written to disk. No R object is returned.
#' @export
saveDynamicChips <- function(chipsSF,
                             chipSize=512,
                             cellSize=1,
                             outDir,
                             mode = "All",
                             useExistingDir = FALSE,
                             doJitter=FALSE,
                             jitterSD=15,
                             useSeed=TRUE,
                             seed=42){

    if(mode == "All"){
      if(useExistingDir == FALSE){
        dir.create(paste0(outDir, "/images"))
        dir.create(paste0(outDir, "/masks"))
      }

      for(i in 1:nrow(chipsSF)){
        c1 <- makeDynamicChip(chipIn=chipsSF[i,],
                              chipSize=chipSize,
                              cellSize=cellSize,
                              doJitter=doJitter,
                              jitterSD=jitterSD,
                              useSeed=useSeed,
                              seed=seed)

        naCntImg = terra::global(is.na(c1$image), fun = "sum")[1,1]
        naCntMsk = terra::global(is.na(c1$mask), fun = "sum")[1,1]

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
                              cellSize=cellSize,
                              doJitter=doJitter,
                              jitterSD=jitterSD,
                              useSeed=useSeed,
                              seed=seed)
        naCntImg = terra::global(is.na(c1$image), fun = "sum")[1,1]
        naCntMsk = terra::global(is.na(c1$mask), fun = "sum")[1,1]
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
                              cellSize=cellSize,
                              doJitter=doJitter,
                              jitterSD=jitterSD,
                              useSeed=useSeed,
                              seed=seed)
        naCntImg = terra::global(is.na(c1$image), fun = "sum")[1,1]
        naCntMsk = terra::global(is.na(c1$mask), fun = "sum")[1,1]
        max_value_df <- terra::global(c1$mask, fun = max, na.rm = TRUE)[1,1]

        if(naCntImg == 0 & naCntMsk == 0 & max_value_df > 0){
          terra::writeRaster(c1$image,
                             paste0(outDir,
                                    "/images/positive/",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))
          terra::writeRaster(c1$mask,
                             paste0(c1$mask,
                                    "/masks/positive",
                                    "chip_",
                                    as.character(i),
                                    ".tif"))

        }else{
          terra::writeRaster(ic1$image,
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
