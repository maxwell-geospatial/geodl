#' makeChipsDF
#'
#' Create data frame and CSV file listing image chips and associated masks
#'
#' This function creates a data frame and, optionally, a CSV file that lists all
#' of the image chips and associated masks in a directory. Three columns are
#' produced. The chpN column provides the name of the chip, the chpPth column
#' provides the path to the chip, and the chpMsk column provides the path to the
#' associated mask. All paths are relative to the input folder as opposed to
#' the full file path so that the results can still be used if the data are copied
#' to a new location on disk or to a new computer.
#'
#' @param folder Full path or path relative to the working directory to the
#' folder containing the image chips and associated masks. You must include the
#' final forward slash in the path (e.g., "C:/data/chips/").
#' @param outCSV File name and full path or path relative to the working directory
#' for the resulting CSV file with a ".csv" extension.
#' @param extension The extension of the image and mask raster data (e.g., ".tif",
#' ".png", ".jpeg", or ".img"). The default is ".tif" since this is the file
#' format used by the utilities in this package. This option is provided if chips
#' are generated using another method.
#' @param mode Either "All", "Positive", or "Divided". This should match the setting
#' used in the makeChips() function. If the makeChipsMultiClass() function was used,
#' this should be set to "All" or left as the default. The default is "All".
#' @param shuffle TRUE or FALSE. Whether or not to shuffle the rows in the table.
#' Rows can be shuffled to potentially reduced autocorrelation in the data. The
#' default is FALSE.
#' @param saveCSV TRUE or FALSE. Whether or not to save the CSV file or just
#' return the data frame. If this is set to FALSE then the outCSV parameter is
#' ignored and no CSV file is generated. The default is FALSE.
#' @return Data frame with three columns (chpN, chpPth, and mskPth) and, optionally,
#' a CSV file written to disk. If mode = "Divided", a division column is added to
#' differentiate "positive" and "background" samples.
#' @examples
#' /dontrun{
#' chpDF <- makeChipsDF(folder = "PATHT TO CHIPS FOLDER",
#'                       outCSV = "OUTPUT CSV FILE AND PATH",
#'                       extension = ".tif",
#'                       mode="Positive",
#'                       shuffle=TRUE,
#'                       saveCSV=TRUE)
#' }
#' @export
makeChipsDF <- function(folder,
                        outCSV,
                        extension=".tif",
                        mode="All",
                        shuffle=FALSE,
                        saveCSV=FALSE){
  if(mode == "All" | mode == "Positive"){
    lstChps <- list.files(paste0(folder, "images/"), pattern=paste0("\\", extension, "$"))
    lstChpsPth <- paste0("images/", lstChps)
    lstMsksPth <- paste0("masks/", lstChps)
    chpDF <- data.frame(chpN=lstChps, chpPth=lstChpsPth, mskPth=lstMsksPth)
  }else{
    lstChpsB <- list.files(paste0(folder, "images/background/"), pattern=paste0("\\", extension, "$"))
    lstChpsP <- list.files(paste0(folder, "images/positive/"), pattern=paste0("\\", extension, "$"))
    lstChpsPthB <- paste0("images/background/", lstChpsB)
    lstMsksPthB <- paste0("masks/background/", lstChpsB)
    lstChpsPthP <- paste0("images/positive/", lstChpsP)
    lstMsksPthP <- paste0("masks/positive/", lstChpsP)
    chpDFB <- data.frame(chpN=lstChpsB, chpPth=lstChpsPthB, mskPth=lstMsksPthB)
    chpDFP <- data.frame(chpN=lstChpsP, chpPth=lstChpsPthP, mskPth=lstMsksPthP)
    chpDFP$division <- "Positive"
    chpDFB$division <- "Backround"
    chpDF <- dplyr::bind_rows(chpDFB, chpDFP)
  }
  if(shuffle == TRUE){
    chpDF <- chpDF |> dplyr::sample_n(nrow(chpDF), replace=FALSE)
  }
  if(saveCSV == TRUE){
    readr::write_csv(chpDF, outCSV, append=FALSE)
  }
  return(chpDF)
}
