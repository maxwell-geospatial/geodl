# makeChipsDF

Create data frame and CSV file listing image chips and associated masks

## Usage

``` r
makeChipsDF(
  folder,
  outCSV,
  extension = ".tif",
  mode = "All",
  shuffle = FALSE,
  saveCSV = FALSE,
  useEdgeDist = FALSE
)
```

## Arguments

- folder:

  Full path or path relative to the working directory to the folder
  containing the image chips and associated masks. You must include the
  final forward slash in the path (e.g., "C:/data/chips/").

- outCSV:

  File name and full path or path relative to the working directory for
  the resulting CSV file with a ".csv" extension.

- extension:

  The extension of the image and mask raster data (e.g., ".tif", ".png",
  ".jpeg", or ".img"). The default is ".tif" since this is the file
  format used by the utilities in this package. This option is provided
  if chips are generated using another method.

- mode:

  Either "All", "Positive", or "Divided". This should match the setting
  used in the makeChips() function. If the makeChipsMultiClass()
  function was used, this should be set to "All" or left as the default.
  The default is "All".

- shuffle:

  TRUE or FALSE. Whether or not to shuffle the rows in the table. Rows
  can be shuffled to potentially reduced autocorrelation in the data.
  The default is FALSE.

- saveCSV:

  TRUE or FALSE. Whether or not to save the CSV file or just return the
  data frame. If this is set to FALSE then the outCSV parameter is
  ignored and no CSV file is generated. The default is FALSE.

- useEdgeDist:

  TRUE or FALSE. Whether to include a `distPth` column pointing to the
  distance-to-class-boundary grids produced by `makeBoundaryDistance`.
  Supported for all modes. Default is FALSE.

## Value

Data frame with three columns (chpN, chpPth, and mskPth) and,
optionally, a CSV file written to disk. If mode = "Divided", a division
column is added to differentiate "positive" and "background" samples. If
`useEdgeDist = TRUE`, an additional `distPth` column is included.

## Details

This function creates a data frame and, optionally, a CSV file that
lists all of the image chips and associated masks in a directory. Three
columns are produced. The chpN column provides the name of the chip, the
chpPth column provides the path to the chip, and the chpMsk column
provides the path to the associated mask. All paths are relative to the
input folder as opposed to the full file path so that the results can
still be used if the data are copied to a new location on disk or to a
new computer.

If `useEdgeDist = TRUE`, a fourth column `distPth` is added that points
to the corresponding distance-to-class-boundary grid produced by
`makeBoundaryDistance`. All three modes are supported. For
`mode = "All"` or `mode = "Positive"` the paths resolve to
`dist/<name>.tif`; for `mode = "Divided"` they resolve to
`dist/positive/<name>.tif` or `dist/background/<name>.tif` to match the
layout written by `makeBoundaryDistance` when it detects the divided
folder structure.

## Examples

``` r
if (FALSE) { # \dontrun{
chpDF <- makeChipsDF(folder = "PATHT TO CHIPS FOLDER",
                      outCSV = "OUTPUT CSV FILE AND PATH",
                      extension = ".tif",
                      mode="Positive",
                      shuffle=TRUE,
                      saveCSV=TRUE,
                      useEdgeDist=TRUE)
} # }
```
