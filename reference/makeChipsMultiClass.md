# makeChipsMultiClass

Deprecated. Use
[`makeChips`](https://maxwell-geospatial.github.io/geodl/reference/makeChips.md)
with `mode = "Positive"` instead.

## Usage

``` r
makeChipsMultiClass(
  image,
  mask,
  n_channels = 3,
  size = 256,
  stride_x = 256,
  stride_y = 256,
  outDir,
  useExistingDir = FALSE
)
```

## Arguments

- image:

  SpatRaster object or path to input image.

- mask:

  SpatRaster object or path to single-band mask.

- n_channels:

  Number of channels in the input image. Default is 3.

- size:

  Size of image chips in pixels. Default is 256.

- stride_x:

  Stride in the x (columns) direction. Default is 256.

- stride_y:

  Stride in the y (rows) direction. Default is 256.

- outDir:

  Full or relative path to the output directory. You must include the
  final forward slash (e.g., "C:/data/chips/").

- useExistingDir:

  TRUE or FALSE. Write into an existing directory. Default is FALSE.

## Value

Image and mask files written to disk in TIFF format. No R object is
returned.

## Details

This function is a deprecated wrapper around
[`makeChips`](https://maxwell-geospatial.github.io/geodl/reference/makeChips.md).
It will be removed in a future version. Replace calls of the form:

`makeChipsMultiClass(image, mask, ...)`

with:

`makeChips(image, mask, ..., mode = "Positive")`

## Examples

``` r
if (FALSE) { # \dontrun{
# Deprecated — use makeChips() instead:
makeChips(image = "INPUT IMAGE NAME AND PATH",
          mask = "INPUT RASTER MASK NAME AND PATH",
          n_channels = 3,
          size = 512,
          stride_x = 512,
          stride_y = 512,
          outDir = "DIRECTORY IN WHICH TO SAVE CHIPS",
          mode = "Positive",
          useExistingDir = FALSE)
} # }
```
