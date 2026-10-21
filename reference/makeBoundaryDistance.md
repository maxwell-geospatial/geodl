# makeBoundaryDistance

Generate distance-to-class-boundary grids from mask chips

## Usage

``` r
makeBoundaryDistance(chipsDir, noDataValue = 0)
```

## Arguments

- chipsDir:

  Full or relative path to the chips directory that contains a "masks"
  subfolder. This is the same directory that was used as the `outDir`
  argument to `makeChips`. Do not include a trailing slash.

- noDataValue:

  Numeric scalar. The value written to every pixel of a distance grid
  when the corresponding mask chip contains no class boundaries (i.e.
  the chip belongs to a single class throughout). Default is `0`.

## Value

Distance-to-boundary raster files written to disk in TIFF format inside
a "dist" subfolder (flat layout) or "dist/positive/" and
"dist/background/" subfolders (divided layout). No R object is returned.

## Details

This function reads raster mask chips from the "masks" subfolder of a
chips directory created by `makeChips` and generates a corresponding
distance surface for each mask. Each output raster encodes, for every
pixel, the distance to the nearest class boundary. A class boundary is
any pixel that is adjacent (in an 8-connected neighborhood) to at least
one pixel belonging to a different class. Boundary pixels themselves
receive a value of 0; all other pixels receive the Euclidean distance
(in pixel units, unless the raster has a projected coordinate system, in
which case map units are used) to the nearest boundary pixel. Output
grids are written to a "dist" subfolder created inside the provided
chips directory. Output files are named identically to their
corresponding masks and are saved in TIFF format.

The function automatically detects whether the chips directory uses a
flat layout (`masks/`) produced by `makeChips` with `mode = "All"` or
`mode = "Positive"`, or a divided layout (`masks/positive/` and
`masks/background/`) produced with `mode = "Divided"`. For the divided
layout, distance grids are written to `dist/positive/` and
`dist/background/` respectively.

## Examples

``` r
if (FALSE) { # \dontrun{
makeBoundaryDistance(chipsDir = "C:/data/chips")

# Fill single-class chips with a large sentinel instead of 0
makeBoundaryDistance(chipsDir = "C:/data/chips", noDataValue = 999)
} # }
```
