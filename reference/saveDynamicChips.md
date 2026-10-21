# saveDynamicChips

Save chips meant to be generated dynamically to disk.

## Usage

``` r
saveDynamicChips(
  chipsSF,
  chipSize = 512,
  cellSize = 1,
  outDir,
  mode = "All",
  useExistingDir = FALSE
)
```

## Arguments

- chipsSF:

  output from makeDynamicChipsSF().

- chipSize:

  size of chips to generate. Default is 512 (512-by-512 cells).

- cellSize:

  cell size of input and output data. Default is 1 m.

- outDir:

  full or relative path to output directory. Must include final forward
  slash in path.

- mode:

  either "All", "Positive", or "Divided". If "All", all chips and masks
  are saved. If "Positive", only chips and masks containing positive
  cells are maintained. If "Divided", background-only and
  positive-containing chips are saved but written to separate
  directories. For multiclass, use "All". Default is "All".

- useExistingDir:

  TRUE or FALSE. Whether or not to use a directory that already contains
  chips. Default is FALSE.

## Value

chips and masks written to disk. No R object is returned.

## Details

Save chips defined by makeDynamicChipsSF() to disk. It is not required
to save dynamically generated chips to disk. This is primarily a utility
function.
