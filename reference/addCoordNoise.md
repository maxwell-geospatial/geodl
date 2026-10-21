# addCoordNoise

Add random positional noise to point geometries.

## Usage

``` r
addCoordNoise(
  x,
  amount,
  method = c("disc", "uniform", "gaussian"),
  seed = NULL
)
```

## Arguments

- x:

  An `sf` object containing only `POINT` geometries.

- amount:

  Numeric scalar controlling the noise magnitude, in the units of `x`'s
  CRS (e.g. meters for a projected CRS). Interpreted as the disc radius
  for `method = "disc"`, the half-width of the uniform range for
  `method = "uniform"`, and the standard deviation for
  `method = "gaussian"`.

- method:

  Character string specifying the noise model: `"disc"` (default;
  uniform over a circular area), `"uniform"` (independent uniform noise
  on X and Y), or `"gaussian"` (independent Gaussian noise on X and Y).

- seed:

  Optional integer to seed the random number generator for reproducible
  noise. Default `NULL` (no seeding).

## Value

An `sf` object with the same attributes as `x` but with perturbed point
coordinates.

## Details

Perturbs the coordinates of an `sf` POINT object by a random offset
drawn from one of three spatial noise models: a uniform disc (uniform
over the area of a circle of radius `amount`), independent uniform noise
on each axis (uniform over a square of half-width `amount`), or
independent Gaussian noise on each axis with standard deviation
`amount`.
