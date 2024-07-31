## R CMD check results

0 errors | 0 warnings | 4 note

* This is a new release.
* Vignettes take a long time to run and require GPU computation, so were pre-rendered
* Some example code is not run because the function is designed to read and write files from a disk

### Note 1

checking CRAN incoming feasibility ... NOTE
Maintainer: 'Aaron Maxwell <Aaron.Maxwell@mail.wvu.edu>'
New submission
Size of tarball: 5820884 bytes

**Response**: Is the issue here that the size is too large?

### Note 2

checking dependencies in R code ... NOTE
Namespace in Imports field not imported from: 'luz'
All declared Imports should be used.

**Response**: Unsure what the issue is here. The 'luz' package is used and specified with luz::

### Note 3

checking files in 'vignettes' ... NOTE
The following directory looks like a leftover from 'knitr': 'figure'
Please remove from your package.

**Response**: Including some pre-rendered figures for vignettes. 

### Note 5
Examples with CPU (user + system) or elapsed time > 5s
user system elapsed
defineMobileUNet 3.91   1.13    1.05

**Response**: Do we need to reduce run time or is this acceptable?


### Abbreviations in DESCRIPTION

Based on a prior comment during CRAN review, we have defined all abbreviations in the DESCRIPTION file. Note that UNet is not an abbreviation. 

### Citations in DESCRIPTION

The citation in the description was changed to the form Authr (YEAR) <doi>.

### spatialPredict() Error

The error in the spatialPredict() function example was corrected. This was a typo. 

### print() statements

All print statements have have converted to messages()

### dontrun{} vs. donttest{}

We have edited the examples so that all examples that can run on the test server are executed. 
For examples that require reading and writing data from disk, we use dontrun{}.
For examples that need the torch C++ backend, we use donttest{}.

