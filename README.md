# Species Distribution Modeling of vector species for <em>American Trypanosomiasis</em> using Semi-Structured Deep Distributional Regression
Species Distribution Modeling of Disease Vector Species using Semi-Structured Deep Distributional Regression

**Disclaimer 1**: The analysis requires the package **`deepregression`**, which is supplied in the folders "repo". Note that this package requires **`python`**, **`tensorflow`** and **`tensorflow_probability`** 

**Disclaimer 2**: The analysis is not entirely reproducible as it relies on some confidential data and packages that could not be made public.

## Folder structure
Overview of project files and folders:

- Preliminary notes:
    + The initial steps of this project requiere access to the **`mastergrids`** folder at Maleria Atlas Project and will thus not be fully reproducible

- **`mastergrids`**:
An **`R`** package that facilitates the import of environmental
raster data from the **`mastergrids`** folder at BDI MAP and some
utility functions to transform rasters to data frames and vice versa
(the data import won't work outside of the BDI/without connection to mastergrids). Also contains two functions `grid_to_df` and `df_to_grid` which
convert RasterLayer/RasterBrick object to a data frame and vice versa.

# single-species-models

This folder contains the necessary code for the single-species SDDR models as well as the comparison benchmarks. The included folders contain the necessary `deepregression` repo, the single-species data sets, the Bayesian Optimization results, auxiliary scripts for data pre-processing and the model formulas and the output of the scripts below.

- **`bayes-hopt-single.R`**
This script performs Bayesian Hyperparameter Optimization using Gaussian processes as a surrogate model for all 7 species and 3 predictor types. Subsequently, the optimized model is randomly initialized and trained ten times to produce the final performance results (runs for quite some days!)

- **`benchmarks-single.R`**
This script produces the univariate benchmark results (**`mgcv`** GAM, XGBoost and MaxEnt)

- **`effect-curves-single-species.R`**
This script produces the partial effect curves of the optimized models for the species <em>Panstrongylus megistus</em> (another species can simply be specified at the beginning). Output is in folder plot-results

- **`performance-results-single-species.R`**
This script takes the **`ParBayesianOptimization`** objects from the folder bayesian-optimization and trains SDDR models for each species and predictor type ten times using random weight initializations to produce the final performance results. Output is in folder performance-results

- **`plots-single-species.R`**
This script produces the predictive maps obtained by SDDR (DNN-only predictor type). This **script cannot be run** without the environmental grid data not included here.


# pooled-models 

