# Species Distribution Modeling of vector species for <em>American Trypanosomiasis</em> using Semi-Structured Deep Distributional Regression
Species Distribution Modeling of Disease Vector Species using Semi-Structured Deep Distributional Regression

**Disclaimer 1**: The analysis requires the **`R`** package **`deepregression`**, which is supplied in the folders "repo". Note that this package requires **`python`**, **`tensorflow`** and **`tensorflow_probability`** and other **`R`** dependencies. See the README files in the "deepregression-master" folder within "repo". Further note that the single-species models use another version of `deepregression` than the pooled and multi-species models.

**Disclaimer 2**: The analysis is not entirely reproducible as it relies on some confidential data and packages that could not be made public. Scripts that cannot be run are **`plots-single-species.R`** for the single-species predictive distribution plots (plots are included in `single-species-models/plot-results/sdm-plots`) and **`full-model-datagen.R`** (resulting data set `full-model-list.Rds` is contained in `pooled-models/data` and `multi-species-models/data`). Both scripts require the **`mastergrids`** package from the Malaria Atlas Project and will thus not be fully reproducible. **`mastergrids`** is an **`R`** package that facilitates the import of environmental raster data from the **`mastergrids`** folder at BDI MAP and some utility functions to transform rasters to data frames and vice versa. Also contains two functions `grid_to_df` and `df_to_grid` which convert RasterLayer/RasterBrick object to a data frame and vice versa.

# Folder structure
Overview of project files and folders:

## single-species-models

This folder contains the necessary code for the single-species SDDR models as well as the comparison benchmarks. Nested folders contain the necessary `deepregression` repo, the single-species data sets, the pre-computed Bayesian Optimization results, auxiliary scripts for data pre-processing and the model formulas, as well as the output of the scripts below.

- **`bayes-hopt-single.R`**
This script performs Bayesian Hyperparameter Optimization using Gaussian processes as a surrogate model for all 7 species and 3 predictor types. Subsequently, the optimized model is randomly initialized and trained ten times to produce the final performance results (runs for quite some days!).

- **`benchmarks-single.R`**
This script produces the univariate benchmark results (`mgcv` GAM, XGBoost and MaxEnt).

- **`effect-curves-single-species.R`**
This script produces the partial effect curves of the optimized models for the species <em>Panstrongylus megistus</em> (another species can simply be specified at the beginning). Output is in folder `plot-results`.

- **`performance-results-single-species.R`**
This script takes the pre-computed `ParBayesianOptimization` objects from the folder bayesian-optimization and trains SDDR models for each species and predictor type ten times using random weight initializations to produce the final performance results. Output is in folder performance-results.

- **`plots-single-species.R`**
This script produces the predictive maps obtained by SDDR (DNN-only predictor type). This **script cannot be run** without the environmental grid data not included here.

## pooled-models 

This folder contains the necessary code for the pooled SDDR models. Nested folders contain the necessary `deepregression` repo, the pooled data set, Bayesian Optimization results, auxiliary scripts for data pre-processing and the model formulas, and the output of the scripts below.

- **`bayes-hopt-pooled.R`**
This script performs Bayesian Hyperparameter Optimization for all three predictor types. Subsequently, the models are estimated ten times to produce the final results. The folder "bayesian-optimization-results" contains the resulting `ParBayesianOptimization` objects and "performance-results" the respective AUC and Brier scores. Also runs for several days.

- **`full-model-datagen.R`**
This script takes the raw data (**not included here**) and produces the pooled and multivariate data sets (`full-model-list.Rds` in data folder) and generates spatially decorrelated cross-validation folds using  `blockCV`. Although the raw data is not included, the resulting data set is included.

## multi-species-models

This folder contains the necessary code for the multi-species SDDR approaches. Nested folders contain the necessary `deepregression` repo, the multivariate data set, pre-computed bayesian optimization results, auxiliary scripts for data pre-processing and the model formulas, and the output of the scripts below.

- **`bayes-hopt-multi-class.R`** 
This script performs Bayesian Hyperparameter Optimization for all three predictor types in the multi-class modeling approach using a Multinoulli distribution to model the label powerset of the response labels. Subsequently, the models are estimated ten times to produce the final results. The folder `multi-class-model` contains the resulting `ParBayesianOptimization` objects and performance results, i.e. the respective AUC and Brier scores. Also runs for several days.

- **`bayes-hopt-multivariate.R`** 
Same as for `bayes-hopt-multi-class`, only that the multivariate data are modeled using seven independent Bernoulli distributions. Results are contained in `multivariate-model`

- **`multi-mars.R`**
This script computes the multivariate benchmark model (MMARS: multi-species multivariate adaptive regression splines). Results are contained in folder `mmars-model`.




