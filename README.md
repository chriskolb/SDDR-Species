# Species Distribution Modeling of vector species for <em>American Trypanosomiasis</em> using Semi-Structured Deep Distributional Regression

This repository contains code and data for the analysis of multiple <em>triatomine</em> species in South and Middle America that act as vector species for the parasitic protozoan <em>Trypanosoma cruzi</em>, a pathogen responsible for one of the most burdensome neglected tropical diseases, <em>American Trypanosomiasis</em> or Chagas disease. 

**Disclaimer 1**: The analysis requires the **`R`** package **`deepregression`**, which is supplied in the folders "repo". Note that these repo versions require **`python`** (3.7.10), **`tensorflow`** (2.0.0) and **`tensorflow_probability`** (0.8.0) installed in a conda environment named **`r-reticulate`**, as well as various **`R`** dependencies. See the README files in the "deepregression-master" folder within "repo". Further note that the single-species models use another version of `deepregression` than the pooled and multi-species models.

**Disclaimer 2**: The analysis is not entirely reproducible as it relies on some confidential data and packages that could not be made public. Scripts that cannot be run are **`plots-single-species.R`** for the single-species predictive distribution plots (plots are included in `single-species-models/plot-results/sdm-plots`) and **`full-model-datagen.R`** (resulting data set `full-model-list.Rds` is contained in `pooled-models/data` and `multi-species-models/data`). Both scripts either require the **`mastergrids`** package from the Malaria Atlas Project or the raw environmental grid data (too big to be included here) and will thus not be fully reproducible. **`mastergrids`** is an **`R`** package that facilitates the import of environmental raster data from the **`mastergrids`** folder at BDI MAP (University of Oxford) and some utility functions to transform rasters to data frames and vice versa.

# Folder structure
Overview of project files and folders:

## single-species-models

This folder contains the necessary code for the single-species SDDR models as well as the comparison benchmarks. Nested folders contain the necessary `deepregression` repo, the single-species data sets, the pre-computed Bayesian Optimization results, auxiliary scripts for data pre-processing and the model formulas, as well as the output of the scripts below.

- **`performance-results-single-species.R`**
This script takes the pre-computed `ParBayesianOptimization` objects from the folder `bayesian-optimization` and trains SDDR models for each species and predictor type ten times using random weight initializations to produce the final performance results. Output is in folder `performance-results`.

- **`benchmarks-single.R`**
This script produces the univariate benchmark results (`mgcv` GAM, XGBoost and MaxEnt).

- **`effect-curves-single-species.R`**
This script produces the partial effect curves of the optimized models for the species <em>Panstrongylus megistus</em> (another species can simply be specified at the beginning). Output is in folder `plot-results`.


- **`plots-single-species.R`**
This script produces the predictive maps obtained by SDDR (DNN-only predictor type). This **script cannot be run** without the environmental grid data not included here.

- **`bayes-hopt-single.R`**
This script performs Bayesian Hyperparameter Optimization using Gaussian processes as a surrogate model for all 7 species and 3 predictor types. Subsequently, the optimized model is randomly initialized and trained ten times (for each species x predictor combination) to produce the final averaged performance results (runs for 7+ days!). Note that the hyperparameter ranges in this script are more general than the bounds used for the single-species models in the thesis, e.g., allowing for more than one hidden layer. Results will thus differ. To re-run the AUC and Brier score results with the pre-computed `ParBayesianOptimization` objects (same as results in thesis), use **`effect-curves-single-species.R`**.

## pooled-models 

This folder contains the necessary code for the pooled SDDR models. Nested folders contain the necessary `deepregression` repo, the pooled data set, Bayesian Optimization results, auxiliary scripts for data pre-processing and the model formulas, and the output of the scripts below.

- **`bayes-hopt-pooled.R`**
This script performs Bayesian Hyperparameter Optimization for all three predictor types. Subsequently, the models are estimated ten times to produce the final results. The folder `bayesian-optimization-results` contains the resulting `ParBayesianOptimization` objects and the folder `performance-results` the respective AUC and Brier scores. Also runs for several days.

- **`full-model-datagen.R`**
This script takes the raw species occurrence and environmental grid data (**not included here**) and produces the pooled and multivariate data sets (`full-model-list.Rds` in `data` folder of pooled-models and multi-species-models) and generates spatially decorrelated cross-validation folds using  `blockCV`. Although the raw data is not included, the resulting data set is included.

## multi-species-models

This folder contains the necessary code for the multi-species SDDR approaches. Nested folders contain the necessary `deepregression` repo, the multivariate data set, pre-computed bayesian optimization results, auxiliary scripts for data pre-processing and the model formulas, and the output of the scripts below.

- **`bayes-hopt-multi-class.R`** 
This script performs Bayesian Hyperparameter Optimization for all three predictor types in the multi-class modeling approach using a Multinoulli distribution to model the label powerset of the response labels. Subsequently, the models are estimated ten times to produce the final results. The folder `multi-class-model` contains the resulting `ParBayesianOptimization` objects and performance results, i.e. the respective AUC and Brier scores. Also runs for several days.

- **`bayes-hopt-multivariate.R`** 
Same as for `bayes-hopt-multi-class`, only that the multivariate data are modeled using seven independent Bernoulli distributions. Results are contained in `multivariate-model`

- **`multi-mars.R`**
This script computes the multivariate benchmark model (MMARS: multi-species multivariate adaptive regression splines). Results are contained in folder `mmars-model`.




