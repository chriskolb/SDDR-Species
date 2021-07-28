# Species Distribution Modeling of vector species for <em>American Trypanosomiasis</em> using Semi-Structured Deep Distributional Regression

This repository contains code and data for the analysis of multiple <em>triatomine</em> species in South and Middle America that act as vector species for the parasitic protozoan <em>Trypanosoma cruzi</em>, a pathogen responsible for one of the most burdensome neglected tropical diseases, <em>American Trypanosomiasis</em> or Chagas disease. 

**Disclaimer 1**: The analysis requires deprecated versions of the **`R`** package **`deepregression`**, which are supplied in the folders named "repo". Note that the repo/code requires **`python`** (3.7.10), **`tensorflow`** (2.0.0) and **`tensorflow_probability`** (0.8.0) installed in a conda environment named **`r-reticulate`** (see `single-species-models/repo/Miniconda setup.txt` for help), as well as various **`R`** dependencies. See the README files in the "deepregression-master" folder within "repo" for details on the **`R`** dependencies. Further note that the single-species models use another version of `deepregression` than the pooled and multi-species models. Figuring out the right set-up and dependencies to run the code can be tedious.

**Disclaimer 2**: The analysis is not entirely reproducible as it relies on some confidential data and packages that could not be made public. Scripts that cannot be run are **`plots-single-species.R`** for the single-species predictive distribution plots (plots are included in `single-species-models/plot-results/sdm-plots`) and **`full-model-datagen.R`** (resulting data set `full-model-list.Rds` is contained in `pooled-models/data` and `multi-species-models/data`). Both scripts either require the **`mastergrids`** package from the Malaria Atlas Project or the raw environmental grid data (too big to be included here) and will thus not be fully reproducible. **`mastergrids`** is an **`R`** package that facilitates the import of environmental raster data from the **`mastergrids`** folder at BDI MAP (University of Oxford) and some utility functions to transform rasters to data frames and vice versa.

# Folder structure
Overview of project files and folders:

## single-species-models

This folder contains the necessary code for the single-species SDDR models as well as the comparison benchmarks. Nested folders contain the necessary `deepregression` repo, the single-species data sets, the pre-computed Bayesian Optimization results, auxiliary scripts for data pre-processing and the model formulas, as well as the output of the scripts below.

- **`performance-results-single-species.R`**
This script takes the pre-computed optimal hyperparameters in the `ParBayesianOptimization` objects from the folder `bayesian-optimization` and trains SDDR models for each species and predictor type ten times using random weight initializations to produce the final performance results. Output is in folder `performance-results`.

- **`benchmarks-single.R`**
This script produces the univariate benchmark results (`mgcv` GAM, XGBoost and MaxEnt).

- **`effect-curves-single-species.R`**
This script produces the partial effect curves of the optimized models for the species <em>Panstrongylus megistus</em> (another species can simply be specified at the beginning). Output is in folder `plot-results`.


- **`plots-single-species.R`**
This script produces the predictive maps obtained by SDDR (DNN-only predictor type). This **script cannot be run** without the environmental grid data not included here.

- **`bayes-hopt-single.R`**
This script performs Bayesian Hyperparameter Optimization using Gaussian processes as a surrogate model for all 7 species and 3 predictor types. Subsequently, the optimized model is randomly initialized and trained ten times (for each species x predictor combination) to produce the final averaged performance results (runs for 7+ days!). Note that the hyperparameter ranges in this script are more general than the bounds used for the single-species models in the thesis, e.g., allowing for more than one hidden layer. Results will thus differ. To re-run the thesis AUC and Brier score results with pre-computed `ParBayesianOptimization` objects (same as results in thesis), use **`effect-curves-single-species.R`**.

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

## Session Info (might help for set-up)




```r
> py_config()
python:         C:/Users/kolchris/AppData/Local/r-miniconda/envs/r-reticulate/python.exe
libpython:      C:/Users/kolchris/AppData/Local/r-miniconda/envs/r-reticulate/python37.dll
pythonhome:     C:/Users/kolchris/AppData/Local/r-miniconda/envs/r-reticulate
version:        3.7.10 (default, Feb 26 2021, 13:06:18) [MSC v.1916 64 bit (AMD64)]
Architecture:   64bit
numpy:          C:/Users/kolchris/AppData/Local/r-miniconda/envs/r-reticulate/Lib/site-packages/numpy
numpy_version:  1.20.2
tensorflow_probability:C:\Users\kolchris\AppData\Local\R-MINI~1\envs\R-RETI~1\lib\site-packages\tensorflow_probability\__init__.p

NOTE: Python version was forced by use_python function
> tf_version()
[1] ‘2.0’
> tfp_version()
[1] ‘0.8’
> sessionInfo()
R version 4.0.3 (2020-10-10)
Platform: x86_64-w64-mingw32/x64 (64-bit)
Running under: Windows Server >= 2012 x64 (build 9200)

Matrix products: default

locale:
[1] LC_COLLATE=German_Germany.1252  LC_CTYPE=German_Germany.1252    LC_MONETARY=German_Germany.1252
[4] LC_NUMERIC=C                    LC_TIME=German_Germany.1252    

attached base packages:
[1] grid      parallel  stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
 [1] deepregression_0.0.0.9000     UBL_0.0.7                     randomForest_4.6-14          
 [4] gstat_2.0-7                   MBA_0.0-9                     earth_5.3.1                  
 [7] plotmo_3.6.1                  TeachingDemos_2.12            plotrix_3.8-1                
[10] DiceKriging_1.6.0             classInt_0.4-3                stars_0.5-3                  
[13] abind_1.4-5                   pals_1.7                      automap_1.0-14               
[16] rsample_0.1.0                 furrr_0.2.3                   future_1.21.0                
[19] fields_12.5                   viridis_0.6.1                 viridisLite_0.4.0            
[22] spam_2.7-0                    dotCall64_1.0-1               blockCV_2.1.4                
[25] pammtools_0.5.7               spatstat_2.2-0                spatstat.linnet_2.3-0        
[28] spatstat.core_2.3-0           rpart_4.1-15                  spatstat.geom_2.2-2          
[31] spatstat.data_2.1-0           geosphere_1.5-10              pbapply_1.4-3                
[34] dismo_1.3-3                   tibble_3.1.2                  tidyr_1.1.3                  
[37] rgeos_0.5-5                   sf_1.0-1                      magrittr_2.0.1               
[40] purrr_0.3.4                   tmap_3.3-2                    stringr_1.4.0                
[43] tmaptools_3.1-1               raster_3.4-13                 sp_1.4-5                     
[46] googlesheets_0.3.0            ggvis_0.4.7                   Hmisc_4.5-0                  
[49] Formula_1.2-4                 survival_3.2-7                testthat_3.0.4               
[52] rstudioapi_0.13               maxnet_0.1.4                  rlang_0.4.11                 
[55] scoring_0.6                   muStat_1.7.0                  doParallel_1.0.16            
[58] iterators_1.0.13              foreach_1.5.1                 ParBayesianOptimization_1.2.4
[61] yardstick_0.0.8               recipes_0.1.16                xgboost_1.4.1.1              
[64] caret_6.0-88                  ggplot2_3.3.5                 lattice_0.20-41              
[67] MLmetrics_1.1.1               DescTools_0.99.42             Metrics_0.1.4                
[70] tfprobability_0.12.0.0        tensorflow_2.5.0              reticulate_1.20              
[73] mgcv_1.8-33                   nlme_3.1-149                  keras_2.4.0                  
[76] dplyr_1.0.7                   Matrix_1.2-18                 devtools_2.4.2               
[79] usethis_2.0.1                

loaded via a namespace (and not attached):
  [1] rappdirs_0.3.3        spacetime_1.2-5       ModelMetrics_1.2.2.2  intervals_0.15.2      knitr_1.33           
  [6] data.table_1.14.0     generics_0.1.0        leaflet_2.0.4.1       timereg_2.0.0         cowplot_1.1.1        
 [11] callr_3.7.0           proxy_0.4-26          lubridate_1.7.10      httpuv_1.6.1          assertthat_0.2.1     
 [16] gower_0.2.2           xfun_0.24             hms_1.1.0             promises_1.2.0.1      fansi_0.5.0          
 [21] progress_1.2.2        readxl_1.3.1          DBI_1.1.1             htmlwidgets_1.5.3     reshape_0.8.8        
 [26] stats4_4.0.3          ellipsis_0.3.2        crosstalk_1.1.1       ggpubr_0.4.0          backports_1.2.1      
 [31] deldir_0.2-10         vctrs_0.3.8           remotes_2.4.0         cachem_1.0.5          withr_2.4.2          
 [36] checkmate_2.0.0       xts_0.12.1            prettyunits_1.1.1     goftest_1.2-2         cluster_2.1.0        
 [41] pacman_0.5.1          lazyeval_0.2.2        crayon_1.4.1          glmnet_4.1-2          pkgconfig_2.0.3      
 [46] labeling_0.4.2        units_0.7-2           pkgload_1.2.1         nnet_7.3-14           globals_0.14.0       
 [51] lifecycle_1.0.0       dbscan_1.1-8          dichromat_2.0-0       cellranger_1.1.0      rprojroot_2.0.2      
 [56] polyclip_1.10-0       carData_3.0-4         zoo_1.8-9             boot_1.3-25           base64enc_0.1-3      
 [61] whisker_0.4           processx_3.5.2        png_0.1-7             rootSolve_1.8.2.1     KernSmooth_2.23-17   
 [66] pROC_1.17.0.1         shape_1.4.6           parallelly_1.26.1     jpeg_0.1-8.1          rstatix_0.7.0        
 [71] ggsignif_0.6.2        scales_1.1.1          memoise_2.0.0         plyr_1.8.6            leafsync_0.1.0       
 [76] compiler_4.0.3        RColorBrewer_1.1-2    cli_3.0.0             listenv_0.8.0         ps_1.6.0             
 [81] htmlTable_2.2.1       MASS_7.3-53           tidyselect_1.1.1      stringi_1.6.2         forcats_0.5.1        
 [86] latticeExtra_0.6-29   tools_4.0.3           lmom_2.8              rio_0.5.27            foreign_0.8-80       
 [91] gridExtra_2.3         gld_2.6.2             prodlim_2019.11.13    farver_2.1.0          pec_2020.11.17       
 [96] digest_0.6.27         FNN_1.1.3             shiny_1.6.0           lava_1.6.9            Rcpp_1.0.6           
[101] car_3.0-11            broom_0.7.8           lwgeom_0.2-6          later_1.2.0           colorspace_2.0-2     
[106] XML_3.99-0.6          fs_1.5.0              tensor_1.5            splines_4.0.3         expm_0.999-6         
[111] spatstat.utils_2.2-0  Exact_2.1             mapproj_1.2.7         sessioninfo_1.1.1     xtable_1.8-4         
[116] jsonlite_1.7.2        leafem_0.1.6          timeDate_3043.102     zeallot_0.1.0         ipred_0.9-11         
[121] R6_2.5.0              lhs_1.1.1             pillar_1.6.1          htmltools_0.5.1.1     mime_0.11            
[126] glue_1.4.2            fastmap_1.1.0         class_7.3-17          codetools_0.2-16      maps_3.3.0           
[131] pkgbuild_1.2.0        mvtnorm_1.1-2         utf8_1.2.1            spatstat.sparse_2.0-0 numDeriv_2016.8-1.1  
[136] curl_4.3.2            tfruns_1.5.0          zip_2.2.0             openxlsx_4.2.4        desc_1.3.0           
[141] munsell_0.5.0         e1071_1.7-7           haven_2.4.1           reshape2_1.4.4        gtable_0.3.0 
```

Moreover, the following packages are installed in the anaconda environment `r-reticulate`

```

(base) C:\>conda activate r-reticulate

(r-reticulate) C:\>pip list
Package                Version
---------------------- -------------------
absl-py                0.13.0
astor                  0.8.1
cached-property        1.5.2
cachetools             4.2.2
certifi                2021.5.30
chardet                4.0.0
cloudpickle            1.1.1
decorator              5.0.9
dm-tree                0.1.6
gast                   0.2.2
google-auth            1.32.1
google-auth-oauthlib   0.4.4
google-pasta           0.2.0
grpcio                 1.38.1
h5py                   3.3.0
idna                   2.10
importlib-metadata     4.6.1
Keras-Applications     1.0.8
Keras-Preprocessing    1.1.2
Markdown               3.3.4
mkl-fft                1.3.0
mkl-random             1.2.1
mkl-service            2.3.0
numpy                  1.20.2
oauthlib               3.1.1
opt-einsum             3.3.0
pip                    21.1.3
protobuf               3.17.3
pyasn1                 0.4.8
pyasn1-modules         0.2.8
requests               2.25.1
requests-oauthlib      1.3.0
rsa                    4.7.2
setuptools             52.0.0.post20210125
six                    1.16.0
tensorboard            2.0.2
tensorflow             2.0.0
tensorflow-estimator   2.0.1
tensorflow-probability 0.8.0
termcolor              1.1.0
typing-extensions      3.10.0.0
urllib3                1.26.6
Werkzeug               2.0.1
wheel                  0.36.2
wincertstore           0.2
wrapt                  1.12.1
zipp                   3.5.0
```

