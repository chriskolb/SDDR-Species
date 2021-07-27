################################################################################################
# Species Distribution Maps per Species for optimized SDDR models and mgcv GAMs
################################################################################################

rm(list=ls())

# script parameters: 
# species, predictor type

species.list <- c("tinfestans", "tdimidiata", "pmegistus", "tbrasiliensis", "tsordida", "tpseudomaculata", "tbarberi")
spec = "tdimidiata"

# define SDDR predictor list (plots are only created for smooth+deep predictor)
preds.list <- c("smooth","deep","smooth-deep")
pr = "smooth"

# fixed number of epochs for optimization
epochs.cv = 100

# directories
main.path <- dirname(rstudioapi::getSourceEditorContext()$path)
repo.path <- file.path(main.path, "repo", "deepregression-master")
temp.data <- file.path(main.path, "temp")
plot.data <- temp.data


#################################################################################################
## set up workspace
#################################################################################################

library(devtools)

# load deepregression and spatial packages
## install.packages("pacman")
pacman::p_load(Matrix, dplyr, keras, mgcv, reticulate, tensorflow, tfprobability, 
               Metrics, DescTools, MLmetrics, caret, xgboost, recipes, yardstick
               ,ParBayesianOptimization, doParallel, muStat, scoring, blockCV, checkmate, 
               furrr, tmap, rgeos, rstudioapi
               )

# force conda environment
use_condaenv("r-reticulate", required = T)

# load deepregression package
load_all(repo.path)

# load mastergrids package
load_all(file.path(main.path, "repo", "mastergrids"))

# seed
seed = 42
tf$random$set_seed(seed = seed)
set.seed(seed)

# timestap to name all output files
timestamp <- format(Sys.time(), "%Y-%m-%d-%H%M")

################################################################################################
## load and prepare data #######################################################################
################################################################################################

for (spec in species.list){
    
    # tinfestans, tdimidiata, pmegistus, tbrasiliensis, tsordida, tpseudomaculata, tbarberi
    species <- spec #"tinfestans"
    
    # choose predictor type: smooth, smooth-deep, deep
    pred.type = pr #"smooth"
    
    # load data
    data <- readRDS(file.path(main.path, "data", paste0(species, ".rds")))
    
    # select minimum unique covariate values for construction of smooth terms
    # tinfestans: 40, tdimidiata: 35, pmegistus: 20 (40?), 
    # tbrasiliensis: 40, tsordida: 25, tpseudomaculata: 20, tbarberi: 30
    
    if (species == "tinfestans"){
      min.unique <- 40
    } else if (species == "tdimidiata"){
      min.unique <- 40 # 35 also works for smooth predictor weirdly
    } else if (species == "pmegistus"){
      min.unique <- 35 
    } else if (species == "tbrasiliensis"){
      min.unique <- 40
    } else if (species == "tsordida"){
      min.unique <- 25 
    } else if (species == "tpseudomaculata"){
      min.unique <- 20
    } else {
      # tbarberi
      min.unique <- 30
    }
    
    # run data pre-processing script
    source(file = file.path(main.path, "auxiliary-scripts", "data-preprocessing.R"))
    
    # combine train and test data for prediction maps
    y.comb = c(y.train, y.test)
    x.comb = rbind(x.train.s, x.test.s)
    
    # define predictor formulae: smooth, smooth.deep, deep
    source(file = file.path(main.path, "auxiliary-scripts","formulae.R"))
    
    # model formula
    if (pred.type == "smooth"){
      form.mod <- form.smooth
    } else if (pred.type == "smooth-deep"){
      form.mod <- form.smooth.deep
    } else{
      form.mod <- form.deep
    }
    form.mod
    
    # set hyperparameters that are not included in bayesian GP optimization
    batch.size.nn = 20
    lambda.l1 = 0
    batchnorm = 1 # 1 = yes, 2 = no
    
    # initialize arguments for testing purposes
    hidden.units.num = 3 # 1=32, 2=64, 3=128, 4=200
    hidden.layers = 2
    dropout.nn = 0.2
    log.lr.nn = -3
    log.decay.nn = -4
    act.num = 1 # 1=relu, 2=tanh
    init.num = 2 # 1=glorot_normal, 2=he_normal
    if(pred.type == "deep"){df.smooth = NULL} else{df.smooth = 9}
    
    
    ################################################################################################
    ### prepare spatial data (grids)
    ################################################################################################
    
    # brick of environmental covariates (5x5 km grid)
    env_grids <- raster::brick(file.path(main.path, "data", "raw-data", "env_grids.gri"))
    
    ## country borders within endemic zone
    countries <- readRDS(file.path(main.path, "data", "raw-data", "countries.Rds"))
    
    # folds containing species data
    folds <- readRDS(file.path(main.path, "data", "raw-data", "folds_list_vector_presence.Rds"))
    names(folds)
    
    # full species names, needed to access the folds object accordingly
    full.names.list <- c("Triatoma infestans", "Triatoma dimidiata", "Panstrongylus megistus",
                          "Triatoma brasiliensis", "Triatoma sordida", "Triatoma pseudomaculata",
                          "Triatoma barberi")
    
    # convert abbreviated species names to full names via function
    full.name.fct <- function(short.name) {
      index <- which(species.list == paste0(species))
      output <- full.names.list[index]
      return(output)
    }
    
    # execute function
    full.name <- full.name.fct(species)
    
    # full spatial data for species at hand
    spec.data <- folds[[full.name]] #Triatoma pseudomaculata etc
    str(spec.data, 1)
    spec.countries <- countries %>% raster::crop(spec.data$extent)
    
    rm(folds, env_grids)
    

    ################################################################################################
    ### prepare DNN with optimized hyperparameters #################################################
    ################################################################################################
    
    # load bayesian hyperparameter optimization object (ParBayesOptimization)
    bayes.opt.result <- readRDS(file.path(main.path, "bayesian-optimization", species, "results", 
                                          paste0("bayesopt-results-", species, "-", pred.type, "-fixed", ".RDS")))
    
    # print optimal hyperparameters
    best.pars <- getBestPars(bayes.opt.result)
    best.pars
    best.pars.bopt <- best.pars
    
    # define learning rate and decay parameters of optimizer
    lr.nn <- best.pars.bopt$lr.nn
    decay.nn <- best.pars.bopt$decay.nn
    
    # set batchnorm to 1 (yes) per default
    batchnorm = 1 # 1 = yes, 2 = no
    
    # hidden units in first hidden layer
    hidden.units.list <- c(32, 64, 128, 200)
    #hidden.units1 <- hidden.units.list[best.pars.bopt$hidden.units.num]
    hidden.units1 <- 128
    
    # activation function
    activations <- list("relu", "tanh")
    act.nn  <- activations[[1]]  # act.num
    
    # initializers
    initializers   <- list(
      "initializer_glorot_normal()",
      "initializer_he_normal()"
    )
    init.nn <- initializers[[2]]
    
    
    ## define NN architecture as function of hyperparameters
    # start.block + building.block.1 + ... + building.block.i + end.block
    
    # start block  
    start.block <- paste0(
      "function(x) x %>%
    layer_dense(units = ", hidden.units1, ", activation = '", act.nn, "', 
                kernel_initializer = ", init.nn, "
                ) %>% \n",
      ifelse(batchnorm==1, "    layer_batch_normalization() %>% \n", ""),
      "    layer_dropout(rate = ", best.pars.bopt$dropout.nn, ", seed = seed) %>% \n"
    )  
    
    # middle blocks (hidden layers 2-x)
    building.block <- paste0(
      "    layer_dense(units = ", 0.5*hidden.units1, ", activation = '", act.nn, "', 
                kernel_initializer = ", init.nn, "
                ) %>% \n",
      ifelse(batchnorm==1, "    layer_batch_normalization() %>% \n", ""), 
      "    layer_dropout(rate = ", best.pars.bopt$dropout.nn, ", seed = seed) %>% \n"
    )  
    nn.body <- paste0(strrep(times = (best.pars.bopt$hidden.layers-1), building.block))
    nn.body 
    
    # output layer
    end.block <- paste0(
      "    layer_dense(units = 1, activation = 'linear', 
                kernel_initializer = ", init.nn, "
                )"
    )
    
    nn_deep = eval(parse(text = paste0(
      start.block, 
      nn.body,
      end.block
    )))
    
    nn_deep 
    
    ################################################################################################
    ### estimate SDDR model with optimized hyperparameters for prediction map ######################
    ################################################################################################
    
    # set seed
    tf$random$set_seed(seed = seed)
    set.seed(seed)

      # define deepregression object
      mod.post.bopt <-  deepregression(y = y.comb,
                                       list_of_formulae = list(logit = form.mod),
                                       list_of_deep_models = list(nn_deep = nn_deep),
                                       data = x.comb,
                                       family = "bernoulli",
                                       lambda_lasso = lambda.l1,
                                       optimizer = optimizer_adam(
                                         decay = decay.nn, 
                                         lr = lr.nn 
                                       ),
                                       #orthog_type = "manual",
                                       #zero_constraint_for_smooths = TRUE,
                                       #absorb_cons = FALSE, 
                                       df = best.pars.bopt$df.smooth
                                       #,validation_data = list(x.test.s, y.test)
                                       #,cv_folds = spat.cv.indices, 
                                       #,monitor_metric = c("accuracy")
      )
      
      # fit model with optimal number of epochs
      hist.post.bopt <- mod.post.bopt %>% fit(
        epochs = epochs.cv
        #,view_metrics = TRUE 
        ,verbose = TRUE
        ,batch_size = batch.size.nn
        ,auc_callback = TRUE
        ,val_data = list(x.test.s, y.test)
        ,validation_split = 0
      )  
      
      # predict on test set
      pred <- mod.post.bopt %>% predict(x.test.s)
      hist(pred)
      summary(pred)
      
      # evaluate prediction with AUC
      auc.test <- MLmetrics::AUC(y_pred = pred, y_true = y.test)
      cat("\nTest AUC for species", spec, "and predictor", pred.type, 
          "is =", auc.test, "\n")
      
      # evaluate prediction with Brier score
      brier.test <- BrierScore(resp = y.test, pred = pred)
      cat("\nTest Brier score for species", spec, "and predictor", pred.type, 
          " is ", brier.test, "\n\n")
      
      
    # create prediction map 
      gc()
      memory.limit(9999999999)
      env_grids <- raster::brick(file.path(main.path, "data", "raw-data", "env_grids.gri"))
      # grid_to_df is mastergrids package function (Bender et al)
      # create data frame with 1 observation per 5x5km square
      grid.pred <- grid_to_df(env_grids, spec.data$hull, spec.data$hull)
      # note: when fitted with bam, use discrete = FALSE within predict for final
      # prediction (takes longer but finer resolution)
      # predict(mod, newdata = pred_df, type = "response")
      
      # bake grid df according to train data parameters (standardize data wrt train set)
      grid.pred.s <- bake(rec_obj, new_data = grid.pred)
      #rm(grid.pred)
      
      # predict response on grid df (takes 10-20 min)
      grid.pred$prediction <- mod.post.bopt %>% predict(grid.pred.s) 
      
      # tranform data frame back to grid for visualization
      grid <- df_to_grid(grid.pred, env_grids[[1]], "prediction")
      
      # initialize  plot
      png(file = file.path(plot.data, paste0("sdm-map-ssdr-", species, "-", pred.type, ".png")), 
          width = 1000, height = 1000, pointsize = 35)
      
      # cast map 
      sdm.map.spec <- tm_shape(spec.countries) +
                                tm_borders() +
                                tm_shape(grid, raster.downsample = F) +
                                tm_raster(style = "cont", breaks = seq(0, 1, by = .2),
                                palette = viridis::viridis(10), alpha = .8) #viridis::magma(1e3)
      
      # display plot
      print(sdm.map.spec)
      
      # save file
      dev.off() 
      
      rm(env_grids, grid.pred, grid.pred.s, sdm.map.spec)
      gc()
    
    #save results
    #simu.res.path <- file.path(temp.data, paste0("simu-runs-", spec, "-", pred.type, "-", timestamp, ".rds"))
    #if(sum(simu.results) != 0){saveRDS(simu.results, file = simu.res.path) } else{print("No luck!")}
    
    
    ################################################################################################
    ### estimate mgcv GAM model for prediction map #################################################
    ################################################################################################
    
    cat("\n", "Initiate mgcv GAM for species", species, "\n")
    
    
    # define GAM formula as in smooth predictor
    form.gam <- formula(paste("presence ~ 1 + s(",
                              paste(varlist[3:(length(varlist))], collapse = ") + s("),
                              ")",
                              " + s(longitude, latitude, bs = 'gp')"
    ))
    form.gam
    
    # fit fast GAM (still takes min 30m with discrete = F)
    mod.gam <- mgcv::bam(
      formula  = form.gam,
      data     = cbind(presence = y.comb, x.comb),
      family   = binomial(),
      method   = "fREML",
      select = TRUE,
      discrete = F) # set to true for much faster computation 
    
    summary(mod.gam)
    
    # predict test fold
    pred.gam <- predict(mod.gam, newdata = x.test.s, type = "response")
    summary(pred.gam)

    # evaluate prediction with AUC
    auc.gam <- round(MLmetrics::AUC(y_pred = pred.gam, y_true = y.test), digits=3)
    auc.gam
    
    # evaluate prediction with Brier score
    brier.gam <- round(BrierScore(resp = y.test, pred = pred.gam), digits=3)
    brier.gam
    
    # create prediction map 
    gc()
    memory.limit(9999999999)
    # create prediction map 
    env_grids <- raster::brick(file.path(main.path, "data", "raw-data", "env_grids.gri"))
    # grid_to_df is mastergrids package function (Bender et al)
    # create data frame with 1 observation per 5x5km square
    grid.pred <- grid_to_df(env_grids, spec.data$hull, spec.data$hull)
    # note: when fitted with bam, use discrete = FALSE within predict for final
    # prediction (takes longer but finer resolution)
    # predict(mod, newdata = pred_df, type = "response")
    
    # bake grid df according to train data parameters (standardize data wrt train set)
    grid.pred.s <- bake(rec_obj, new_data = grid.pred)
    gc()
    # predict response on grid df (takes 10-20 min)
    grid.pred$prediction <- predict(mod.gam, newdata = grid.pred.s, type = "response")
    
    # tranform data frame back to grid for visualization
    env_grids <- raster::brick(file.path(main.path, "data", "raw-data", "env_grids.gri"))
    grid <- df_to_grid(grid.pred, env_grids[[1]], "prediction")
    
    # initialize  plot
    png(file = file.path(plot.data, paste0("sdm-map-gam-", species, "-", pred.type, ".png")), 
        width = 1000, height = 1000, pointsize = 35)
    
    # cast map 
    sdm.map.spec <- tm_shape(spec.countries) +
      tm_borders() +
      tm_shape(grid, raster.downsample = F) +
      tm_raster(style = "cont", breaks = seq(0, 1, by = .2),
                palette = viridis::viridis(10), alpha = .8)
    
    # display plot
    print(sdm.map.spec)
    
    # save file
    dev.off() 
    
    rm(grid.pred.s, env_grids, grid.pred)
    gc()
    
  
  # species loop end
}



