################################################################################################
# Model comparisons for single-species models
################################################################################################

rm(list=ls())

species.list <- c("tinfestans", "tdimidiata", "pmegistus", "tbrasiliensis", "tsordida", "tpseudomaculata", "tbarberi")
spec = "pmegistus"

# directories
main.path <- dirname(rstudioapi::getSourceEditorContext()$path)
repo.path <- file.path(main.path, "repo", "deepregression-master")
temp.data <- file.path(main.path, "temp")

# timestap to name all output files
timestamp <- format(Sys.time(), "%Y-%m-%d-%H%M")


#################################################################################################
## set up workspace
#################################################################################################

library(devtools)

# load deepregression and spatial packages
## install.packages("pacman")
pacman::p_load(Matrix, dplyr, keras, mgcv, reticulate, tensorflow, tfprobability, 
               Metrics, DescTools, MLmetrics, caret, xgboost, recipes, yardstick
               ,ParBayesianOptimization, doParallel, muStat, scoring, rlang, maxnet
               ,rstudioapi
)


# force conda environment
use_condaenv("r-reticulate", required = T)

# load deepregression package
load_all(repo.path)

# seed
seed = 42
tf$random$set_seed(seed = seed)
set.seed(seed)

################################################################################################
## load and prepare data #######################################################################
################################################################################################

# create data frame for results (nummodels*metrics x species via cbind)
insample.species <- c("Triatoma infestans", "Triatoma dimidiata", "Panstrongylus megistus",
                      "Triatoma brasiliensis", "Triatoma sordida", "Triatoma pseudomaculata",
                      "Triatoma barberi")

compar.single <- as.data.frame(matrix(rep(0,42),nrow = 6, ncol=7))
compar.single <- cbind(model = c(rep("mgcv",2), rep("xgboost",2), rep("maxent",2)), compar.single)
colnames(compar.single) <- c("model", species.list)
rownames(compar.single) <- c("AUC1", "Brier1","AUC2", "Brier2","AUC3", "Brier3")
compar.single

for (spec in species.list){
    # tinfestans, tdimidiata, pmegistus, tbrasiliensis, tsordida, tpseudomaculata, tbarberi
    species <- spec #"tinfestans"
    
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
    
    # define predictor formulae: smooth, smooth.deep, deep
    source(file = file.path(main.path, "auxiliary-scripts", "formulae.R"))
    varlist <- varlist
    
    ################################################################################################
    #################################### GAM comparison  ###########################################
    ################################################################################################
    
    cat("\n", "Initiate mgcv GAM for species", species, "\n")
    
    
    # define GAM formula as in smooth predictor
    form.gam <- formula(paste("presence ~ 1 + s(",
                              paste(varlist[3:length(varlist)], collapse = ") + s("),
                              ")",
                              " + s(longitude, latitude, bs = 'gp')"
    ))
    
    # fit fast GAM
    mod.gam <- mgcv::bam(
      formula  = form.gam,
      data     = cbind(presence = y.train, x.train.s),
      family   = binomial(),
      method   = "fREML",
      select = TRUE,
      discrete = T)
    summary(mod.gam)
    
    # predict test fold
    pred.gam <- predict(mod.gam, newdata = x.test.s, type = "response")
    summary(pred.gam)
    assign(paste0("pred.gam.", species), pred.gam)
    
    # evaluate prediction with AUC
    auc.gam <- round(MLmetrics::AUC(y_pred = pred.gam, y_true = y.test), digits=3)
    auc.gam
    assign(paste0("auc.gam.", species), auc.gam)
    
    # evaluate prediction with Brier score
    brier.gam <- round(BrierScore(resp = y.test, pred = pred.gam), digits=3)
    brier.gam
    assign(paste0("brier.gam.", species), brier.gam)
    
    compar.single[1:2,species] <- c(auc.gam, brier.gam)
    print(compar.single)
    
    ################################################################################################
    #################################### xgboost comparison  #######################################
    ################################################################################################
    
    cat("\n", "Initiate xgboost for species", species, "\n")
    
    # prepare xgb data matrix
    new_train <- model.matrix(~.+0,data = x.train.s) 
    new_test <- model.matrix(~.+0,data = x.test.s) 
    dtrain <- xgb.DMatrix(data = new_train, label = as.numeric(y.train))
    dtest <- xgb.DMatrix(data = new_test, label = as.numeric(y.test))
    
    
    # define parameters
    params <- list(booster = "gbtree",
                   objective = "binary:logistic"
                   #eta=0.3,
                   #gamma=0,
                   #max_depth=6,
                   #min_child_weight=1,
                   #subsample=1,
                   #colsample_bytree=1
    )
    
    # estimate model
    mod.xgb <- xgb.train(
      params = params, 
      data = dtrain, 
      nrounds = 1000, 
      watchlist = list(val=dtest, train=dtrain), 
      print_every_n = 100, 
      #early_stop_rounds = 20,
      maximize = T , 
      eval_metric = "auc"
    )
    
    # predict test set
    pred.xgb <- predict(mod.xgb, dtest)
    #hist(pred.xgb)
    summary(pred.xgb)
    assign(paste0("pred.xgb.", species), pred.xgb)
    
    # evaluate prediction via AUC
    auc.xgb <- AUC(y_pred = pred.xgb, y_true = y.test)
    auc.xgb
    assign(paste0("auc.xgb.", species), auc.xgb)
    
    # evaluate prediction with Brier score
    brier.xgb <- round(BrierScore(resp = y.test, pred = pred.xgb), digits=3)
    brier.xgb
    assign(paste0("brier.xgb.", species), brier.xgb)
    
    compar.single[3:4,species] <- c(auc.xgb, brier.xgb)
    print(compar.single)
    

    ################################################################################################
    ########################### maxent comparison (via glmnet maxnet)  #############################
    ################################################################################################
    
    cat("\n", "Initiate maxent (maxnet) for species", species, "\n")
    
    # fit model on training data
    mod.maxnet <- maxnet::maxnet(p = y.train, data = x.train.s, clamp = F)
    
    # predict test data
    pred.maxnet <- predict(mod.maxnet, newdata = x.test.s, clamp = F, type = c("logistic"))
    summary(pred.maxnet)
    assign(paste0("pred.maxnet.", species), pred.maxnet)
    
    # evaluate prediction via AUC
    auc.maxnet <- AUC(y_pred = pred.maxnet, y_true = y.test)
    auc.maxnet
    assign(paste0("auc.maxnet.", species), auc.maxnet)
    
    # evaluate prediction with Brier score
    brier.maxnet <- round(BrierScore(resp = y.test, pred = pred.maxnet), digits=3)
    brier.maxnet
    assign(paste0("brier.maxnet.", species), brier.maxnet)
    
    compar.single[5:6,species] <- c(auc.maxnet, brier.maxnet)
    print(compar.single)
    
    # increase column counter
    #j = j+1
    
# end loop over species
}

compar.single
colnames(compar.single) <- c("model", insample.species)

# save results
saveRDS(compar.single, file = file.path(temp.data, paste0("single-compar-results-", timestamp, ".rds")))
