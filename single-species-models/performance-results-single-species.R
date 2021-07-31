################################################################################################
# AUC and Brier mean score simulation runs with optimized hyperparameters for all species and 
# predictor types (smooth, deep, smooth-deep)
################################################################################################

rm(list=ls())


species.list <- c("tinfestans", "tdimidiata", "pmegistus", "tbrasiliensis", "tsordida", "tpseudomaculata", "tbarberi")
spec = "pmegistus"

# define SDDR predictor list
preds.list <- c("smooth", "deep", "smooth-deep")
pr = "smooth-deep"

# fixed number of epochs
epochs.cv = 100

# number of simulations for mean result of optimized model
numsims = 10

# directories
main.path <- dirname(rstudioapi::getSourceEditorContext()$path)
repo.path <- file.path(main.path, "repo", "deepregression-master")
temp.data <- file.path(main.path, "temp")


#################################################################################################
## set up workspace
#################################################################################################

library(devtools)

# load deepregression and spatial packages
## install.packages("pacman")
pacman::p_load(Matrix, dplyr, keras, mgcv, reticulate, tensorflow, tfprobability, 
               Metrics, DescTools, MLmetrics, caret, xgboost, recipes, yardstick
               ,ParBayesianOptimization, doParallel, muStat, scoring, rstudioapi
               ,Rcpp
)

# force conda environment
use_condaenv("r-reticulate", required = T)

# load deepregression package
load_all(repo.path)

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
  for (pr in preds.list){
    
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
    
    # define predictor formulae: smooth, smooth.deep, deep
    source(file = file.path(main.path, "auxiliary-scripts", "formulae.R"))
    
    # model formula
    if (pred.type == "smooth"){
      form.mod <- form.smooth
    } else if (pred.type == "smooth-deep"){
      form.mod <- form.smooth.deep
    } else{
      form.mod <- form.deep
    }
    form.mod
    
    # initialize arguments for testing purposes
    batch.size.nn = 20
    lambda.l1 = 0
    batchnorm = 1 # 1 = yes, 2 = no
    hidden.units.num = 3 # 1=32, 2=64, 3=128, 4=200
    hidden.layers = 2
    dropout.nn = 0.2
    log.lr.nn = -3
    log.decay.nn = -4
    act.num = 1 # 1=relu, 2=tanh
    init.num = 2 # 1=glorot_normal, 2=he_normal
    if(pred.type == "deep"){df.smooth = NULL} else{df.smooth = 9}
    
    ################################################################################################
    ### prepare DNN with optimized hyperparameters #################################################
    ################################################################################################
    
    # load bayes opt object
    
    # load bayesian hyperparameter optimization object (ParBayesOptimization)
    bayes.opt.result <- readRDS(file.path(main.path, "bayesian-optimization", species, "results", 
                                          paste0("bayesopt-results-", species, "-", pred.type, "-fixed", ".RDS")))
    
    # print optimal hyperparameters
    best.pars <- getBestPars(bayes.opt.result)
    best.pars
    best.pars.bopt <- best.pars
    
    # re-transform log rates from hyperparam search
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
    ### run simulations with optimized hyperparameter set ##########################################
    ################################################################################################
    
    # set seed
    tf$random$set_seed(seed = seed)
    set.seed(seed)
    
    # create data frame for AUC results
    auc.runs <- as.data.frame( rep(0, numsims) )
    names(auc.runs) <- "runs"
    
    # create data frame for Brier results 
    brier.runs <- as.data.frame(rep(0,numsims))
    names(brier.runs) <- "runs"
    
    for(simu in 1:numsims){
      
      # define deepregression object
      mod.post.bopt <-  deepregression(y = y.train,
                                       list_of_formulae = list(logit = form.mod),
                                       list_of_deep_models = list(nn_deep = nn_deep),
                                       data = x.train.s,
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
        ,view_metrics = TRUE 
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
          " in simulation number", simu, "is =", auc.test, "\n")
      
      # evaluate prediction with Brier score
      brier.test <- BrierScore(resp = y.test, pred = pred)
      cat("\nTest Brier score for species", spec, "and predictor", pred.type, 
          "in simulation number", simu, " is ", brier.test, "\n\n")
      
      auc.runs$runs[simu] <- auc.test # evt [simu,1]
      brier.runs$runs[simu] <- brier.test
      
      # end loop over simulation runs  
    }
    
    # aggregate test AUC and Brier Score results
    auc.runs
    auc.mean <- round(mean(auc.runs[,1]), digits =3)
    auc.sd   <- signif(stdev(auc.runs[,1], unbiased = T), digits=1)
    auc.res <- c(auc.runs$runs, auc.mean, auc.sd)
    
    brier.runs
    brier.mean <- round(mean(brier.runs[,1]), digits=3)
    brier.sd <- signif(stdev(brier.runs[,1], unbiased = T), digits=1)
    brier.res <- c(brier.runs$runs, brier.mean, brier.sd)
    
    simu.results <- cbind(auc.res, brier.res)
    rownames(simu.results) <- c(1:numsims,"means", "stdevs") #[(numsims+1):(numsims+2)]
    colnames(simu.results) <- c("AUC", "Brier")
    
    cat(" Final univariate single-species model for species", species, "and predictor", pred.type, "\n",  
        "yields mean test AUC of = ", auc.mean,
        " with sd = ", auc.sd, "\n")
    cat(" Final univariate single-species model for species", species, "and predictor", pred.type, "\n",  
        "yields mean test Brier Score of = ", brier.mean,
        " with sd = ", brier.sd, "\n")
    
    simu.results
    
    #save results
    simu.res.path <- file.path(temp.data, paste0("simu-runs-", spec, "-", pred.type, "-", timestamp, ".rds"))
    if(sum(simu.results) != 0){saveRDS(simu.results, file = simu.res.path) } else{print("No luck!")}
    
    
    #pred.type loop end
  }
  
  # species loop end
}

