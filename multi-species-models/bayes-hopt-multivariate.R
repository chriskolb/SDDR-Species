################################################################################################
# Bayesian Hyperparameter Optimization for Multi-Response model including all species
################################################################################################

rm(list=ls())

# script parameters: 

# define SDDR predictor list
preds.list <- c("deep", "smooth", "smooth-deep")

# loop over predictor specifications
#pred.type = "deep" # smooth, deep, smooth-deep
for(pred.type in preds.list){

# fixed number of epochs for optimization
epochs.cv = 100

# bayesian optimization steps
init.pts = 100
optim.steps = 100

# number of simulations for mean result of optimized model
numsims = 10

# set subsampling type (none, lpros, lprus, adasyn)
subsampling <- "none"
#subsampling <- "adasyn"
#subsampling <- "lpros"
#subsampling <- "lprus"

# set Bayesian Optimization criterion (auc, loss)
optim.crit <- "auc"
#optim.crit <- "loss"

# directories
main.path <- dirname(rstudioapi::getSourceEditorContext()$path)
repo.path <- file.path(main.path, "repo", "deepregression-master")
temp.data <- file.path(main.path, "temp")

#################################################################################################
## set up workspace
#################################################################################################

library(devtools)

# load required packages
## install.packages("pacman")
pacman::p_load(Matrix, dplyr, keras, mgcv, reticulate, tensorflow, tfprobability, 
               Metrics, DescTools, MLmetrics, caret, xgboost, recipes, yardstick,
               Rcpp, ParBayesianOptimization, doParallel, muStat, scoring, Hmisc, blockCV, 
               UBL)

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

# load list of prepared objects
full.list <- readRDS(file = file.path(main.path, "data", "full-model-list.rds"))

# load df.tune as training data for bayesian hyperparameter optimization
train <- full.list$basic.dfs$df.tune.var

# original test sets from single species df
test <- full.list$basic.dfs$test.full

# wrap df in list for pre-processing script
data <- list(train = train, test = test)

# set minimum number of unique observations per covariate
min.unique <- 45

# run data pre-processing script
source(file = file.path(main.path, "auxiliary-scripts", "data-preprocessing.R"))

# define predictor formulae:  smooth, smooth-deep, deep
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
########################## Initialize ScoringFunction and DNN ##################################
################################################################################################

# define scoring function for bayesian optimization
scoringFunction <- function(hidden.units.num = 3, hidden.layers = 3, log.lr.nn = -3, log.decay.nn = -4, 
                            dropout.nn = 0.3, batchnorm = 1, df.smooth = 9) {
  
  if(pred.type == "deep"){df.smooth = NULL} else{df.smooth = df.smooth}
  
  ## initialize hyperparameter arguments
  # transform log rates to nominal value
  lr.nn <- exp(log.lr.nn)
  decay.nn <- exp(log.decay.nn)
  
  # hidden units in first hidden layer
  hidden.units.list <- c(32, 64, 128, 200)
  hidden.units1 <- hidden.units.list[hidden.units.num]
  
  # activation function
  activations <- list("relu", "tanh")
  act.nn  <- activations[[act.num]]  # act.num
  
  # initializers
  initializers   <- list(
    "initializer_glorot_normal()",
    "initializer_he_normal()"
  )
  #"initializer_glorot_uniform()", "initializer_he_uniform()"
  init.nn <- initializers[[init.num]]
  
  
  ## define NN architecture as function of hyperparameters
  # start.block + building.block.1 + ... + building.block.i + end.block
  
  # start block  
  start.block <- paste0(
    "function(x) x %>%
    layer_dense(units = ", hidden.units1, ", activation = '", act.nn, "', 
                kernel_initializer = ", init.nn, "
                ) %>% \n",
    ifelse(batchnorm==1, "    layer_batch_normalization() %>% \n", ""),
    "    layer_dropout(rate = ", dropout.nn, ", seed = seed) %>% \n"
  )  
  
  # middle blocks (hidden layers 2-x)
  building.block <- paste0(
    "    layer_dense(units = ", 0.5*hidden.units1, ", activation = '", act.nn, "', 
                kernel_initializer = ", init.nn, "
                ) %>% \n",
    ifelse(batchnorm==1, "    layer_batch_normalization() %>% \n", ""), 
    "    layer_dropout(rate = ", dropout.nn, ", seed = seed) %>% \n"
  )  
  nn.body <- paste0(strrep(times = (hidden.layers-1), building.block))
  nn.body 
  
  # output layer
  end.block <- paste0(
    "    layer_dense(units = 7, activation = 'linear', 
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
  ###########################  CV for bayesian optimization ######################################
  ################################################################################################
  
  
  # repeat estimation twice to reduce variance of result
  #for(rep in 1:2){
  
  # initialize 4-fold CV results 
  # validation auc results for all cv folds
  val.auc.folds <- c(NULL, NULL, NULL, NULL)
  
  # cross-validated cross-entropy loss (multi-label loss = summed binary cross-entropies)
  cv.loss.folds <- c(NULL, NULL, NULL, NULL)
  
  # validation loss results from randomly chosen validation_split within training fold
  val.loss.folds <- c(NULL, NULL, NULL, NULL)
  
  # loop over spatially de-correlated cv folds
  for(fold in 1:4){
    
    force(cat("\n", "\n", "### Fitting fold", fold, "...\n") )
    
    # define deepregression object
    suppressWarnings(
      mod.bayes.opt <-  deepregression(y = y.train[fold.var != fold,], 
                                       list_of_formulae = list(logit = form.mod),
                                       list_of_deep_models = list(nn_deep = nn_deep), #nn_deep = nn_deep
                                       data = x.train.s[fold.var != fold,],
                                       family = "bernoulli",
                                       dist_fun = function(x) tfd_bernoulli(x),
                                       lambda_lasso = lambda.l1,
                                       df = df.smooth,
                                       orthog_type = "manual",
                                       zero_constraint_for_smooths = TRUE,
                                       absorb_cons = FALSE, 
                                       optimizer = optimizer_adam(
                                         decay = decay.nn, 
                                         lr = lr.nn 
                                       )
                                       #validation_data = list(x.train.s[fold.var == fold,], y.train[fold.var == fold])
                                       #cv_folds = spat.cv.indices, 
                                       #monitor_metric = auc_metric
      )
    )
    
    # cross-validate number of epochs 
    # [do cross-validation manually in bayesian optimization als val_auc cannot be used as metric here]
    #cvres.bopt <- mod.bayes.opt %>% cv(epochs = 30, verbose = TRUE, plot = TRUE, print_folds = T, patience = 10)
    #resiter.bopt<- stop_iter_cv_result(cvres.bopt, whichFUN = which.max) # whichFUN = which.max if accuracy metric is used
    
    resiter.bopt <- epochs.cv
    
    # fit model with optimal numebr of epochs
    suppressWarnings(
      hist.bopt <- mod.bayes.opt %>% fit(epochs = resiter.bopt, 
                                         view_metrics = FALSE, 
                                         verbose = TRUE,
                                         batch_size = batch.size.nn,
                                         #auc_callback = TRUE,
                                         #val_data = list(x.train.s[fold.var == fold,], y.train[fold.var == fold]),
                                         validation_split = 0.1
      )
    )
    
    # predict on val set
    sapply(x.train.s[fold.var == fold,], function(x) length(unique(x)))
    pred <- mod.bayes.opt %>% predict(x.train.s[fold.var == fold,])
    colnames(pred) <- insample.species
    #hist(pred)
    summary(pred)
    
    # predict on test set
    #pred.test <- mod.bayes.opt %>% predict(x.test.s)
    #hist(pred)
    #summary(pred.test)
    
    ### save validation AUCs for that fold for each species and compute average
    
    #spec.weights <- colSums(y.train[fold.var == fold,])
    #spec.weights <- spec.weights/sum(spec.weights)
    
    aucs.fold <- NULL
    #losses.fold <- NULL
    for(spec in 1:7){
      auc.bhopt.spec.fold <- MLmetrics::AUC(y_pred = pred[,spec], y_true = y.train[fold.var == fold, spec])
      aucs.fold <- rbind(aucs.fold, auc.bhopt.spec.fold)
      
      #loss.bhopt.spec.fold <- loss_binary_crossentropy(y_true = y.train[fold.var == fold, spec], y_pred = pred[,spec])
      #loss.bhopt.spec.fold <- k_eval(loss.bhopt.spec.fold)
      #losses.fold <- rbind(losses.fold, loss.bhopt.spec.fold)
      }
    auc.bayes.opt.fold <- mean(aucs.fold)
    
    ### compute multi-label cv loss as binary cross-entropy summed over species (independent bernoullis)
    
    loss.bayes.opt.fold <- loss_binary_crossentropy(y_true = y.train[fold.var == fold,], y_pred = pred)
    loss.bayes.opt.fold <- k_eval(loss.bayes.opt.fold)*len(insample.species)
    loss.bayes.opt.fold <- mean(loss.bayes.opt.fold)
    
    val.auc.folds[fold] <- auc.bayes.opt.fold
    cv.loss.folds[fold] <- loss.bayes.opt.fold
    val.loss.folds[fold] <- hist.bopt$metrics$val_loss[epochs.cv]
    
    
    cat(" Multi-label (macro) AUC for fold", fold, "is = ", auc.bayes.opt.fold, "\n",
        "Cross-entropy loss for fold", fold, " is = ", loss.bayes.opt.fold, "\n",
        "validation(_split) loss on all folds except", fold, "is =", hist.bopt$metrics$val_loss[epochs.cv]
    )
  
  # end loop over CV folds    
  }
  
  
  # compute spatially cross-validated multi-label AUC as mean over folds
  spat.cv.auc <- mean(val.auc.folds, na.rm=TRUE)
  spat.cv.auc
  
  # compute spatially cross-validated cross-entropy (loss) as mean over folds
  spat.cv.loss <- mean(cv.loss.folds, na.rm=TRUE)
  spat.cv.loss
  
  # compute simple cv loss from validation_split as mean over folds
  val.loss <- mean(val.loss.folds, na.rm=TRUE)
  val.loss
  
  cat("\nCross-validated multi-label macro AUC at last epoch is = ", spat.cv.auc, "\nwith number of epochs = ", epochs.cv, "\n",
      "\nCross-validated multi-label cross-entropy at last epoch is = ", spat.cv.loss, "\n",
      "\nMulti-label loss from validation_split at last epoch is =  ", val.loss, "\n"
  )
  
  # assign criterion to optimize
  if(optim.crit == "auc"){criterion = spat.cv.auc} else{criterion = -spat.cv.loss}
  #assign(paste0("criterion", rep), criterion)
  
  rm(mod.bayes.opt, hist.bopt)
  
  # end loop over CV repetitions 
  #}
  
  # average criterion over reps to avoid finding spurious hyperparameter optima
  mean.crit <- criterion #mean(criterion1, criterion2)
  
  cat("\nMean criterion for Bayesian Hyperparameter Optimization is = ", mean.crit, "\n")
  
  return(list(
    Score              = mean.crit
    ,cv.auc            = spat.cv.auc
    ,cv.loss           = spat.cv.loss
    ,val.loss          = val.loss
    #,spat.cv.epochs   = spat.cv.epochs
    #,cv.results      = val.auc.folds
    #,opt.fold.epochs = opt.epochs
  )
  )
}


# define wrapper for scoring function for each predictor to optimize different hyperparameters

if(pred.type == "smooth-deep"){
  ScoringWrapper <- function(hidden.units.num = 3, hidden.layers = 3, log.lr.nn = -4, log.decay.nn = -4, 
                             dropout.nn = 0.3, df.smooth = 9){
    
    res <- scoringFunction(hidden.units.num = hidden.units.num,
                           hidden.layers = hidden.layers, log.lr.nn = log.lr.nn,
                           log.decay.nn = log.decay.nn, dropout.nn = dropout.nn,
                           batchnorm = 1, df.smooth = df.smooth)
    mean.crit <- res$Score
    
    return(list(Score = mean.crit))
  }
} else {
  if(pred.type == "deep"){
    ScoringWrapper <- function(hidden.units.num = 3, hidden.layers = 3, log.lr.nn = -4, log.decay.nn = -4, 
                               dropout.nn = 0.3){
      
      res <- scoringFunction(hidden.units.num = hidden.units.num,
                             hidden.layers = hidden.layers, log.lr.nn = log.lr.nn,
                             log.decay.nn = log.decay.nn, dropout.nn = dropout.nn,
                             batchnorm = 1, df.smooth = NULL)
      mean.crit <- res$Score
      
      return(list(Score = mean.crit))
    }
    
  } else{
    # pred.type == "smooth"
    ScoringWrapper <- function(log.lr.nn = -4, log.decay.nn = -4, df.smooth = 9){
      
      res <- scoringFunction(hidden.units.num = NULL,
                             hidden.layers = NULL, log.lr.nn = log.lr.nn,
                             log.decay.nn = log.decay.nn, dropout.nn = NULL,
                             batchnorm = NULL, df.smooth = df.smooth)
      mean.crit <- res$Score
      
      return(list(Score = mean.crit))
    }
    
  }
}

################################################################################################
######################## Run Bayes. Optim. on ScoringWrapper function  #########################
################################################################################################


# define parameter bounds for search
# act.num, hidden.units, init.num, dropout.nn, lr.nn, decay.nn, batch.size.nn, epochs.cv, lambda.l1

if(pred.type == "smooth-deep"){
bounds <- list(
  hidden.units.num  = c(1L, 4L) # 32, 64, 128, 200
  ,hidden.layers    = c(1L, 4L) # layers after first hidden layer have hidden.units/2 units
  ,log.lr.nn        = c(-6, -2.5)
  ,log.decay.nn     = c(-6, -2.5)
  #,batch.size.nn   = c(5L, 30L)
  ,dropout.nn       = c(0.01, 0.49)
  #,lambda.l1       = c(1e-6, 5e-2)
  #,epochs.cv       = c(4L, 16L)
  ,df.smooth        = c(5L, 15L)
  #,act.num         = c(1L,2L) # 1=relu,2=tanh
  #,init.num        = c(1L,2L) # 1=glorot_normal, 2=he_normal
  #,batchnorm       = c(1L,2L) # 1=yes, 2=no
)
} else {
  if(pred.type == "deep"){
    bounds <- list(
      hidden.units.num  = c(1L, 4L) # 32, 64, 128, 200
      ,hidden.layers    = c(1L, 4L) # layers after first hidden layer have hidden.units/2 units
      ,log.lr.nn        = c(-6, -2.5)
      ,log.decay.nn     = c(-6, -2.5)
      #,batch.size.nn   = c(5L, 30L)
      ,dropout.nn       = c(0.01, 0.49)
      #,lambda.l1       = c(1e-6, 5e-2)
      #,epochs.cv       = c(4L, 16L)
      #,df.smooth       = c(3L, 12L)
      #,act.num         = c(1L,2L) # 1=relu,2=tanh
      #,init.num        = c(1L,2L) # 1=glorot_normal, 2=he_normal
      #,batchnorm       = c(1L,2L) # 1=yes, 2=no
    )
  } else {
    # pred.type == "smooth"
    bounds <- list(
      #hidden.units.num  = c(1L, 4L) # 32, 64, 128, 200
      #,hidden.layers    = c(1L, 4L) # layers after first hidden layer have hidden.units/2 units
      log.lr.nn          = c(-5, -1)
      ,log.decay.nn      = c(-5, -1)
      #,batch.size.nn    = c(5L, 30L)
      #,dropout.nn       = c(0.01, 0.49)
      #,lambda.l1        = c(1e-6, 5e-2)
      #,epochs.cv        = c(4L, 16L)
      ,df.smooth         = c(5L, 15L)
      #,act.num          = c(1L,2L) # 1=relu,2=tanh
      #,init.num         = c(1L,2L) # 1=glorot_normal, 2=he_normal
      #,batchnorm        = c(1L,2L) # 1=yes, 2=no
    )
  }
}

## run hyperparameter search and optimization
ScoreResult <- bayesOpt(
  FUN = ScoringWrapper
  , bounds = bounds
  , initPoints = init.pts
  , iters.n = optim.steps
  , iters.k = 1
  , acq = "ucb"
  , kappa = 3
  #, eps = 0.1 # for acq = "ei"
  , gsPoints = 10
  , parallel = FALSE
  , verbose = 2
  , plotProgress = TRUE
  , errorHandling = "continue"
  , saveFile = file.path(temp.data, paste0("bhopt-MR-", optim.crit, "-", subsampling, "-", pred.type,
                                           "-spatcv-",  timestamp, ".rds"))
  )

################################################################################################
## prepare model with optimized hyperparameters ################################################
################################################################################################

# get best parameters for highest CV AUC
best.pars.bopt <- getBestPars(ScoreResult)
#best.pars.bopt <- getBestPars(ScoreResultAdd)
best.pars.bopt

# re-transform log rates from hyperparam search
lr.nn <- exp(best.pars.bopt$log.lr.nn) 
decay.nn <- exp(best.pars.bopt$log.decay.nn)

# set batchnorm to 1 (yes) per default
batchnorm = 1 # 1 = yes, 2 = no

# hidden units in first hidden layer
hidden.units.list <- c(32, 64, 128, 200)
hidden.units1 <- hidden.units.list[best.pars.bopt$hidden.units.num]

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
  "    layer_dense(units = 7, activation = 'linear', 
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
##################### train and evaluate on species-specific data sets #########################
################################################################################################

insample.species <- c("Triatoma infestans", "Triatoma dimidiata", "Panstrongylus megistus",
                      "Triatoma brasiliensis", "Triatoma sordida", "Triatoma pseudomaculata",
                      "Triatoma barberi")

#insample.species <- c("Triatoma infestans", "Triatoma dimidiata")

# create data frame for results (numsims x species)
simu.runs.spec <- as.data.frame(1:numsims)
names(simu.runs.spec) <- "run"

# create data frame for results (numsims x species)
brier.runs.spec <- as.data.frame(1:numsims)
names(brier.runs.spec) <- "run"

# set seed
tf$random$set_seed(seed = seed)
set.seed(seed)

# loop over insample species
for(spec in insample.species){
  
  ## prepare species-specific data sets
  
  # load list of prepared objects
  #full.list <- readRDS(file = file.path(main.path, "full-model-list.rds"))
  
  # load species-specific train and test sets
  train <- eval(parse(text = paste0("full.list$train.dfs.var.spec$train.full.var.", gsub(" ", "", spec, fixed = TRUE))))
  
  # original test sets from single species df
  test <- full.list$basic.dfs$test.full
  test <- test[test$spec == spec,]
  
  # wrap df in list for pre-processing script
  data <- list(train = train, test = test)
  
  # set minimum number of unique observations per covariate
  min.unique <- 45
  
  # run data pre-processing script
  source(file = file.path(main.path, "auxiliary-scripts", "data-preprocessing-species.R"))
  
  # recalibrate formulae to new data frame (other covariates)
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
  
  # train model on species-specific full data set for numsims simulations
  
  simu.runs <- NULL
  simu.runs <- as.data.frame( rep(0, numsims) )
  names(simu.runs) <- spec
  
  brier.runs <- NULL
  brier.runs <- as.data.frame(rep(0, numsims))
  names(brier.runs) <- spec
  
  for(simu in 1:numsims){
    
    # define deepregression object
    mod.post.bopt <-  deepregression(y = y.train,
                                     list_of_formulae = list(logit = form.mod),
                                     list_of_deep_models = list(nn_deep = nn_deep),
                                     data = x.train.s,
                                     family = "bernoulli",
                                     dist_fun = function(x) tfd_bernoulli(x),
                                     lambda_lasso = lambda.l1,
                                     df = best.pars.bopt$df.smooth,
                                     orthog_type = "manual",
                                     zero_constraint_for_smooths = TRUE,
                                     absorb_cons = FALSE, 
                                     optimizer = optimizer_adam(
                                       decay = decay.nn, 
                                       lr = lr.nn 
                                     )
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
      #,auc_callback = TRUE
      #,val_data = list(x.test.s, y.test)
      ,validation_split = 0
    )  
    
    
    # predict on test set
    pred <- mod.post.bopt %>% predict(x.test.s)
    colnames(pred) <- insample.species
    hist(pred)
    summary(pred)
    pred.spec <- pred[,spec]
    
    # evaluate prediction with AUC
    auc.test <- MLmetrics::AUC(y_pred = pred.spec, y_true = y.test[,spec])
    #auc.test <- Metrics::auc(actual = y.test[,spec], predicted = pred.spec)
    cat("\nTest AUC for species", spec, "and predictor", pred.type, 
        "in simulation number", simu, " is ", auc.test, "\n")
    
    # evaluate prediction with Brier score
    brier.test <- BrierScore(resp = y.test[,spec], pred = pred.spec)
    cat("\nTest Brier score for species", spec, "and predictor", pred.type,
        "in simulation number", simu, " is ", brier.test, "\n\n")
    
    
    simu.runs[simu,1] <- auc.test
    brier.runs[simu,1] <- brier.test
  
  # end loop over simulation runs  
  }
  
  simu.runs
  auc.mean <- round(mean(simu.runs[,1]), digits =3)
  auc.sd   <- signif(stdev(simu.runs[,1], unbiased = T), digits=1)

  brier.runs
  brier.mean <- round(mean(brier.runs[,1]), digits=3)
  brier.sd <- signif(stdev(brier.runs[,1], unbiased = T), digits=1)

  cat(" Final multi-response model for species", spec, "and predictor", pred.type, "\n",  
      "yields mean test AUC of = ", auc.mean,
      " with sd = ", auc.sd, "\n")
  cat(" Final multi-response model for species", spec, "and predictor", pred.type, "\n",  
      "yields mean test Brier Score of = ", brier.mean,
      " with sd = ", brier.sd, "\n")
  
  simu.runs.spec <- cbind(simu.runs.spec, simu.runs)
  brier.runs.spec <- cbind(brier.runs.spec, brier.runs)

# end loop over species  
}

# aggregate test AUC results for all species
simu.summary <- simu.runs.spec  %>% summarize_all(list(m=mean, sd=stats::sd))
simu.means <- round(simu.summary[1:(len(insample.species)+1)], digits=3)
names(simu.means) <- names(simu.runs.spec)
simu.sds <- signif(simu.summary[(len(insample.species)+2):(2*len(insample.species)+2)], digits=1)
names(simu.sds) <- names(simu.runs.spec)
simu.runs.all <- rbind(simu.runs.spec, simu.means, simu.sds)
rownames(simu.runs.all)[(numsims+1):(numsims+2)] <- c("means", "stdevs")
View(simu.runs.all)

# aggregate test Brier score for all species
brier.summary <- brier.runs.spec %>% summarize_all(list(m=mean, sd = stats::sd))
brier.means <- round(brier.summary[1:(len(insample.species)+1)], digits=3)
names(brier.means) <- names(brier.runs.spec)
brier.sds <- signif(brier.summary[(len(insample.species)+2):(2*len(insample.species)+2)], digits=1)
names(brier.sds) <- names(brier.runs.spec)
brier.runs.all <- rbind(brier.runs.spec, brier.means, brier.sds)
rownames(brier.runs.all)[(numsims+1):(numsims+2)] <- c("means", "stdevs")
View(brier.runs.all)

# save results
saveRDS(simu.runs.all, file = file.path(temp.data, paste0("simu-runs-MR-", optim.crit, "-", subsampling, "-", pred.type,
                                                           "-",  timestamp, ".rds")))
saveRDS(brier.runs.all, file = file.path(temp.data, paste0("brier-simus-MR-", optim.crit, "-", subsampling, "-", pred.type,
                                                            "-",  timestamp, ".rds")))

# end loop over predictor types from beginning of script
}











