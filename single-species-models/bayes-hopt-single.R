################################################################################################
# Bayesian Hyperparameter Optimization (and performance scoring) for single-species models
################################################################################################

rm(list=ls())

species.list <- c("tinfestans", "tdimidiata", "pmegistus", "tbrasiliensis", "tsordida", "tpseudomaculata", "tbarberi")
#species.list <- c("pmegistus")
#spec = "pmegistus"

# define SDDR predictor list
preds.list <- c("smooth", "deep", "smooth-deep")
#preds.list <- c("smooth-deep")
#pr = "smooth"

# fixed number of epochs for optimization
epochs.cv = 100

# bayesian optimization steps
init.pts = 100
optim.steps = 100

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
    ###################Initialize ScoringFunction and DNN ##########################################
    ################################################################################################
    
    
    ### Scoring Function begins here
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
      ############################## Cross-Validation  ###############################################
      ################################################################################################
      
      # validation auc results for all cv folds
      val.auc.folds <- c(NULL, NULL, NULL, NULL)
      
      for(fold in 1:4){
        
        force(cat("\n", "\n", "### Fitting fold", fold, "for species", species, "and predictor type", pred.type, "...\n") )
        
        # define deepregression object
        suppressWarnings(
          mod.bayes.opt <-  deepregression(y = y.train[fold.var != fold], 
                                           list_of_formulae = list(logit = form.mod),
                                           list_of_deep_models = list(nn_deep = nn_deep),
                                           data = x.train.s[fold.var != fold,],
                                           family = "bernoulli",
                                           lambda_lasso = lambda.l1,
                                           df = df.smooth,
                                           #orthog_type = "manual",
                                           #zero_constraint_for_smooths = TRUE,
                                           #absorb_cons = FALSE, 
                                           optimizer = optimizer_adam(
                                             decay = decay.nn, 
                                             lr = lr.nn 
                                           )
                                           #validation_data = list(x.train.s[fold.var == fold,], y.train[fold.var == fold])
                                           #cv_folds = spat.cv.indices, 
                                           #monitor_metric = auc_metric
          )
        )
        
        resiter.bopt <- epochs.cv
        
        # fit model with optimal numebr of epochs
        suppressWarnings(
          hist.bopt <- mod.bayes.opt %>% fit(epochs = resiter.bopt, 
                                             view_metrics = FALSE, 
                                             verbose = TRUE,
                                             batch_size = batch.size.nn,
                                             #auc_callback = TRUE,
                                             #val_data = list(x.train.s[fold.var == fold,], y.train[fold.var == fold])
                                             validation_split = 0.1
          )
        )
        
        # predict on val set
        pred <- mod.bayes.opt %>% predict(x.train.s[fold.var == fold,])
        #hist(pred)
        summary(pred)
        
        # save validation AUCS to compute mean(val_auc) after cv-loop has ended
        auc.bayes.opt.run <- MLmetrics::AUC(y_pred = pred, y_true = y.train[fold.var == fold])
        auc.bayes.opt.run
        val.auc.folds[fold] <- auc.bayes.opt.run 
        
        cat("Validation AUC for this fold is = ", auc.bayes.opt.run, "\n")
      
      # end loop over CV folds  
      }
      
      
      # compute mean val AUC as spatially cross-validated AUC
      spat.cv.auc <- mean(val.auc.folds, na.rm=TRUE)
      spat.cv.auc
      
      cat("\nCross-validated AUC at last epoch is = ", spat.cv.auc, "\nwith number of epochs = ", epochs.cv, "\n")
      
      rm(mod.bayes.opt, hist.bopt)
      
      return(list(Score = spat.cv.auc))
  }
    
    
    
    # define wrapper for scoring function for each predictor to optimize different hyperparameters
    
    if(pred.type == "smooth-deep"){
      ScoringWrapper <- function(hidden.units.num = 3, hidden.layers = 3, log.lr.nn = -4, log.decay.nn = -4, 
                                 dropout.nn = 0.3, df.smooth = 9){
        
        res <- scoringFunction(hidden.units.num = hidden.units.num,
                               hidden.layers = hidden.layers, log.lr.nn = log.lr.nn,
                               log.decay.nn = log.decay.nn, dropout.nn = dropout.nn,
                               batchnorm = 1, df.smooth = df.smooth)
        spat.cv.auc <- res$Score
        
        return(list(Score = spat.cv.auc))
      }
    } else {
      if(pred.type == "deep"){
        ScoringWrapper <- function(hidden.units.num = 3, hidden.layers = 3, log.lr.nn = -4, log.decay.nn = -4, 
                                   dropout.nn = 0.3){
          
          res <- scoringFunction(hidden.units.num = hidden.units.num,
                                 hidden.layers = hidden.layers, log.lr.nn = log.lr.nn,
                                 log.decay.nn = log.decay.nn, dropout.nn = dropout.nn,
                                 batchnorm = 1, df.smooth = NULL)
          spat.cv.auc <- res$Score
          
          return(list(Score = spat.cv.auc))
        }
        
      } else{
        # pred.type == "smooth"
        ScoringWrapper <- function(log.lr.nn = -4, log.decay.nn = -4, df.smooth = 9){
          
          res <- scoringFunction(hidden.units.num = NULL,
                                 hidden.layers = NULL, log.lr.nn = log.lr.nn,
                                 log.decay.nn = log.decay.nn, dropout.nn = NULL,
                                 batchnorm = NULL, df.smooth = df.smooth)
          spat.cv.auc <- res$Score
          
          return(list(Score = spat.cv.auc))
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
        hidden.units.num  = c(3L, 4L) # 32, 64, 128, 200
        ,hidden.layers    = c(1L, 2L) # layers after first hidden layer have hidden.units/2 units
        ,log.lr.nn        = c(-7, -2.5)
        ,log.decay.nn     = c(-7, -3.5)
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
          hidden.units.num  = c(2L, 4L) # 32, 64, 128, 200
          ,hidden.layers    = c(1L, 2L) # layers after first hidden layer have hidden.units/2 units
          ,log.lr.nn        = c(-7, -2.5)
          ,log.decay.nn     = c(-7, -3.5)
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
          log.lr.nn          = c(-6, -1)
          ,log.decay.nn      = c(-6, -3)
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
      , saveFile = file.path(temp.data, paste0("bhopt-single-", spec, "-", pred.type, "-", timestamp, ".rds"))
    )
    
    
    ################################################################################################
    ### prepare DNN with optimized hyperparameters #################################################
    ################################################################################################
    
    # load bayes opt object
    #bayes.opt.result <- readRDS(file.path())
    
    best.pars.bopt <- getBestPars(ScoreResult)
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

