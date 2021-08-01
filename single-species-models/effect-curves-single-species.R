################################################################################################
# Partial effect curves for SDDR (smooth predictor) and mgcv GAM for species P. Megistus
################################################################################################

rm(list=ls())


species.list <- c("tinfestans", "tdimidiata", "pmegistus", "tbrasiliensis", "tsordida", "tpseudomaculata", "tbarberi")
#species.list <- c("pmegistus")
spec = "pmegistus"

# define SDDR predictor list
#preds.list <- c("smooth", "deep", "smooth-deep")
pr = "smooth"

# fixed number of epochs for optimization
epochs.cv = 100

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
    ### prepare smooth model with optimized hyperparameters ########################################
    ################################################################################################
    
    
    # load bayesian hyperparameter optimization object (ParBayesOptimization)
    bayes.opt.result <- readRDS(file.path(main.path, "bayesian-optimization", species, "results", 
                                          paste0("bayesopt-results-", species, "-", "smooth", "-fixed", ".RDS")))
    
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
    ### run smooth model with optimized hyperparameter set #########################################
    ################################################################################################
    
    # set seed
    tf$random$set_seed(seed = seed)
    set.seed(seed)
      
    # define deepregression object
     mod.post.bopt <-  deepregression(y = y.train,
                                       list_of_formulae = list(logit = form.smooth),
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
      cat("\nTest AUC for species", spec, "and predictor smooth", 
          "is =", auc.test, "\n")
      
      # evaluate prediction with Brier score
      brier.test <- BrierScore(resp = y.test, pred = pred)
      cat("\nTest Brier score for species", spec, "and predictor smooth", 
          " is ", brier.test, "\n\n")
      
      
      ### partial effect curves
      
      sddr.curves <- mod.post.bopt %>% plot()
      dev.off()
      
      # exemplary plot for rainfall variable
      rainfall.sddr <- sddr.curves[[1]]
      rainfall.sddr.x <- rainfall.sddr$value
      rainfall.sddr.y <- rainfall.sddr$partial_effect
      
      # position of plots in sddr.curves list for species spec and smooth predictor
      # elevation (2), slope (3), urban (10), population (12), landcover10 (16), landcover12 (17)
      
      # loop over plots to extract values
      varlist.curves <- varlist[3:len(varlist)]
      df.sddr <- data.frame(x = rep(-1, nrow(x.train.s)))
      for(count in 1:(length(sddr.curves)-1)){
        sddr.x <- as.numeric(sddr.curves[[count]]$value)
        sddr.y <- as.numeric(sddr.curves[[count]]$partial_effect)
        df.var <- data.frame(sddr.x, sddr.y)
        colnames(df.var) <- c(paste0("x.", varlist.curves[count]), paste0("y.", varlist.curves[count]))
        df.sddr <- cbind(df.sddr, df.var)
        #colnames(df.sddr)[ncol(df.sddr)] <- paste0("y", count)
      }
      df.sddr <- df.sddr[,-1]
      
      
      ################################################################################################
      ### prepare smooth-deep model with optimized hyperparameters ###################################
      ################################################################################################
      
      # load bayesian hyperparameter optimization object (ParBayesOptimization)
      bayes.opt.result <- readRDS(file.path(main.path, "bayesian-optimization", species, "results", 
                                            paste0("bayesopt-results-", species, "-", "smooth-deep", "-fixed", ".RDS")))
      
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
      ### run smooth-deep model with optimized hyperparameter set ####################################
      ################################################################################################
      
      # set seed
      tf$random$set_seed(seed = seed)
      set.seed(seed)
      
      # define deepregression object
      mod.post.bopt <-  deepregression(y = y.train,
                                       list_of_formulae = list(logit = form.smooth.deep),
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
      cat("\nTest AUC for species", spec, "and predictor smooth-deep", 
          "is =", auc.test, "\n")
      
      # evaluate prediction with Brier score
      brier.test <- BrierScore(resp = y.test, pred = pred)
      cat("\nTest Brier score for species", spec, "and predictor smooth-deep", 
          " is ", brier.test, "\n\n")
      
      
      ### partial effect curves
      
      sddr.curves.sd <- mod.post.bopt %>% plot()
      dev.off()
      
      # position of plots in sddr.curves list for species spec and smooth predictor
      # elevation (2), slope (3), urban (10), population (12), landcover10 (16), landcover12 (17)
      
      # loop over plots to extract values
      varlist.curves <- varlist[3:len(varlist)]
      df.sddr.sd <- data.frame(x = rep(-1, nrow(x.train.s)))
      for(count in 1:(length(sddr.curves.sd)-1)){
        sddr.x <- as.numeric(sddr.curves.sd[[count]]$value)
        sddr.y <- as.numeric(sddr.curves.sd[[count]]$partial_effect)
        df.var <- data.frame(sddr.x, sddr.y)
        colnames(df.var) <- c(paste0("x.", varlist.curves[count]), paste0("y.", varlist.curves[count]))
        df.sddr.sd <- cbind(df.sddr.sd, df.var)
        #colnames(df.sddr)[ncol(df.sddr)] <- paste0("y", count)
      }
      df.sddr.sd <- df.sddr.sd[,-1]
      
            
      ################################################################################################
      ### estimate mgcv GAM model for partial effects ################################################
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
        data     = cbind(presence = y.train, x.train.s),
        family   = binomial(),
        method   = "fREML",
        select = F,
        gamma = 0.8,
        discrete = T) # set to true for much faster computation 
      
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
      
      
      ### partial effect curves
      gam.curves <- plot(mod.gam)
        print(paste0("Press button to continue")) # need to do this as a quick fix
        print(paste0("Press button to continue")) # because each line of code is one key stroke
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))
        print(paste0("Press button to continue"))

      dev.off()
      
      # exemplary plot for rainfall variable
      rainfall.gam <- gam.curves[[1]]
      rainfall.gam.x <- rainfall.gam$x
      rainfall.gam.y <- rainfall.gam$fit
      
      # position of plots in gam.curves list for species spec and smooth predictor
      # elevation (2), slope (3), urban (10), population (12), landcover10 (16), landcover12 (17)
      
      
      # loop over plots to extract values
      varlist.curves <- varlist[3:len(varlist)]
      df.gam <- data.frame(x = rep(-1, len(gam.curves[[1]]$x)))
      for(count in 1:(length(gam.curves)-1)){
        gam.x <- as.numeric(gam.curves[[count]]$x)
        gam.y <- as.numeric(gam.curves[[count]]$fit)
        df.var <- data.frame(gam.x, gam.y)
        colnames(df.var) <- c(paste0("x.", varlist.curves[count]), paste0("y.", varlist.curves[count]))
        df.gam <- cbind(df.gam, df.var)
        #colnames(df.sddr)[ncol(df.sddr)] <- paste0("y", count)
      }
      df.gam <- df.gam[,-1]
  
      # summary of data sets for partial effect curves comparison
      dim(df.sddr)
      names(df.sddr)
      dim(df.sddr.sd)
      names(df.sddr.sd)
      dim(df.gam)
      names(df.gam)

      ################################################################################################
      ### create plots with partial effects  #########################################################
      ################################################################################################
      
      # define counter for loop for test purposes
      count = 1
      
      for(count in 1:len(varlist.curves)){
      
      # create common data frame
      df.sddr.small <- df.sddr[, c(2*count-1, 2*count)]
      df.sddr.small$Model = "SDDR (1)"
      df.sddr.sd.small <- df.sddr.sd[, c(2*count-1, 2*count)]
      df.sddr.sd.small$Model = "SDDR (2)"
      df.gam.small <- df.gam[, c(2*count-1, 2*count)]
      df.gam.small$Model <- "GAM"
      df.small <- rbind(df.sddr.small, df.sddr.sd.small, df.gam.small)
      
      
      p = ggplot() + 
        geom_line(data = df.small, 
                  aes_string(x = paste0("x.", varlist.curves[count]), y = paste0("y.", varlist.curves[count]), 
                             color = "Model"), size = 1.5)  +
        xlab(paste0(varlist.curves[count])) +
        ylab('Partial Effect') +
        scale_colour_manual(values=c("red","blue", "green")) +
        theme_light(base_size = 30, base_line_size = 12/18)
      
      # initialize  plot
      png(file = file.path(temp.data, paste0("effect-curves-", spec,"-", varlist.curves[count], ".png")), 
          width = 1000, height = 800, pointsize = 35)
      
      # display plot
      print(p)
      
      #save plot
      dev.off()
      
      # delay function for avoid error while saving plots
      delay <- function(x)
      {
        p1 <- proc.time()
        Sys.sleep(x)
        proc.time() - p1 # The cpu usage should be negligible
      }
      #delay(1)
      
      # alternative saving route
      #ggsave(file.path(temp.data, paste0("effect-curves-pmeg-", varlist.curves[count], ".png")))
      
      # end loop over smooth effects
    }
