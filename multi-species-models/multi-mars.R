
#################################################################################################
## set up workspace
#################################################################################################

rm(list=ls())

library(devtools)

# load required packages
## install.packages("pacman")
pacman::p_load(Matrix, dplyr, keras, mgcv, reticulate, tensorflow, tfprobability, 
               Metrics, DescTools, MLmetrics, caret, xgboost, recipes, yardstick,
               ParBayesianOptimization, doParallel, muStat, scoring, Hmisc, blockCV, earth, UBL)

# force conda environment
#use_condaenv("r-reticulate", required = T)

# directories
main.path <- dirname(rstudioapi::getSourceEditorContext()$path)
repo.path <- file.path(main.path, "repo", "deepregression-master")
temp.data <- file.path(main.path, "temp")

# load deepregression package
#load_all(repo.path)

# seed
seed = 42
tf$random$set_seed(seed = seed)
set.seed(seed)

# define timestamp to identify saved files
timestamp <- format(Sys.time(), "%Y-%m-%d-%H%M")

# set range for degree of interaction (3 or 4)
doirange <- 1:4

# set subsampling type (none, lpros, lprus, adasyn)
subsampling <- "none"
#subsampling <- "lpros"
#subsampling <- "lprus"
#subsampling <- "adasyn"


# load list of prepared objects
full.list <- readRDS(file = file.path(main.path, "data", "full-model-list.rds"))


#for(imbalance.method in c("none", "lprus", "lpros", "adasyn")){
#  subsampling <- imbalance.method

################################################################################################
################## train and evaluate MMARS on species-specific data sets ######################
################################################################################################

insample.species <- c("Triatoma infestans", "Triatoma dimidiata", "Panstrongylus megistus",
                      "Triatoma brasiliensis", "Triatoma sordida", "Triatoma pseudomaculata",
                      "Triatoma barberi")
mars.tot <- NULL

# loop over degrees of interaction
for(doi in doirange){
  
 mars.aucs <- NULL
 mars.briers <- NULL

 # loop over insample species
 for(spec in insample.species){
  
   ## prepare species-specific data sets
  
   # load list of prepared objects
   #full.list <- readRDS(file = file.path(data.path, "full-model-list.rds"))
  
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
  
   # estimate multi-response MARS
   multi.mars <- suppressWarnings(earth(x = x.train.s, y = y.train, degree = doi, glm=list(family=binomial)) )
   # predict test set
   pred <- predict(object = multi.mars, newdata = x.test.s, type = "response")
   colnames(pred) <- insample.species
   
   # compute AUC and Brier score on test data of species
   auc.spec <- AUC(y_pred = pred[,spec], y_true = y.test[,spec])
   brier.spec <- BrierScore(resp = y.test[,spec], pred = pred[,spec])
   
   cat("AUC for this species and doi =", doi, "is = ", auc.spec, "\n")
   cat("Brier Score for this species and doi =", doi, "is = ", brier.spec, "\n")
   
   mars.aucs <- cbind(mars.aucs, auc.spec)
   mars.briers <- cbind(mars.briers, brier.spec)
  
  # end loop over species 
  }

 # summarise results for specific doi
 colnames(mars.aucs) <- insample.species
 mars.aucs <- data.frame(mars.aucs)
 colnames(mars.briers) <- insample.species
 mars.briers <- data.frame(mars.briers)

 mars.res.doi <- rbind(mars.aucs, mars.briers)
 rownames(mars.res.doi) <- c(paste0("AUC", doi), paste0("Brier", doi))
 mars.res.doi$doi <- doi
 mars.res.doi

 mars.tot <- rbind(mars.tot, mars.res.doi)

# end loop over degrees of interaction
}

# rearrange rows for legibility
if(len(doirange) == 4){mars.tot <- mars.tot[c(1,3,5,7,2,4,6,8),]} else{mars.tot <- mars.tot[c(1,3,5,2,4,6),]}

View(mars.tot)

saveRDS(mars.tot, file = file.path(temp.data, paste0("results-MMARS-doi-",  timestamp, ".rds")))












