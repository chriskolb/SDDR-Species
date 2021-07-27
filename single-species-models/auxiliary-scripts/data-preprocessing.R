################################################################################################
##################### pre-processing and preparation of data of one species ####################
################################################################################################

# input: list of two named data, containing data frames train and test from folds[[species]]
# output: (x.train, x.train.s, y.train), (x.test, x.test.s, y.test), spat.cv.indices
# suffix .s indicates standardization, i.e. for each covariate: 
# x_new = (x_old - mean(train))/sd(train)

################################################################################################

# seed
seed = 42
set.seed(seed)

# get train and test data frames
train        <- data$train
train <- train %>% filter(TRUE)
train.orig   <- train
test         <- data$test
test <- test %>% filter(TRUE)
test.orig    <- data$test





# define covariate list
varlist <- c("longitude", "latitude", 
             "rainfall", "elevation", "slope",
             "lst_day", "lst_night", "lst_diff", #land surface temperature
             "reflectance_tcb", "reflectance_tcw", "reflectance_evi", # tasseled cap + vegetation
             "urban", "nighttimelights", "population", "accessibility", # urbanity variables
             "landcover00", "landcover01", "landcover02", "landcover03",
             "landcover04", "landcover05", "landcover06", "landcover07", 
             "landcover08", "landcover09", "landcover10", "landcover11",
             "landcover12", "landcover13", "landcover14",
             "landcover15", "landcover16", "landcover17",
             "landcover18"
             )


reduced.landcover.train <- train[names(train) %in% varlist[16:34]]
reduced.landcover.train <- reduced.landcover.train %>% select_if(function(col) n_distinct(col) >= min.unique)

reduced.landcover.test <- test[names(test) %in% varlist[16:34]]
reduced.landcover.test <- reduced.landcover.test %>% select_if(function(col) n_distinct(col) >= min.unique)

reduced.landcover.names <- intersect(names(reduced.landcover.train), names(reduced.landcover.test))
reduced.landcover.names

varlist <- c(varlist[1:15], reduced.landcover.names)


# reduce data frame
train <- train[names(train) %in% c(varlist, "fold", "presence")]
train <- train[complete.cases(train),]
test <- test[names(test) %in% c(varlist, "fold", "presence")]
test <- test[complete.cases(test),]

# randomly upsample presence locations to balance classes for more stable optimization
summary(train$presence)
train <- upSample(x = train[, -(ncol(train)-1)], # dep. var. "presence" is (p-2)th variable 
                  y = as.factor(train$presence), 
                  list = FALSE,
                  yname = "presence"
                  )

train$presence <- as.numeric(train$presence)-1
summary(train$presence)

# shuffle training data so 0s and 1s are not clustered
set.seed(seed)
train <- train[sample(nrow(train)),]


# build spatial cv indices for train data for cv_folds option
obs.indices <- 1:nrow(train)
fold.var <- train$fold
fold1.ind <- obs.indices[train$fold==1]
fold2.ind <- obs.indices[train$fold==2] 
fold3.ind <- obs.indices[train$fold==3] 
fold4.ind <- obs.indices[train$fold==4] 

spat.cv.indices <- list(
  fold1 = list(
            train = c(fold2.ind, fold3.ind, fold4.ind),
            val   = fold1.ind
  ),
  fold2 = list(
            train = c(fold1.ind, fold3.ind, fold4.ind),
            val   = fold2.ind
  ),
  fold3 = list(
            train = c(fold1.ind,fold2.ind,fold4.ind),
            val   = fold3.ind
  ),
  fold4 = list(
            train = c(fold1.ind,fold2.ind,fold3.ind),
            val   = fold4.ind
  )
  
  )

rm(obs.indices, fold1.ind, fold2.ind, fold3.ind, fold4.ind)

# name y variable
y.train <- as.integer(train$presence)
y.test <- as.integer(test$presence)

# create standardized features
# test data must be standardized same as training data: use recipes package

recipe.formula <- formula(paste("presence ~ ", paste(varlist, collapse = " + ")))

# create pre-processing recipe
  rec_obj <- recipe(recipe.formula, data = train) %>%
    step_center(all_predictors(), -all_outcomes() ) %>%
    step_scale(all_predictors(), -all_outcomes() ) %>%
    prep(data = train)

# bake train data according to train data parameters
train.s <- bake(rec_obj, new_data = train)
x.train.s <- train.s[, 1:(ncol(train.s)-1) ]
#x.train.s$fold <- train$fold
x.train <- train[, 1:(ncol(train)-2) ]

# bake test data according to train data standardization
test.s <- bake(rec_obj, new_data = test)
x.test.s <- test.s[, 1:(ncol(test.s)-1) ]
x.test <- test[, 1:(ncol(test)-2) ]



# print result of data prep

cat("######################################################################################\n")
cat("########################### Pre-processing was successful!############################\n")
cat("######################################################################################\n", "\n")

cat(        " species: ", species, "\n",
            "train obs = ", nrow(train.orig), "; ",
            "test obs  = ", length(y.test), "; ",
            "upSampled train obs = ", nrow(x.train), "\n",
            "number of covariates = ", ncol(x.train), "\n",
            "number of spat cv. folds = ", length(spat.cv.indices), "\n",
            "returned objects :\n", 
            "(x.train, x.train.s, y.train), (x.test, x.test.s, y.test), spatial cv indices", "\n",
            "suffix .s indicates standardization (m=0, s=1)", "\n"
            )

cat("\n######################################################################################\n")


# remove auxiliary objects
rm(reduced.landcover.names, reduced.landcover.train, 
   reduced.landcover.test, test, test.s, train, train.s  
   )



