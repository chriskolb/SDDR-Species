################################################################################################
########## pre-processing and preparation of species-specific full data set ####################
################################################################################################

# input: list of two named data, containing data frames train and test 
# output: (x.train, x.train.s, y.train), (x.test, x.test.s, y.test)
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
train <- train[names(train) %in% c(varlist, "newfold", "presence", "spec", "species")]
train <- train[complete.cases(train),]
test <- test[names(test) %in% c(varlist, "newfold", "presence", "spec", "species")]
test <- test[complete.cases(test),]


################################################################################################
### dealing with class imbalance ### comment out if not desired

pre.labels <- train$spec
levels(pre.labels) <- c(levels(pre.labels), "none")
pre.labels[train$presence == 0] <- "none"
train$auxlabels <- pre.labels
summary(train$auxlabels)

pre.labels.test <- test$spec
levels(pre.labels.test) <- c(levels(pre.labels.test), "none")
pre.labels.test[test$presence == 0] <- "none"
test$auxlabels <- pre.labels.test
summary(test$auxlabels)

for(element in insample.species){
  sp <- paste(element)
  test$auxlabels[as.character(test$auxlabels) == "none" & as.character(test$species) == sp] <- sp
}
summary(test$auxlabels)

# keep only insample observations (identical to include.tgb == "no")
#train <- train[train$species %in% insample.species,]
#test <- test[test$species %in% insample.species,]

if(subsampling != "none"){

## using ADASYN (SMOTE-like) approach to oversample label-power set
if(subsampling == "adasyn"){
train$species <- NULL
train.adasyn <- train
train.adasyn$presence <- as.factor(train.adasyn$presence)
train.adasyn <- data.frame(train.adasyn)
train.adasyn <- within(train.adasyn, rm(presence, spec))
names(train.adasyn)
#str(train.adasyn)
#adasyn.form <- formula(paste0("auxlabels~", paste(names(train.adasyn)[1:(len(train.adasyn)-4)], collapse="+")))
adasyn.form <- formula("auxlabels~.")
adasyn.form

set.seed(seed)
train.adasyn <- AdasynClassif(adasyn.form, train.adasyn, baseClass = NULL, beta = 1, k = 4) # dist = "HEOM"

}

## using simple random oversampling of label-power set
if(subsampling == "lpros"){
set.seed(seed)
train.lpros <- upSample(x = within(train, rm(auxlabels)), 
                  y = train$auxlabels, 
                  list = FALSE,
                  yname = "auxlabels"
                  )
}


### using mixed random downsampling/upsampling of label-power set ###
# downsample species with more than 500 observations
# upsample species with less than 500 observations
if(subsampling == "lprus"){
train.lprus <- train
counts <- data.frame(table(train.lprus$auxlabels))
colnames(counts) <- c("spec", "freq")
counts$spec <- as.character(counts$spec)
counts

down.specs <- counts$spec[counts$freq>500]
down.lprus <- train.lprus[train.lprus$auxlabels %in% down.specs,]
down.lprus$auxlabels <- droplevels(down.lprus$auxlabels)
set.seed(seed)
down.lprus <- downSample(x = within(down.lprus, rm(auxlabels)), y = down.lprus$auxlabels, yname = "auxlabels")

up.specs <- counts$spec[counts$freq<500]
up.lprus <- train.lprus[train.lprus$auxlabels %in% up.specs,]
up.lprus$auxlabels <- droplevels(up.lprus$auxlabels)

train.lprus <- rbind(down.lprus, up.lprus)
train.lprus$auxlabels <- factor(train.lprus$auxlabels, levels = c(insample.species, "none"))
set.seed(seed)
train.lprus <- upSample(x = within(train.lprus, rm(auxlabels)), y = train.lprus$auxlabels, yname = "auxlabels")
table(train.lprus$auxlabels)

}

# define subsampling method used in estimation
 if(subsampling == "adasyn"){train = train.adasyn} else 
  if(subsampling == "lpros"){train = train.lpros} else 
    if(subsampling == "lprus"){train = train.lprus} 
  else {cat("Class imbalance will be ignored, subsampling = ",subsampling, ".\n")}
  

}


cat(table(train$auxlabels))
cat("\n")
cat(table(test$auxlabels))
cat("\n")

################################################################################################


# shuffle training data so 0s and 1s are not clustered
set.seed(seed)
train <- train[sample(nrow(train)),]

#train.specs <- train$spec
#test.specs <- test$spec
#train.insamples <- train$insample
#test.insamples <- test$insample
#train$insample <- NULL
#test$insample <- NULL

# construct response matrix
y.train.uni <- as.integer(train$auxlabels != "none")
train$presence <- y.train.uni
y.test.uni <- as.integer(test$presence)

train.specs.int <- as.integer(train$auxlabels)-1L
y.train.onehot <- to_categorical(train.specs.int, num_classes = len(levels(train$auxlabels)))
y.train <- y.train.onehot[,1:len(insample.species)]
colnames(y.train) <- insample.species
colnames(y.train.onehot) <- levels(train$auxlabels)

#train.specs.int <- as.integer(train.specs)-1
#train.specs.int <- as.integer(train.specs.int)
#y.train <- to_categorical(train.specs.int, num_classes = 7)
#y.train[y.train.uni == 0,] <- rep(0,7)
#colnames(y.train) <- insample.species

test.specs.int <- as.integer(test$auxlabels)-1L
y.test.onehot <- to_categorical(test.specs.int, num_classes = len(levels(test$auxlabels)))
y.test <- y.test.onehot[,1:len(insample.species)]
colnames(y.test) <- insample.species
colnames(y.test.onehot) <- levels(test$auxlabels)


# compare to presence points to check consistency
colSums(y.train)
colSums(y.train.onehot)
table(train$auxlabels[y.train.uni == 1])
colSums(y.test)
colSums(y.test.onehot)
table(test$auxlabels[y.test.uni == 1])

# save auxlevel factors and species just in case
train.auxlabels <- train$auxlabels
test.auxlabels <- test$auxlabels
if(subsampling != "adasyn"){
train.species <- train$species
train$species <- NULL
test.species <- test$species
test$species <- NULL
}
train.specs <- train$spec
test.specs <- test$spec


# create standardized features
# test data must be standardized same as training data: use recipes package

recipe.formula <- formula(paste("presence ~ ", paste(varlist, collapse = " + ") )) # " + spec"

# create pre-processing recipe
rec_obj <- recipe(recipe.formula, data = train) %>%
  #step_dummy(spec, one_hot = F) %>%
  step_center(all_predictors(), -all_outcomes() ) %>%
  step_scale(all_predictors(), -all_outcomes() ) %>%
  prep(data = train)

# bake train data according to train data parameters
train.s <- bake(rec_obj, new_data = train)

# put presence column last (dirty)
train$presence <- NULL
train$presence <- y.train.uni
train.s$presence <- NULL
train.s$presence <- y.train.uni

x.train.s <- train.s[, 1:(ncol(train.s)-1) ]
#x.train.s$fold <- train$fold
x.train <- train[, 1:(ncol(train)-1) ]

# bake test data according to train data standardization
test.s <- bake(rec_obj, new_data = test)
test.s$presence <- NULL
test.s$presence <- y.test
test$presence <- NULL
test$presence <- y.test
x.test.s <- test.s[, 1:(ncol(test.s)-1) ]
x.test <- test[, 1:(ncol(test)-1) ]



# print result of data prep

cat("######################################################################################\n")
cat("########################### Pre-processing was successful!############################\n")
cat("######################################################################################\n", "\n")

cat(" species = ", spec, "\n", 
  "subsampling approach = ", subsampling, "\n",
  "train obs = ", nrow(train.orig), "; ",
  "test obs  = ", nrow(y.test), "; ",
  "processed train obs = ", nrow(x.train.s), "\n",
  "number of covariates = ", ncol(x.train.s), "\n",
  "returned objects :\n", 
  "(x.train, x.train.s, y.train), (x.test, x.test.s, y.test), train/testfreqs/specs", 
  "\n",
  "suffix .s indicates standardization (m=0, s=1)", "\n"
)

cat("\n######################################################################################\n")


# remove auxiliary objects
rm(rec_obj, recipe.formula, reduced.landcover.names, reduced.landcover.train, 
   reduced.landcover.test, test.s, train.s, sp, pre.labels, pre.labels.test
)
