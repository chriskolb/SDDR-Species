################################################################################################
## set up workspace and define parameters
################################################################################################

rm(list=ls())

library(devtools)

# load deepregression and spatial packages
## install.packages("pacman")
pacman::p_load(Matrix, dplyr, keras, mgcv, reticulate, tensorflow, tfprobability, 
               # own packages for hyperparam tuning and model comparison
               Metrics, MLmetrics, caret, xgboost, recipes, yardstick, doParallel, 
               ParBayesianOptimization, muStat, scoring, Hmisc, ggvis, googlesheets,
               # spatial packages
               raster, tmaptools, stringr, tmap, purrr, sp, magrittr, sf, rgeos,
               tidyr, tibble, dismo, pbapply, geosphere, spatstat, pammtools, MLmetrics,
               ggplot2, caret, blockCV, fields, furrr, rlang, rsample, automap, pals,
               stars, classInt, Rcpp
               )


# force conda environment
use_condaenv("r-reticulate")

# data frame directory
main_path <- dirname(rstudioapi::getSourceEditorContext()$path)
data_path <- file.path(main_path, "data")

# temp data directory
temp.path <- file.path(main_path, "temp")

# out directory
out.path <- temp.path

# seed
seed = 42
tf$random$set_seed(seed = seed)
set.seed(seed)

################################################################################################
## load and prepare data
################################################################################################

## country borders within endemic zone
countries <- readRDS(file.path(data_path, "raw-data", "countries.Rds"))

## main data object that contains data for all species
# ignore all that have '(other)' in the name
# probably best to select subset or focus on one species in the beginning

# list with elements
# - $species: name of the species
# - $hull: spatial polygon that defines the spatial extent investigated
# for that species
# - $blocks: spatial polygons used to define obsevations used for train/test
# when performing spatial cross validation
# - $extent: spatial extent of the extended hull (can be used to subset other spatial objects)
# - $train: the actual data (folds 1 - 4 of the spatial CV)
# - $test: the actual data (fold 5 of the spatial CV)
# - $block_width, vario, vario_ragnge: can be ignored

folds <- readRDS(file.path(data_path, "raw-data" , "folds_list_vector_presence.Rds"))
str(folds, 1)
names(folds)

################################################################################################
############################ Combine single species data into one df ###########################
################################################################################################

species.list <- c("tinfestans", "tdimidiata", "pmegistus", "tbrasiliensis", "tsordida", 
                  "tpseudomaculata", "tbarberi")

insample.species <- c("Triatoma infestans", "Triatoma dimidiata", "Panstrongylus megistus",
                      "Triatoma brasiliensis", "Triatoma sordida", "Triatoma pseudomaculata",
                      "Triatoma barberi")

preds.list <- c("smooth", "smooth-deep", "deep")

train.full <- NULL
test.full <- NULL

for (spec in insample.species){
  # load data
  data <- folds[[spec]]
  assign(gsub(" ", "", spec, fixed = TRUE), data)
  train <- data$train
  train$spec <- rep(paste(spec), nrow(train))
  train$type <- rep(paste("train"), nrow(train))
  train <- as.data.frame(train)
  test <- data$test
  test$spec <- rep(paste(spec), nrow(test))
  test$type <- rep(paste("test"), nrow(test))
  test <- as.data.frame(test)
  # append data
  train.full <- rbind(train.full, train)
  test.full <- rbind(test.full, test)
  rm(train, test)
}

# extract presence observations of all species (1x orrurence)
# use all the other species (besides our 7) background
df.full <- rbind(train.full, test.full)

# 0 = not in 7 selected species, "true" background points 
# 1 = insample presence point, 
# 2 = insample obs of one of 7 species that was used as background in other species data
df.full$insample <- 0
df.full$insample[df.full$species %in% insample.species & df.full$presence == 1] <- 1
df.full$insample[df.full$species %in% insample.species & df.full$presence == 0] <- 2

table(df.full$insample)

# create table that makes clear where observations come from
obs.origin <- matrix(rep(NA, 8*10), nrow = 10, ncol = 8)
colnames(obs.origin) <- c(insample.species, "sum")
rownames(obs.origin) <- c("presence", "inventory T.inf.", "inventory T.dimi.", "inventory P.meg.",
                         "inventory T.bras.", "inventory T.sord.", "inventory T.pseudo.", 
                         "inventory T.barb.", "other TGA", "sum")
i = 1
for (sp in insample.species){
  # presence points (insample points)
  obs.origin[1,i] <- nrow(df.full[df.full$spec == sp & df.full$presence == 1,])
  # inventory absences (insample points)
  obs.origin[2,i] <- nrow(df.full[df.full$spec == sp & df.full$presence == 0 & df.full$species == "Triatoma infestans",])
  obs.origin[3,i] <- nrow(df.full[df.full$spec == sp & df.full$presence == 0 & df.full$species == "Triatoma dimidiata",])
  obs.origin[4,i] <- nrow(df.full[df.full$spec == sp & df.full$presence == 0 & df.full$species == "Panstrongylus megistus",])
  obs.origin[5,i] <- nrow(df.full[df.full$spec == sp & df.full$presence == 0 & df.full$species == "Triatoma brasiliensis",])
  obs.origin[6,i] <- nrow(df.full[df.full$spec == sp & df.full$presence == 0 & df.full$species == "Triatoma sordida",])
  obs.origin[7,i] <- nrow(df.full[df.full$spec == sp & df.full$presence == 0 & df.full$species == "Triatoma pseudomaculata",])
  obs.origin[8,i] <- nrow(df.full[df.full$spec == sp & df.full$presence == 0 & df.full$species == "Triatoma barberi",])
  # target group absences (outsample species)
  obs.origin[9,i] <- nrow(df.full[df.full$spec == sp & df.full$presence == 0 & df.full$species %nin% insample.species,])
  # sum of previous row entries should equal number of observations in species-specific data set
  obs.origin[10,i] <- sum(obs.origin[1:9,i])
  cat(obs.origin[10,i] == nrow(df.full[df.full$spec == sp,]), "\n")
  i = i + 1
}
obs.origin[,8] <- rowSums(obs.origin, na.rm = TRUE)

View(obs.origin)


################################################################################################
############### Systematic exploration of presence-points, duplicates and absences #############
################################################################################################

### composition of df.full

# 11300 unique IDs in dataset of 30460 obs
dupes <- as.data.frame(table(df.full$id))
dupes <- dupes %>% transmute(id = as.integer(as.character(Var1)), totduplis = Freq)
head(dupes)

# merge ID frequency to df.full
df.full <- left_join(df.full, dupes)
# in total 2905/1 + 8022/2 + 1557/3 + 5396/4 + 12580/5 = 11300 unique points in df
table(df.full$totduplis)

# 7541 unique presence points (with 7541 unique IDs)
presence.ids <- df.full$id[df.full$presence==1]
# of 7541 presence IDs, 1754 appear once (point in one hull), 2259 points twice (two ext. hulls) etc.
table(dupes$totduplis[dupes$id %in% presence.ids])
# of 3759 unique TGA IDs, 1151 appear once (in one hull), 1752 points twice (in two hulls), etc.
table(dupes$totduplis[dupes$id %nin% presence.ids])
# 11300 unique IDS = 7541 presence IDs + 3759 unique TGA IDs
# presence points get duplicated as inventory absences, #times = #intersecting hulls in point loc?
# unique TGA points get duplicated as absences of specific species, #times = #intersecting hulls

# generate duplicate counter by id variable
df.full$number <- 1
df.full <- df.full %>% group_by(id) %>% mutate(duplicounter = cumsum(number))
table(df.full$duplicounter)
df.full$number <- NULL

# Load crosstab function
source("http://pcwww.liv.ac.uk/~william/R/crosstab.r")
# crosstab repo taken from remotes::install_github("lehmansociology/lehmansociology")

# table with insample status by original species-specific data set and train/test split
tab.insample <- crosstab(df.full, row.vars = c("spec", "type"), col.vars = c("insample"), 
                    type = "f", addmargins = TRUE)
tab.insample

tab.totduplis <- crosstab(df.full, row.vars = c("spec", "type"), col.vars = c("totduplis"), 
                         type = "f", addmargins = TRUE)
tab.totduplis

tab.dupes <- crosstab(df.full, row.vars = c("duplicounter"), col.vars = c("totduplis"), 
                          type = "f", addmargins = TRUE)
tab.dupes

tab.insample.dupes <- crosstab(df.full, row.vars = c("totduplis"), col.vars = c("insample"), 
                      type = "f", addmargins = TRUE)
tab.insample.dupes 

tab.presence <- crosstab(df.full, row.vars = c("spec", "insample"), col.vars = c("presence"), 
                               type = "f", addmargins = TRUE)
tab.presence 

####

# re-define train and test sets with additional variables
train.full <- df.full[df.full$type == "train",]
test.full <- df.full[df.full$type == "test",]


### train set composition (not interesting, for completeness and data exploration/understaning)

# train set specific duplicate count
tr.dupes <- as.data.frame(table(train.full$id))
tr.dupes <- tr.dupes %>% transmute(id = as.integer(as.character(Var1)), trainduplis = Freq)
head(tr.dupes)

# merge ID frequency to train.full
train.full <- left_join(train.full, tr.dupes)
# in total 4525/1 + 5552/2 + 3864/3 + 5644/4 + 4570/5 = 10914 unique points in df
table(train.full$trainduplis)

# 6159 unique presence points (with 6159 unique IDs)
tr.presence.ids <- train.full$id[train.full$presence==1]
# of 6159 presence IDs, 2129 appear once (point in one hull), 1519 points twice (two ext. hulls) etc.
table(tr.dupes$trainduplis[tr.dupes$id %in% tr.presence.ids])
# of 4755 unique TGA IDs, 2396 appear once (in one hull), 1257 points twice (in two hulls), etc.
table(tr.dupes$trainduplis[tr.dupes$id %nin% tr.presence.ids])

# generate duplicate counter by id variable
train.full$number <- 1
train.full <- train.full %>% group_by(id) %>% mutate(trainduplicounter = cumsum(number))
table(train.full$trainduplicounter)
train.full$number <- NULL

tab.train.insample <- crosstab(train.full, row.vars = c("spec"), col.vars = c("insample"), 
                              type = "f", addmargins = TRUE)
tab.train.insample

tab.train.duplis <- crosstab(train.full, row.vars = c("spec"), col.vars = c("trainduplis"), 
                            type = "f", addmargins = TRUE)
tab.train.duplis

tab.train.insample.dupes <- crosstab(train.full, row.vars = c("trainduplis"), col.vars = c("insample"), 
                                    type = "f", addmargins = TRUE)
tab.train.insample.dupes 

tab.train.presence <- crosstab(train.full, row.vars = c("spec", "insample"), col.vars = c("presence"), 
                         type = "f", addmargins = TRUE)
tab.train.presence 


### test set composition (obtain unique test set ids to exclude from train set)


# test set specific duplicate count
te.dupes <- as.data.frame(table(test.full$id))
te.dupes <- te.dupes %>% transmute(id = as.integer(as.character(Var1)), testduplis = Freq)
head(te.dupes)
unique.test.ids <- te.dupes$id
len(unique.test.ids)

# merge ID frequency to test.full
test.full <- left_join(test.full, te.dupes)
# in total 3681/1 + 1336/2 + 1056/3 + 232/4 = 4759 unique points in df
table(test.full$testduplis)

# 1382 unique presence points (with 1382 unique IDs)
te.presence.ids <- test.full$id[test.full$presence==1]
# of 1382 presence IDs, 939 appear once (point in one hull), 234 points twice (two ext. hulls) etc.
table(te.dupes$testduplis[te.dupes$id %in% te.presence.ids])
# of 3377 unique TGA IDs, 2742 appear once (in one hull), 434 points twice (in two hulls), etc.
table(te.dupes$testduplis[te.dupes$id %nin% te.presence.ids])

# generate duplicate counter by id variable
test.full$number <- 1
test.full <- test.full %>% group_by(id) %>% mutate(testduplicounter = cumsum(number))
table(test.full$testduplicounter)
test.full$number <- NULL

# crosstab tables
tab.test.insample <- crosstab(test.full, row.vars = c("spec"), col.vars = c("insample"), 
                              type = "f", addmargins = TRUE)
tab.test.insample

tab.test.duplis <- crosstab(test.full, row.vars = c("spec"), col.vars = c("testduplis"), 
                               type = "f", addmargins = TRUE)
tab.test.duplis

tab.test.insample.dupes <- crosstab(test.full, row.vars = c("testduplis"), col.vars = c("insample"), 
                                    type = "f", addmargins = TRUE)
tab.test.insample.dupes 

tab.test.presence <- crosstab(test.full, row.vars = c("spec", "insample"), col.vars = c("presence"), 
                               type = "f", addmargins = TRUE)
tab.test.presence 


################################################################################################
############### Creation of tuning data set independent of any test set points  ###############
################################################################################################

# first, obtain vector unique.test.ids (above) for all observations in test.full
# then, exclude from df.full (or train.full, indiff.) all obs with point id in unique.test.ids
# left with all points with ids that never appear in any insample species test set

### tune set composition

# this excludes 1) test set observations 2) duplicates of test set points in train data
df.tune <- df.full[df.full$id %nin% unique.test.ids,]

# tune (train) set specific duplicate count
tune.dupes <- as.data.frame(table(df.tune$id))
tune.dupes <- tune.dupes %>% transmute(id = as.integer(as.character(Var1)), tuneduplis = Freq)
head(tune.dupes)
unique.tune.ids <- tune.dupes$id
len(unique.tune.ids)

# merge ID frequency to df.tune
df.tune <- left_join(df.tune, tune.dupes)
# in total 2580/1 + 4414/2 + 780/3 + 2320/4 + 4570/5 = 6541 unique points in df
table(df.tune$tuneduplis)

# 4365 unique presence points (with 4365 unique IDs)
tune.presence.ids <- df.tune$id[df.tune$presence==1]
len(tune.presence.ids)
# of 4365 presence IDs, 1602 appear once (point in one hull), 1297 points twice (two ext. hulls) etc.
table(tune.dupes$tuneduplis[tune.dupes$id %in% tune.presence.ids])
# of 2176 unique TGA IDs, 978 appear once (in one hull), 910 points twice (in two hulls), etc.
table(tune.dupes$tuneduplis[tune.dupes$id %nin% tune.presence.ids])

# generate duplicate counter by id variable
df.tune$number <- 1
df.tune <- df.tune %>% group_by(id) %>% mutate(tuneduplicounter = cumsum(number))
table(df.tune$tuneduplicounter)
df.tune$number <- NULL

### summarize df.tune

summary(df.tune$presence)
table(df.tune$presence)
table(df.tune$insample) # 4028 TGA, 4365 insample presences and 6271 inventory duplicates (insample absences)

tab.tune.insample <- crosstab(df.tune, row.vars = c("spec"), col.vars = c("insample"), 
                              type = "f", addmargins = TRUE)
tab.tune.insample

tab.tune.totduplis <- crosstab(df.tune, row.vars = c("spec"), col.vars = c("tuneduplis"), 
                               type = "f", addmargins = TRUE)
tab.tune.totduplis

tab.tune.insample.dupes <- crosstab(df.tune, row.vars = c("tuneduplis"), col.vars = c("insample"), 
                                    type = "f", addmargins = TRUE)
tab.tune.insample.dupes 

tab.tune.presence <- crosstab(df.tune, row.vars = c("spec", "insample"), col.vars = c("presence"), 
                              type = "f", addmargins = TRUE)
tab.tune.presence 

# tab.test.insample and tab.tune.insample do not add up to all obs! 
# This is because additional obs were removed from train set

################################################################################################
################ Generate species-specific multi-species data sets (larger) ####################
################################################################################################

### generate spec-specific train sets as df.full less all point ids in unique.spec.test.ids

for (spec in insample.species){
  
  # get species-specific test set
  test.spec <- test.full[test.full$spec == spec,]
  
  # count numobs per point id variable
  spec.dupes <- as.data.frame(table(test.spec$id))
  spec.dupes <- spec.dupes %>% transmute(id = as.integer(as.character(Var1)), spectestduplis = Freq)
  table(spec.dupes$spectestduplis)
  
  # get unique test set ids from insample species
  unique.spec.test.ids <- spec.dupes$id
  num.unique.test.ids <- len(unique.spec.test.ids)
  #assign(paste0("num.test.ids.", gsub(" ", "", spec, fixed = TRUE)), num.unique.test.ids)
  id.name <- paste0("unique.test.ids.", gsub(" ", "", spec, fixed = TRUE))
  assign(id.name, unique.spec.test.ids)
  
  # define insample species train set as df.full w/o unique.spec.test.ids
  train.spec <- df.full[df.full$id %nin% unique.spec.test.ids,]
  spec.dupes.tr <- as.data.frame(table(train.spec$id))
  spec.dupes.tr <- spec.dupes.tr %>% transmute(id = as.integer(as.character(Var1)), spectrainduplis = Freq)
  table(spec.dupes.tr$spectrainduplis)
  unique.spec.train.ids <- spec.dupes.tr$id
  assign(paste0("unique.train.ids.", gsub(" ", "", spec, fixed = TRUE)), unique.spec.train.ids)
  num.unique.train.ids <- len(unique.spec.train.ids)
  #assign(paste0("num.train.ids.", gsub(" ", "", spec, fixed = TRUE)), num.unique.train.ids)
  train.name <- paste0("train.full.", gsub(" ", "", spec, fixed = TRUE))
  assign(train.name, train.spec)
  
  cat(" Species:", spec, "\n",
      "Full-model train set has", num.unique.train.ids, "unique training points and",
      num.unique.test.ids, "unique test points.\n",
      "Total train obs = ", nrow(train.spec),"; total test obs = ", nrow(test.spec), "\n",
      "Train insample table:(", table(train.spec$insample), "); test table:(", table(test.spec$insample), ") (0,1,2)\n",
      "Train presence:(", table(train.spec$presence), "); test presence:(", table(test.spec$presence), ") (0,1)\n.", "\n")

  rm(test.spec, spec.dupes, unique.spec.test.ids, num.unique.test.ids, id.name, 
     train.spec, spec.dupes.tr, unique.spec.train.ids, num.unique.train.ids, train.name)
  
  }


################################################################################################
############## Generate variant data sets for different pseudo-absence selection ###############
################################################################################################

# 1) reduce df.tune and train.full.spec to insample values < 2 (no inventory absences)
# 2) afterwards, remove duplicated true TGA observations

### tuning data set

df.tune.var <- df.tune[df.tune$insample<2,]

# generate duplicate counter by id variable
df.tune.var$number <- 1
df.tune.var <- df.tune.var %>% group_by(id) %>% mutate(varduplicounter = cumsum(number))
table(df.tune.var$varduplicounter)
df.tune.var$number <- NULL

# all remaining duplicates are in duplicates of true TGA
crosstab(df.tune.var, row.vars = c("insample"), col.vars = c("varduplicounter"), 
                            type = "f", addmargins = TRUE)

# remove duplicates
df.tune.var <- df.tune.var[df.tune.var$varduplicounter == 1,]
table(df.tune.var$presence)
table(df.tune.var$insample)

# tune.var (train) set specific duplicate count
tune.var.dupes <- as.data.frame(table(df.tune.var$id))
tune.var.dupes <- tune.var.dupes %>% transmute(id = as.integer(as.character(Var1)), tuneduplis = Freq)
head(tune.var.dupes)
unique.tune.var.ids <- tune.var.dupes$id
len(unique.tune.var.ids)

cat(" Variant Tuning training set... \n",
    "variant: no duplicates whatsoever \n",
    "variant train obs:", nrow(df.tune.var), "\n",
    "variant presence-absence: ", table(df.tune.var$presence), "\n",
    "mean prevalence in variant data set is ", round(mean(df.tune.var$presence),2), "\n.\n"
     )


### species-specific full-model data sets

for(spec in insample.species){
  train.var.spec <- eval(parse(text = paste0("train.full.", gsub(" ", "", spec, fixed = TRUE))))
  train.var.spec <- train.var.spec[train.var.spec$insample<2,]
  
  # count numobs per point id variable
  var.dupes <- as.data.frame(table(train.var.spec$id))
  var.dupes <- var.dupes %>% transmute(id = as.integer(as.character(Var1)), varduplis = Freq)
  table(var.dupes$varduplis)
  train.var.spec <- left_join(train.var.spec, var.dupes)
  
  # generate duplicate counter by id variable
  train.var.spec$number <- 1
  train.var.spec <- train.var.spec %>% group_by(id) %>% mutate(varduplicounter = cumsum(number))
  table(train.var.spec$varduplicounter)
  train.var.spec$number <- NULL
  
  # all remaining duplicates are in duplicates of true TGA
  crosstab(train.var.spec, row.vars = c("insample"), col.vars = c("varduplicounter"), 
                             type = "f", addmargins = TRUE)
  
  # remove duplicates
  train.var.spec <- train.var.spec[train.var.spec$varduplicounter == 1,]
  table(train.var.spec$presence)
  table(train.var.spec$insample)
  
  train.var.name <- paste0("train.full.var.", gsub(" ", "", spec, fixed = TRUE))
  assign(train.var.name, train.var.spec)
  
  cat(" Species:", spec, "\n",
      "variant: no duplicates whatsoever \n",
      "variant train obs:", nrow(train.var.spec), "\n",
      "variant presence-absence: ", table(train.var.spec$presence), "\n",
      "mean prevalence in variant data set is ", round(mean(train.var.spec$presence),2), "\n.\n"
       )
  
 }



################################################################################################
######################### Generate blockCV folds / split for tuning data #######################
################################################################################################

# create own train/test split (0.7/0.3)
# other option is to no split and let deepregression::cv do the CV fold splitting
#set.seed(seed)
#train.indices <- createDataPartition(y= df.tune$presence, p = 0.8, list = FALSE)
#train.tune  <- df.tune[train.indices,]
#val.tune <- df.tune[-train.indices,]

# check composition of train/val tune sets
#table(train.tune$presence)
#table(train.tune$insample)
#table(val.tune$presence)
#table(val.tune$insample)

# repeat split for df.tune.var 
#train.ind.var <- createDataPartition(y= df.tune.var$presence, p = 0.8, list = FALSE)
#train.tune.var  <- df.tune.var[train.ind.var,]
#val.tune.var <- df.tune.var[-train.ind.var,]

# check composition
#table(train.tune.var$presence)
#table(train.tune.var$insample)
#table(val.tune.var$presence)
#table(val.tune.var$insample)


### create spatially decorrelated CV folds for df.tune and df.tune.var using blockCV::spatialBlock

# choose block width either as min, max or median value or species-specific block widths => max?
block.widths <- rep(NA,length(insample.species))
i = 1
for(spec in insample.species){
  spec.list <- eval(parse(text = paste0(gsub(" ", "", spec, fixed = TRUE))))
  block.width.spec <- spec.list$block_width
  block.widths[i] <- block.width.spec
  rm(block.width.spec)
  i = i + 1
}
summary(block.widths) # median around 300,000 meters


## df.tune : create SpatialPointsDataFrame for spatialBlock command #######################

# choose block width for df.tune points
block.width.tune <- 1.1*max(block.widths)
block.width.tune

# create new species variable to evenly distribute presence points
df.tune$spec.cv <- as.character(df.tune$spec)
df.tune$spec.cv[df.tune$presence == 0] <- "other"
tab.tune.spec.cv <- crosstab(df.tune, row.vars = c("spec.cv"), 
                             col.vars = c("insample"), 
                             type = "f", 
                             addmargins = TRUE
                             )
tab.tune.spec.cv

# define CRS
crs <- CRS("+init=epsg:4326")
# df.tune as SPDF
coords <- cbind(df.tune$longitude, df.tune$latitude)
df.tune.redu <- df.tune  %>% select(-one_of(c("longitude", "latitude")))
sp <- SpatialPoints(coords, proj4string = crs)
df.tune.spdf <- SpatialPointsDataFrame(coords, df.tune.redu)
#df.tune.spdf <- SpatialPointsDataFrame(coords, df.tune.redu, proj4string = crs)
#df.tune.spdf <- df.tune.spdf %>% st_transform(4326) # WGS84 - good default

# seed procedure
set.seed(seed) 

# spatialBlock command
block.cv.res <- spatialBlock(speciesData = df.tune.spdf,
                   species = "spec.cv",
                   #rasterLayer = awt,
                   theRange = block.width.tune, # size of the blocks
                   k = 4, # number of folds to create
                   selection = "random",
                   iteration = 200, # find evenly dispersed folds, 100 better
                   #numLimit = 10,
                   biomod2Format = TRUE,
                   xOffset = 0, # shift the blocks horizontally
                   yOffset = 0
                   )

# add new spatial CV folds variable to df.tune
df.tune$newfold <- as.character(block.cv.res$foldID)
table(df.tune$newfold)

# crosstab original species-specific data set vs new fold membership
crosstab(df.tune, row.vars = c("spec"), col.vars = c("newfold"), 
                             type = "f", addmargins = TRUE)

# crosstab presence points by species vs newfold
crosstab(df.tune, row.vars = c("spec.cv"), col.vars = c("newfold"), 
         type = "f", addmargins = TRUE)


## df.tune.var : create SpatialPointsDataFrame for spatialBlock ####################

# choose block width for df.tune.var points
block.width.tune.var <- 1*max(block.widths)
block.width.tune.var

# create new species variable to evenly distribute presence points
df.tune.var$spec.cv <- as.character(df.tune.var$spec)
df.tune.var$spec.cv[df.tune.var$presence == 0] <- "other"
tab.tune.var.spec.cv <- crosstab(df.tune.var, row.vars = c("spec.cv"), 
                             col.vars = c("insample"), 
                             type = "f", 
                             addmargins = TRUE
                             )
tab.tune.var.spec.cv

# define CRS
crs <- CRS("+init=epsg:4326")
# df.tune.var as SPDF
coords.var <- cbind(df.tune.var$longitude, df.tune.var$latitude)
df.tune.var.redu <- df.tune.var  %>% select(-one_of(c("longitude", "latitude")))
sp.var <- SpatialPoints(coords.var, proj4string = crs)
df.tune.var.spdf <- SpatialPointsDataFrame(coords.var, df.tune.var.redu)
#df.tune.var.spdf <- SpatialPointsDataFrame(coords, df.tune.var.redu, proj4string = crs)
#df.tune.var.spdf <- df.tune.var.spdf %>% st_transform(4326) # WGS84 - good default

# seed procedure
set.seed(seed)

# spatialBlock command
block.cv.res.var <- spatialBlock(speciesData = df.tune.var.spdf,
                             species = "spec.cv",
                             #rasterLayer = awt,
                             theRange = block.width.tune.var, # size of the blocks
                             k = 4, # number of folds to create
                             selection = "random", #systematic?
                             iteration = 200, # find evenly dispersed folds, 100 better
                             #numLimit = 10,
                             biomod2Format = TRUE,
                             xOffset = 0, # shift the blocks horizontally
                             yOffset = 0
                             )

# add new spatial CV folds to df.tune.var
df.tune.var$newfold <- as.character(block.cv.res.var$foldID)
table(df.tune.var$newfold)

# crosstab original species-specific data set vs new fold membership
crosstab(df.tune.var, row.vars = c("spec"), col.vars = c("newfold"), 
         type = "f", addmargins = TRUE)

# crosstab presence points by species vs newfold
crosstab(df.tune.var, row.vars = c("spec.cv"), col.vars = c("newfold"), 
         type = "f", addmargins = TRUE)


################################################################################################
################################## explore data sets further  ##################################
################################################################################################

# create common extend as union of extended hulls
full.ext <- folds[[insample.species[1]]]$extent
blocks.Triatomainfestans <- folds[[insample.species[1]]]$blocks$blocks
for (spec in insample.species[2:7]){
  # load data
  data <- folds[[spec]]
  ext.spec <- data$extent
  ext.spec
  full.ext <- raster::union(full.ext, ext.spec)
  print(full.ext)
  blocks.spec <- data$blocks$blocks
  assign(paste0("blocks.", gsub(" ", "", spec, fixed = TRUE)), blocks.spec)
  
  rm(data, ext.spec)
}
full.ext

##################### long computation time ###################################################
# crop country borders to extent                                                              #
#countries.full <- countries %>% raster::crop(full.ext) ########################################
#rm(countries)                                                                                 #
##################### long computation time ###################################################

## convert data frames to spatialpointsdataframe
# define CRS
crs <- CRS("+init=epsg:4326")

### visualize df.tune points vs test.full points

## first, plot df.tune vs test.full

# df.tune
coords <- cbind(df.tune$longitude, df.tune$latitude)
df.tune.redu <- df.tune  %>% select(-one_of(c("longitude", "latitude")))
sp <- SpatialPoints(coords, proj4string = crs)
df.tune.spdf <- SpatialPointsDataFrame(coords, df.tune.redu, proj4string = crs)

# test.full
coords.test <- cbind(test.full$longitude, test.full$latitude)
test.full.redu <- test.full %>% select(-one_of(c("longitude", "latitude")))
sp.test <- SpatialPoints(coords.test, proj4string = crs)
test.full.spdf <- SpatialPointsDataFrame(coords.test, test.full.redu, proj4string = crs)

# cv blocks for respective species
cv.blocks <- block.cv.res$blocks

## next, plot df.tune.var vs test.full (no duplicates in train set and no ids from test set)
# plot should look the same since only duplicates were removed

# df.tune.var
coords.var <- cbind(df.tune.var$longitude, df.tune.var$latitude)
df.tune.var.redu <- df.tune.var  %>% select(-one_of(c("longitude", "latitude")))
sp.var <- SpatialPoints(coords.var, proj4string = crs)
df.tune.var.spdf <- SpatialPointsDataFrame(coords.var, df.tune.var.redu, proj4string = crs)

# test.full
coords.test <- cbind(test.full$longitude, test.full$latitude)
test.full.redu <- test.full %>% select(-one_of(c("longitude", "latitude")))
sp.test <- SpatialPoints(coords.test, proj4string = crs)
test.full.spdf <- SpatialPointsDataFrame(coords.test, test.full.redu, proj4string = crs)

# cv blocks for respective species
cv.blocks <- block.cv.res.var$blocks

################################################################################################
################################ consolidate objects and save  #################################
################################################################################################

# we now have combined the multi-species presence-only data into severa data constructs
# consolidate objects in lists and save 
# data sets are  hereafter ready for pre-processing of variables for model fitting

# df.full, df.tune, df.tune.var (includes new blockCV folds in column newfolds)
# train.full, test.full
basic.dfs <- list(df.full = df.full, df.tune = df.tune, df.tune.var = df.tune.var, train.full = train.full,
                  test.full = test.full)

# train.full.spec for all 7 insample species
train.dfs.spec  <-  list(
                         train.full.Triatomainfestans = train.full.Triatomainfestans,
                         train.full.Triatomadimidiata = train.full.Triatomadimidiata,
                         train.full.Panstrongylusmegistus = train.full.Panstrongylusmegistus,
                         train.full.Triatomabrasiliensis = train.full.Triatomabrasiliensis,
                         train.full.Triatomasordida = train.full.Triatomasordida,
                         train.full.Triatomapseudomaculata = train.full.Triatomapseudomaculata,
                         train.full.Triatomabarberi = train.full.Triatomabarberi
                         )

# train.full.var.spec for all 7 insample species
train.dfs.var.spec <- list(
                           train.full.var.Triatomainfestans = train.full.var.Triatomainfestans,
                           train.full.var.Triatomadimidiata = train.full.var.Triatomadimidiata,
                           train.full.var.Panstrongylusmegistus = train.full.var.Panstrongylusmegistus,
                           train.full.var.Triatomabrasiliensis = train.full.var.Triatomabrasiliensis,
                           train.full.var.Triatomasordida = train.full.var.Triatomasordida,
                           train.full.var.Triatomapseudomaculata = train.full.var.Triatomapseudomaculata,
                           train.full.var.Triatomabarberi = train.full.var.Triatomabarberi
                           )

# presence.ids, tr.presence.ids, te.presence.ids, tune.presence.ids
# unique.test.ids, unique.test.ids.spec 1-7, unique.train.ids.spec 1-7, unique.tune.ids
id.list <- list(
                presence.ids = presence.ids,
                tr.presence.ids = tr.presence.ids,
                te.presence.ids = te.presence.ids,
                tune.presence.ids = tune.presence.ids, # also equal to tune.var.presence.ids
                unique.test.ids = unique.test.ids,
                unique.tune.ids = unique.tune.ids,
                unique.tune.var.ids = unique.tune.var.ids,
                unique.test.ids.Triatomainfestans = unique.test.ids.Triatomainfestans,
                unique.test.ids.Triatomadimidiata = unique.test.ids.Triatomadimidiata,
                unique.test.ids.Panstrongylusmegistus = unique.test.ids.Panstrongylusmegistus,
                unique.test.ids.Triatomabrasiliensis = unique.test.ids.Triatomabrasiliensis,
                unique.test.ids.Triatomasordida = unique.test.ids.Triatomasordida,
                unique.test.ids.Triatomapseudomaculata = unique.test.ids.Triatomapseudomaculata,
                unique.test.ids.Triatomabarberi = unique.test.ids.Triatomabarberi,
                unique.train.ids.Triatomainfestans = unique.train.ids.Triatomainfestans,
                unique.train.ids.Triatomadimidiata = unique.train.ids.Triatomadimidiata,
                unique.train.ids.Panstrongylusmegistus = unique.train.ids.Panstrongylusmegistus,
                unique.train.ids.Triatomabrasiliensis = unique.train.ids.Triatomabrasiliensis,
                unique.train.ids.Triatomasordida = unique.train.ids.Triatomasordida,
                unique.train.ids.Triatomapseudomaculata = unique.train.ids.Triatomapseudomaculata,
                unique.train.ids.Triatomabarberi = unique.train.ids.Triatomabarberi
                )



# block.cv.res, block.cv.res.var, df.tune.spdf, df.tune.var.spdf, test.full.spdf
# blocks.spec 1-7, full.ext
block.cv.list <- list(
                    block.cv.res = block.cv.res,
                    block.cv.res.var = block.cv.res.var,
                    df.tune.spdf = df.tune.spdf,
                    df.tune.var.spdf = df.tune.var.spdf,
                    test.full.spdf = test.full.spdf,
                    full.ext = full.ext,
                    old.blocks.species = list(
                                            blocks.Triatomainfestans = blocks.Triatomainfestans,
                                            blocks.Triatomadimidiata = blocks.Triatomadimidiata,
                                            blocks.Panstrongylusmegistus = blocks.Panstrongylusmegistus,
                                            blocks.Triatomabrasiliensis = blocks.Triatomabrasiliensis,
                                            blocks.Triatomasordida = blocks.Triatomasordida,
                                            blocks.Triatomapseudomaculata = blocks.Triatomapseudomaculata,
                                            blocks.Triatomabarberi = blocks.Triatomabarberi
                                            )
                                        )                         

# tables: obs.origin, tab.insample, tab.presence, tab.test.presence, tab.totduplis,
#         tab.train.duplis, tab.train.insample, tab.tune.presence, tab.tune.spec.cv,
#         tab.tune.totduplis, tab.tune.var.spec.cv

tables.list <- list(
                obs.origin = obs.origin, 
                tab.insample = tab.insample, 
                tab.presence = tab.presence, 
                tab.test.presence = tab.test.presence, 
                tab.totduplis = tab.totduplis,
                tab.train.duplis = tab.train.duplis, 
                tab.train.insample = tab.train.insample, 
                tab.tune.presence = tab.tune.presence, 
                tab.tune.spec.cv = tab.tune.spec.cv,
                tab.tune.totduplis = tab.tune.totduplis,
                tab.tune.var.spec.cv = tab.tune.var.spec.cv
                )


# final list of lists containing full model preparation results
full.model.list <- list(
                        basic.dfs = basic.dfs,
                        train.dfs.spec = train.dfs.spec,
                        train.dfs.var.spec = train.dfs.var.spec,
                        id.list = id.list,
                        block.cv.list = block.cv.list,
                        tables.list = tables.list
                        )

# save list
saveRDS(full.model.list, file = file.path(out.path, paste0("full-model-list.rds")) )

  



################################################################################################
################################################################################################
################################################################################################
################################################################################################


