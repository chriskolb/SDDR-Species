################################################################################################
################# formulae for all predictor combinations in deepregressions ######### #########
################################################################################################


# linear predictor
form.lin <- formula(paste(
  "~1 +",
  paste(varlist, collapse = " + ")
))


# linear predictor
form.lin.smooth <- formula(paste(
  "~1 +",
  paste(varlist, collapse = " + "),
  " + s(",
  paste(varlist[3:length(varlist)], collapse = ") + s("),
  ")",
  " + s(longitude, latitude, bs='gp')"
))


# linear predictor + dnn predictor
form.lin.deep <- formula(paste(
  "~1 +",
  paste(varlist, collapse = " + "),
  " + nn_smooth(", 
  paste(varlist, collapse = ", "),
  ")"
))


# structured non-linear predictor, might want to add bs = 'gp' in bivariate smooth
form.smooth <- formula(paste(
  "~1 + s(",
  paste(varlist[3:length(varlist)], collapse = ") + s("),
  ")",
  " + s(longitude, latitude, bs='gp')"
))


# structured + unstructured non-linear predictors (might use bs = 'gp' in bivariate smooth)
# same predictor as in Bender et al. but extended through a neural network predictor
form.smooth.deep <- formula(paste(
  "~1 + s(",
  paste(varlist[3:length(varlist)], collapse = ") + s("),
  ")",
  " + s(longitude, latitude, bs = 'gp')",
  " + nn_deep(", 
  paste(varlist, collapse = ", "),
  ")"
))

# unstructured (deep) non-linear predictor
form.deep <-   formula(paste(
  "~1 +",
  "nn_deep(", 
  paste(varlist, collapse = ", "),
  ")"
))

# predictor comprising linear, structured non-linear and unstructured non-linear effects
# does not work currently
form.lin.smooth.deep <- formula(paste(
  "~1 +",
  paste(varlist, collapse = " + "),
  "+ s(",
  paste(varlist[3:length(varlist)], collapse = ") + s("),
  ")",
  " + s(longitude, latitude, bs = 'gp')",
  " + nn_smooth(", 
  paste(varlist, collapse = ", "),
  ")"
))


################################################################################################
# keras custom auc metric
# loosely adapted/corrected from here: https://www.kaggle.com/springmanndaniel/keras-r-embeddings-baseline
# to use in deepregression, specify argument monitor_metric = auc_metric

fun.auc <- function(pred.probs, labels) {
  pr   <- pred.probs
  labs <- labels
  auc.res <- Metrics::auc(actual = labs, predicted =  pr)
  return(auc.res)
}

np <- import("numpy", convert = T)
auc.tf.out <- function(y_true, y_pred) {
  
  auc.numpy.out <- function(y_true, y_pred){
    out <- fun.auc(y_pred, y_true)
    return(np$double(out))
  }
  
  return(tensorflow::tf$numpy_function(func = auc.numpy.out, inp = c(y_true,y_pred), Tout = tensorflow::tf$double))
}

auc_metric <- keras::custom_metric(name = "auc", metric_fn = auc.tf.out)
################################################################################################


cat("######################################################################################\n")
cat("########################## Defining formulae was successful!##########################\n")
cat("######################################################################################\n")
cat("Formulae corresponding to predictor types: \n",
    "lin\n",
    "lin+smooth\n",
    "lin+dnn\n",
    "smooth\n",
    "smooth+dnn\n",
    "dnn\n",
    "lin+smooth+dnn \n",
    "-----------------------------------------------------------\n",
    "Formulae are of the following form:\n form.example = \n",
    "~1 + a + b + c + s(a) + s(b) + s(c) + nn_model(a,b,c)\n",
    "------------------------------------------------------------\n",
    "Custom metrics defined: auc_metric"
    )

cat("\n######################################################################################\n")


