David Ruegamer
9/15/2020

# Table of Contents

0.  [Introduction](#introduction)
1.  [Interface and Model
    Specification](#interface-and-model-specification)
2.  [Details on Model Fitting](#details-on-model-fitting)
    1.  [Specification of the Family](#specification-of-the-family)
    2.  [Specification of the Formulas](#specification-of-the-formulas)
    3.  [Specification of the DNNs](#specification-of-the-dnns)
3.  [Model Fitting and Tuning](#model-fitting-and-tuning)
    1.  [Model Fitting](#model-fitting)
    2.  [Model Tuning](#model-tuning)
4.  [Other Methods](#other-methods)
    1.  [coef](#coef)
    2.  [plot](#plot)
    3.  [predict](#predict)
    4.  [mean, sd, quantile](#mean-and-sd)
    5.  [get distribution](#get-distribution)
    6.  [log score](#log-score)
5.  [Penalties](#penalties)
    1.  [Smoothing Penalties](#smoothing-penalties)
6.  [Neural Network Settings](#neural-network-settings)
    1.  [Shared DNN](#shared-dnn)
    2.  [Optimizer and Learning Rate](#optimizer-and-learning-rate)
    3.  [Orthogonalization](#orthogonalization)
7.  [Advanced Usage](#advanced-usage)
    1.  [Bayesian Regression](#bayesian-regression)
    2.  [Offsets](#offsets)
    3.  [Constraints](#constraints)
    4.  [Custom Distribution Function](#custom-distribution-function)
    5.  [Other Model Classes](#other-model-classes)
        1.  [Mixture Models](#mixture-models)
        2.  [Transformation Models](#transformation-models)
    6.  [Custom Orthogonalization](#custom-orthogonalization)
8.  [Toy Examples](#toy-examples)
    1.  [Deep Additive Regression](#deep-additive-regression)
    2.  [Deep Logistic Regression](#deep-logistic-regression)
    3.  [Deep GAM](#deep-gam)
    4.  [GAMLSS](#gamlss)
    5.  [Deep GAMLSS](#deep-gamlss)
    6.  [Zero-inflated
        Distributions](#zero-inflated-poisson-distribution)
9.  [Real World Applications](#real-world-applications)
    1.  [Deep Mixed Model for Wage Panel
        Data](#deep-mixed-model-for-wage-panel-data)
    2.  [High-Dimensional Ridge and Lasso Regression on Colon Cancer
        Data](#high-dimensional-ridge-and-lasso-regression)
    3.  [Mixture of Normal Distributions for Acidity
        Modeling](#mixture-of-normal-distributions-for-acidity-modeling)
    4.  [Unstructured Data Examples](#unstructured-data-examples)
        1.  [MNIST Pictures with Multinomial
            Response](#mnist-multinomial)
        2.  [Sentiment Analysis using IMDB Reviews](#text-as-input)

## Introduction

`deepregression` is an implementation of a large number of statistical
regression models, but fitted in a neural network. It can be used for
mean regression as well as for distributional regression,
i.e. estimating any parameter of the assumed distribution, not just the
mean. Each parameter can be defined by a linear predictor.
`deepregression` uses the pre-processing of the package `mgcv` to build
smooth terms and has a similar formula interface as `mgcv::gam`. As all
models are estimated in a neural network, `deepregression` can not only
make use of TensorFlow as a computing engine, but allows to specify
parameters also by additional deep neural networks (DNNs). This allows
to include, e.g., CNNs or LSTMs into the model formula and thus
incorporate unstructured data sources into a regression model. When
combining structured regression models with DNNs, the software uses an
orthogonalization cell to make the structured parts of the model formula
(the linear and smooth terms) identifiable in the presence of the
DNN(s).

## Interface and Model Specification

As in many deep learning (DL) implementations, fitting a model (a
network) is a two-step procedure. At first, the model is initialized by
defining all model and distribution assumptions:

``` r
mod <- deepregression(
  
  # supply the response; usually a vector
  y = y, 
  
  # supply the data; a data.frame or list
  data = data, 
  
  # specify the distribution to be learned
  family = "gumbel"
  
  # specify a list of formulas, each corresponding to one parameter
  # of the defined distribution (details later)
  list_of_formulae = list( # 
    param1 = ~ 1 + s(x, bs = "tp") + my_deep_mod(x)
    param2 = ~ 0 + x + my_deep_mod2(z)
      ),
  
  # define a list of DNNs, with list names corresponding
  # to the names given in the formula (details later)
  list_of_deep_models = list(my_deep_mod = some_function,
                             my_deep_mod2 = some_function2)

)
```

Once the model `mod` has been set up, we can train the neural network
using the `fit` function:

``` r
mod %>% fit(
  
  # number of iterations to train the network
  epochs = 1000, 
  
  # should progress be printed in the console
  verbose = FALSE, 
  
  # in RStudio you can get a visualization
  # of the training process using TRUE for
  # this option
  view_metrics = FALSE
  
  )
```

The following section will give more details on how to define different
parameters and how to train a `deepregression` model in practice.

## Details on Model Fitting

### Specification of the Family

The `family` argument is used to specify the family that is learned.
Possible choices are:

``` r
possible_families = c(
  
  "normal", "bernoulli", "bernoulli_prob", "beta", "betar",
  "cauchy", "chi2", "chi","exponential", "gamma_gamma",
  "gamma", "gammar", "gumbel", "half_cauchy", "half_normal", 
  "horseshoe", "inverse_gamma", "inverse_gaussian", "laplace", 
  "log_normal", "logistic", "multinomial", "multinoulli", 
  "negbinom", "pareto", "poisson", "poisson_lograte", "student_t",
  "student_t_ls", "truncated_normal", "uniform", "zip",
  "transformation_model"
  
)
```

The last option can be used to specify deep conditional transformation
models as proposed in XXX (2020). It is also possible to define mixtures
of families in `deepregression` as proposed by Ruegamer, Pfisterer and
Bischl (2020) or custom distributions. Details can be found in the
[Advanced Usage](#advanced-usage) Section. The parameters of each
distribution can be found in the help file `?make_tfd_dist`.

### Specification of the Formulas

The formulas defining each parameter of the distribution can be
specified in the same way as for `mgcv::gam` including `s`-terms, `ti`-
and `te`-terms and all further specifications of those smooth effects
like the smoothing basis (see `?mgcv::s`). Factor variables in the
formula are treated also the same way as in conventional regression
packages by creating an encoded matrix (usually dummy-encoding / one-hot
encoding). The exclusion of the intercept in one linear predictor can be
defined as per usual using the `0` in the formula: `~ 0 + ...`.

Example:

``` r
mod <- deepregression(
  y = y, data = data,
  family = "bernoulli",
  list_of_formulae = list(
    logit = ~ 1 + fac_var + s(x, bs = "ps") + s(z, bs = "re") + deep(w1,w2,w3)
  ),
  list_of_deep_models = list(deep = function_for_ws)
)
)
```

This model definition translated to a logistic regression model where
the logits of the model are defined by an intercept (`1`), a linear
effect of a factor variable `fac_var`, a p-spline for variable `x`
(`s(x, bs = "ps")`), a random effect for variable `z` and a DNN for
variables `w1,w2,w3` (`deep(w1,w2,w3)`). The next subsection will
explain how to define the DNNs.

### Specification of the DNNs

The DNNs specified in the `list_of_formulae` must also be passed as
named list elements in the `list_of_deep_models` (see previous example),
i.e., each model term in the `list_of_formulae` that is not part of the
usual `gam`-terms must be listed there. The named list
`list_of_deep_models` contains the DNNs as functions of their input.
These functions take as many inputs as defined for the neural network
(the three `w`s in the above example) and ends with a fully-connected
layer with one hidden unit (or any other layer that results in the
appropriate amount of outputs). The following specifies an exemplary
function for the above example using the pipe (`%>%`) notation:

``` r
function_for_ws <- function(x)
{
  
  # this specifies a two hidden-layer DNN with dropouts between each layer
  
  # not that you do not need to specify several arguments
  # this is done automatically
  x %>% 
    layer_dense(units = 24, activation = "relu", use_bias = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 12, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = "linear")
  
}
```

To ensure identifiability when structured regression terms and a DNN
share some input covariates, an orthogonalization cell is automatically
included before the last dense layer. In this case it is required to use
a `'linear'` activation function as in this example.

## Model Fitting and Tuning

### Model Fitting

Model fitting can be done using the `fit.deepregression` function as
explained in [Interface and Model
Specification](#interface-and-model-specification). The `fit` function
is a wrapper for the corresponding `fit` function of `keras` models and
inherits the `keras::fit` arguments. More specifically - `epochs` to
specify the number of iterations - `callbacks` to specify information
that is called after each epoch (used, e.g., for early stopping) -
`validation_split` to specify the amount of data (in \[0,1)) that is
used to validate the model (while the rest is used to train the model) -
`validation_data` to specify any predefined validation data and several
convenience arguments such as a logical argument `early_stopping` to
active early stopping and `patience` to define the patience used in
early stopping.

### Model Tuning

In a similar manner as `fit`, `deepregression` offers a cross-validation
function `cv` for model tuning. This can be used to fine-tune the model
by, e.g., changing the formula(s), changing the DNN structure or
defining the amount of smoothness (using the `df` argument in
`deepregression`; see the next section for details). Folds in the `cv`
function can be specified using the `cv_folds` argument. This takes
either an integer for the number of folds or a list where each list item
is again a list of two, one element with data indices for training and
one with indices for testing. The `patience` for early stopping can be
set using the respective argument.

## Other Methods

There are different other methods that can be applied to a
`deepregression` object after training.

### coef

The `coef` function extracts the coefficients (network weights) of all
layers with structured definition, i.e., coefficients of linear of
additive effects.

### plot

The `plot` function can be applied to `deepregression` objects to plot
the estimates of non-linear effects, i.e., splines and tensor product
splines. Using `which` a specific effect can be selected using the
corresponding integer in the structured part of the formula (if `NULL`
all effects are plotted), while the integer given for `which_param`
indicates which distribution parameter is chosen for plotting. Via the
argument `plot = FALSE`, the function can also be used to just return
the data used for plotting.

### predict

The `predict` function works as for other (keras) models. It either just
takes the model as input and returns the predictions on the training
data set, or, when supplied with new data, produces the corresponding
prediction for the new data. As `deepregression` learns a distribution
and not only a mean prediction, the user can choose via the argument
`apply_fun` what type of distribution characteristic is to be used for
prediction (per default `apply_fun = tfd_mean` predicts the mean of the
distribution).

### mean, sd and quantile

`mean`, `sd` and `quantile` are convenience functions that directly
return the mean, standard deviation or quantile of the learned
distribution. All three functions work for the given training data as
well as for new provided `data` and the `quantile` function additionally
has an argument `value` to provide the quantile of interest.

### get\_distribution

Instead of returning summary statistics, the function `get_distribution`
returns the whole TensorFlow distribution with all its functionalities,
such as sampling, computing the PDF or CDF, etc.

### log\_score

`log_score` is another convenience function that directly returns the
evaluated log-likelihood based on the estimated parameters and the
provided data (including a response vector `this_y`). If `data` and
`this_y` are not provided, the function will calculate the score on the
training data.

## Penalties

`deepregression` allows for different penalties including \(L_1\)-,
\(L_2\)- and smoothing penalties. While the latter is implicitly created
when using `s`-, `ti`- or `te`-terms in the formula, the \(L_1\)- and
\(L_2\)-penalty are used to penalize linear predictor models without
smooth terms by defining the amount of penalization via `lambda_lasso`
or `lambda_ridge`, respectively. If both terms are used, this
corresponds to a elasticnet-type penalty. Since the model object
returned by `deepregression` is a list where the first element is a
`keras` model, additional penalties can always be specified after the
model has be initialized using the `keras::add_loss` API. `add_loss`
requires a lambda function that penalizes the trainable network weights.
This function can be passed to the model building in deepregression
using the argument `additional_penalty`.

### Smoothing Penalties

Apart from specifying the smoothing penalties in each smooth term in the
formulas (see `mgcv::s` for more details), two further options are
available in `deepregression`. The first option is to use a single
degrees-of-freedom specification using the argument `df`
in`deepregression`. Using the Demmler-Reinsch Orhtogonalization, all
smoothing parameters are then calculated based on this specification
(e.g., setting `df = 5` results in `sp = 1.234` for one smooth, but `sp
= 133.7` for another smooth due to their different nature and data).
This ensures that no smooth term has more flexibility than the other
term which makes sense in certain situations. If `df` is left
unspecified, the default is used which is the smallest basis dimension
among all smooths (leaving the least flexible smooth unpenalized while
the others are penalized to have the same degrees-of-freedom as this
one). The second option in `deepregression` is to specify a list of
length `length(list_of_formulae)`. This not only allows to specify
different `df` for each distribution parameter, but also to specify
different `df` for each smooth term in each formula by providing a
vector of the same length as the number of smooths in the parameter’s
formula.

For expert usage: The definition of the degrees-of-freedom can be
changed using the `hat1` argument. When set to `TRUE` (default is
`FALSE` yielding the usual definition of the effective
degrees-of-freedom), the `df` are assumed to be the sum of the hat
matrix of the corresponding smooth term. In certain situations it is
also necessary to scale the penalty. This can be done using the argument
`sp_scale`, which `1` per default.

## Neural Network Settings

Since `deepregression` is a holistic neural network, certain settings
and advantages from DL can be made of.

### Shared DNN

In addition to what was introduced in the Section [Specification of the
DNNs](#specification-of-the-dnns) `deepregression` allows to share one
DNN between some or all distribution parameters. This can make sense for
several reasons, in particular, a reduction of the number of parameters
that have to be estimated. The following example will demonstrate how to
share a DNN (an LSTM model) between two parameters:

``` r
# Deep Neural Network
lstm_mod <- function(x) x %>%
    layer_embedding(input_dim = tokenizer$num_words,
                    output_dim = embedding_size) %>%
    layer_flatten() %>% 
    layer_dense(2) # note the number of output units
# must be equal to the number of parameters learned

# define formulas for mean and scale parameter
form_lists <- list(
    mean = ~ 1 + s(xa) + s(xb),
    scale = ~ 1 + s(xc)
  )
  

# shared network list (of the same length as form_lists)
tt_list <- list( ~ lstm_mod(texts))[rep(1,2)]
  
mod <- deepregression(y = data$y,
                      list_of_formulae = form_lists,
                      list_of_deep_models = 
                        list(lstm_mod = lstm_mod), 
                      family = "normal", 
                      train_together = tt_list,
                      data = list(xa = xa, # these could be vectors 
                                  xb = xb,
                                  xc = xc, 
                                  # and texts is typically matrix
                                  # based on tokenization of the
                                  # text sequence
                                  texts = texts
                                  )
  )
```

This example would create an additive model for a normal distribution,
where both the mean and the scale parameter are trained by the same
network and in the last layer, the hidden features are then split and
added to the respective linear predictor.

### Optimizer and Learning Rate

`deepregression` directly passes the `optimizer` to `keras`. It is
therefore possible to specify any optimizer in `deepregression` that is
also available in `keras`:

  - `optimizer_adadelta`
  - `optimizer_adagrad`
  - `optimizer_adam`
  - `optimizer_adamax`
  - `optimizer_adadelta`
  - `optimizer_nadam`
  - `optimizer_rmsprop`
  - `optimizer_sgd`

Many of those work well. Adam is the default and its learning rate can
be changed using the `learning_rate` argument. If specified differently,
the optimizer uses its own defaults, which can also be defined. For
example:

``` r
deepregression(..., optimizer = optimizer_adadelta(lr = 3, decay = 0.1))
```

This overwrites the `learning_rate` argument of `deepregression`.

### Orthogonalization

In `deepregression` per default orthogonalizes the DNNs in predictor
formulas if they are given the same term as present in a structured
partial effect. For example, for

``` r
deepregression(
  ...,
  list_of_formulae = list(logit = ~ 1 + x + s(z) + s(w) + q + deep_mod(x,y,z)),
  list_of_deep_models = list(deep_mod = deep_model)
)
```

the `deep_mod` is orthogonalized w.r.t. `x` and the design matrix of
`s(z)`, but not w.r.t. `q` or the design matrix of `s(w)`, nor `y`.
Orthogonalization is done to ensure identifiability of the the
structured terms, but can also be deactivated using `orthogonalize =
FALSE`. Per default, orthogonalization extracts the terms automatically
which overlap in the DNNs and the structured model formula. For expert
use there is also a custom orthogonalization (see [Custom
Orthogonalization](#custom-orthogonalization))

## Advanced Usage

`deepregression` allows for several advanced user inputs.

### Bayesian Regression

The network can be turned into a Bayesian neural network (BNN) by using
so-called variational layers, which place a prior distribution over the
weight of each layer and the network is trained to minimize the ELBO
criterion. This allows to obtain an approximate posterior distribution
which also characterizes the uncertainty in the network’s weights. This
can be done by setting `variational = TRUE`. In addition, experts can
define their own prior distributions and posterior distribution using
the arguments `prior_fun` and `posterior_fun`, respectively. Per
default, a normal prior distribution is used with the inverse of
smoothing penalty matrices as covariance and mean field posterior
approximation.

### Offsets

Several statistical models require an offset to be used in one or more
linear predictors. These can be specified using the argument `offset` as
a list of column vectors (i.e. matrices with 1 column) or `NULL` to
include no offset. Per default, no offset is used in any parameter.

### Constraints

For smooth effects, two options are currently available to constraint
their estimation. `absorb_cons` (default `FALSE`) will absorb
identifiability constraints into the smoothing basis (see
`?mgcv::smoothCon`) and `zero_constraint_for_smooths` will constraint
all smooth to sum to zero over their domain, which is usually
recommended to prevent identifiability issues (default `TRUE`).

### Custom Distribution Function

It is also possible to define custom distribution function to be learned
in `deepregression` using the `dist_fun` argument. To specify a custom
distribution, define the a function as follows:

``` r
function(x) do.call(your_tfd_dist, # a distribution from tfprobability
                    lapply(1:ncol(x)[[1]], # which iterates over the number
                           # of parameters
                           function(i)
                             # and applies a transformation, e.g., tf$exp(),
                             # to ensure that the parameter lives in the correct
                             # domain
                             your_trafo_list_on_inputs[[i]](
                               x[,i,drop=FALSE]
                               )
                           ))
```

## Other Model Classes

Apart from custom distribution, which also allow bijective
transformations of functions, `deepregression` includes special model
classes with additional support.

### Mixture Models

`deepregression` can be used to fit *Neural Mixture Distributional
Regression* models, a large class of mixture models fitted in a neural
network. Example codes can be found [here](...)

### Transformation Models

The `deepregression` package also allows to fit *Deep Conditional
Transformation Models*, a large class of transformation models fitted in
a neural network. Example codes can be found [here](...)

### Custom Orthogonalization

If there is reason to orthogonalize a DNN w.r.t. a model term that is
not explicitly present, the `%OZ%`-operator can be used. For example,

``` r
deepregression(
  ...,
  list_of_formulae = list(logit = ~ 1 + deep_mod(x,y,z) %OZ% (x + s(z)) + s(w) + q),
  list_of_deep_models = list(deep_mod = deep_model)
)
```

would orthogonalize the network in the same way as in the previous
[Orthogonalization](#orthogonalization) example, but without having `x`
or `s(z)` as an actual predictor in the formula. Left of the operator, a
DNN has to be given. On the right of the operator either a single model
term (such as `x`, `s(x)`, `te(x,z)`) or a combination of model terms
using brackets (as in the example above) separated with `+` must be
supplied.

## Toy Examples

### Deep Additive Regression

We first create a very simple regression where we try to model the
non-linear part of the data generating process using a complex neural
network and an intercept using a structured linear part.

``` r
set.seed(24)

# generate the data
n <- 1500
b0 <- 1

# training data; predictor 
x <- runif(n) %>% as.matrix()
true_mean_fun <- function(xx) sin(10*xx) + b0

# training data
y <- true_mean_fun(x) + rnorm(n = n, mean = 0, sd = 2)

data = data.frame(x = x)

# test data
x_test <- runif(n) %>% as.matrix()

validation_data = data.frame(x = x_test)

y_test <- true_mean_fun(x_test) + rnorm(n = n, sd = 2)
#####################################################################

#####################################################################
# Define a Deep Model
# We use three hidden layers for the location:
deep_model <- function(x) x %>% 
  layer_dense(units = 256, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")
#####################################################################

#####################################################################
# Initialize the model using the function
# provided in deepregression
mod <- deepregression(
  # supply data (response and data.frame for covariates)
  y = y,
  data = data,
  # define how parameters should be modeled
  list_of_formulae = list(loc = ~ 1 + d(x), scale = ~1),
  list_of_deep_models = list(d = deep_model)
)
```

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
# fit model (only do a few iterations for demonstration)
mod %>% fit(epochs=100, verbose = FALSE, view_metrics = FALSE)
# predict
mean <- mod %>% fitted()
true_mean <- true_mean_fun(x) - b0

# compare means
plot(true_mean + b0 ~ x, ylab="partial effect")
points(c(as.matrix(mean)) ~ x, col = "red")
legend("bottomright", col=1:2, pch = 1, legend=c("true mean", "deep prediction"))
```

![](tutorial_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

This is just for demonstration that a neural network can also capture
non-linearities, but often requires a lot of effort to get proper smooth
estimates.

### Deep GAM

We now create a very simple logistic additive regression first where we
try to model the non-linear part of the data generating process using
both a complex neural network and a spline.

``` r
set.seed(24)

# generate the data
n <- 1500
b0 <- 1

# training data; predictor 
x <- runif(n) %>% as.matrix()
true_mean_fun <- function(xx) plogis(sin(10*xx) + b0)

# training data
y <- rbinom(n = n, size = 1, prob = true_mean_fun(x))

data = data.frame(x = x)

# test data
x_test <- runif(n) %>% as.matrix()

validation_data = data.frame(x = x_test)

y_test <- rbinom(n = n, size = 1, prob = true_mean_fun(x_test))
#####################################################################

#####################################################################
# Define a Deep Model
# We use three hidden layers for the location:
deep_model <- function(x) x %>% 
  layer_dense(units = 128, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 8, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")
#####################################################################

#####################################################################
# Initialize the model using the function
# provided in deepregression
mod <- deepregression(
  # supply data (response and data.frame for covariates)
  y = y,
  data = data,
  # define how parameters should be modeled
  list_of_formulae = list(logits = ~ 1 + s(x, bs = "tp") + d(x)),
  list_of_deep_models = list(d = deep_model),
  # family binomial n=1
  family = "bernoulli",
  df = 10 # use no penalization for spline
)
```

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
# fit model, save weights
history <- mod %>% fit(epochs=100, verbose = FALSE, view_metrics = FALSE,
                       save_weights = TRUE)

# plot history of spline - just to see how the spline evolves over iterations
BX <- mod$init_params$parsed_formulae_contents[[1]]$smoothterms$x[[1]]$X
coef_history <- history$weighthistory[-1,]
f_history <- sapply(1:ncol(coef_history), function(j) BX%*%coef_history[,j])
library(ggplot2)
library(reshape2)
df <- melt(cbind(x=x, as.data.frame(f_history)), id.vars="x")
df$variable = as.numeric(df$variable)
ggplot(df, aes(x=x,y=value, colour=as.integer(variable), group=factor(variable))) + 
  geom_line() + 
  scale_colour_gradient(name = "epoch", 
                        low = "blue", high = "red") + 
  ylab("partial effect s(x)") + theme_bw()
```

![](tutorial_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
# since we are usually only interested in the final result,
# all you would have to do is plot(mod) to get the non-linear
# estimate
```

We can check which of the function the cross-validation would have
chosen by doing the following:

``` r
mod <- deepregression(
  # supply data (response and data.frame for covariates)
  y = y,
  data = data,
  # define how parameters should be modeled
  list_of_formulae = list(logits = ~ 1 + s(x, bs = "tp") + d(x)),
  list_of_deep_models = list(d = deep_model),
  # family binomial n=1
  family = "bernoulli",
  df = 10 # use no penalization for spline
)
```

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
cvres <- mod %>% cv(epochs=10, cv_folds = 2) # should be 100
```

    ## Fitting Fold  1  ... 
    ## Done in 1.579051  secs 
    ## Fitting Fold  2  ... 
    ## Done in 1.099053  secs

![](tutorial_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

Set the stopping iteration to the CV optimal value (which in this case
is not optimal at all) and train the whole model again:

``` r
bestiter <- stop_iter_cv_result(cvres)
# fit model
mod %>% fit(epochs=bestiter, verbose = FALSE, view_metrics = FALSE)
# plot model
mod %>% plot()
points(sin(10*(sort(x))) ~ sort(x), col = "red", type="l", ylim=c(0,1))
```

![](tutorial_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

### GAMLSS

We not create a standard GAMLSS model with Gaussian distribution by
modeling the expectation using additive terms and the standard deviation
by a linear term.

``` r
set.seed(24)

# generate the data
n <- 1500
b0 <- 1

# training data; predictor 
x <- runif(n) %>% as.matrix()
z <- runif(n) %>% as.matrix()
true_mean_fun <- function(xx,zz) sin(10*xx) + zz^2 + b0
true_sd_fun <- function(xl) exp(2 * xl)
true_dgp_fun <- function(xx,zz)
{
  
  eps <- rnorm(n) * true_sd_fun(xx)
  y <- true_mean_fun(xx, zz) + eps
  return(y)
  
}

# compose training data with heteroscedastic errors
y <- true_dgp_fun(x,z)
data = data.frame(x = x, z = z)

# test data
x_test <- runif(n) %>% as.matrix()
z_test <- runif(n) %>% as.matrix()

validation_data = data.frame(x = x_test, z = z_test)

y_test <- true_dgp_fun(x_test, z_test)
#####################################################################

#####################################################################
# Define a Deep Model
# We use three hidden layers for the location:
deep_model <- function(x) x %>% 
  layer_dense(units = 128, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 8, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")
#####################################################################

#####################################################################
# Initialize the model using the function
# provided in deepregression
mod <- deepregression(
  # supply data (response and data.frame for covariates)
  y = y,
  data = data,
  # define how parameters should be modeled
  list_of_formulae = list(loc = ~ 1 + s(x, bs="tp") + s(z, bs="tp"),
                          scale = ~ 0 + x),
  list_of_deep_models = list(NULL),
  # family binomial n=1
  family = "normal",
  df = 10
)
```

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
# fit model
mod %>% fit(epochs=100, verbose = FALSE, view_metrics = FALSE)
# summary(mod)
# coefficients
mod %>% coef()
```

    ## [[1]]
    ## [[1]]$structured_nonlinear
    ##              [,1]
    ##  [1,]  1.62146640
    ##  [2,]  0.48366538
    ##  [3,] -2.38512802
    ##  [4,] -0.77210301
    ##  [5,] -0.02427807
    ##  [6,] -0.08192946
    ##  [7,]  0.73934132
    ##  [8,]  0.43060657
    ##  [9,]  0.40252650
    ## [10,]  0.15566026
    ## [11,]  0.01690309
    ## [12,] -0.10902073
    ## [13,] -0.01107935
    ## [14,] -0.46997577
    ## [15,] -0.28232029
    ## [16,] -0.08464167
    ## [17,] -0.22759320
    ## [18,]  0.28605318
    ## [19,]  0.39069128
    ## 
    ## 
    ## [[2]]
    ## [[2]]$structured_linear
    ##          [,1]
    ## [1,] 2.024265

``` r
# plot model
par(mfrow=c(2,2))
plot(sin(10*x) ~ x)
plot(z^2 ~ z)
mod %>% plot()
```

![](tutorial_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
# get fitted values
meanpred <- mod %>% fitted()
par(mfrow=c(1,1))
plot(meanpred[,1] ~ x)
```

![](tutorial_files/figure-gfm/unnamed-chunk-16-2.png)<!-- -->

### Deep GAMLSS

We now extend the example 4 by a Deep model part.

``` r
set.seed(24)

# generate the data
n <- 1500
b0 <- 1

# training data; predictor 
x <- runif(n) %>% as.matrix()
z <- runif(n) %>% as.matrix()
true_mean_fun <- function(xx,zz) sin(10*xx) + zz^2 + b0
true_sd_fun <- function(xl) exp(2 * xl)
true_dgp_fun <- function(xx,zz)
{
  
  eps <- rnorm(n) * true_sd_fun(xx)
  y <- true_mean_fun(xx, zz) + eps
  return(y)
  
}

# compose training data with heteroscedastic errors
y <- true_dgp_fun(x,z)
data = data.frame(x = x, z = z)

# test data
x_test <- runif(n) %>% as.matrix()
z_test <- runif(n) %>% as.matrix()

validation_data = data.frame(x = x_test, z = z_test)

y_test <- true_dgp_fun(x_test, z_test)
#####################################################################

#####################################################################
# Define a Deep Model
# We use three hidden layers for the location:
deep_model <- function(x) x %>% 
  layer_dense(units = 128, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 8, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")
#####################################################################

#####################################################################
# Initialize the model using the function
# provided in deepregression
mod <- deepregression(
  # supply data (response and data.frame for covariates)
  y = y,
  data = data,
  # define how parameters should be modeled
  list_of_formulae = list(loc = ~ 1 + s(x, bs="tp") + d(z),
                          scale = ~ 1 + x),
  list_of_deep_models = list(d = deep_model),
  # family normal
  family = "normal"
)
```

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
# fit model
mod %>% fit(epochs=50, verbose = FALSE, view_metrics = FALSE)
# plot model
mod %>% plot()
```

![](tutorial_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
# get coefficients
mod %>% coef()
```

    ## [[1]]
    ## [[1]]$structured_nonlinear
    ##              [,1]
    ##  [1,]  0.67609823
    ##  [2,] -0.07861418
    ##  [3,] -1.43900073
    ##  [4,] -0.77573609
    ##  [5,] -0.22893305
    ##  [6,] -0.15451908
    ##  [7,]  0.50099188
    ##  [8,]  0.27190116
    ##  [9,] -0.39989737
    ## [10,]  0.44732609
    ## 
    ## 
    ## [[2]]
    ## [[2]]$structured_linear
    ##             [,1]
    ## [1,] 0.009654002
    ## [2,] 2.025900126

### Zero-inflated Poisson Distribution

An example how to fit an access the ZIP distribution. We first create
data and add some dummy covariates.

``` r
# create data
n <- 5000
prob = 0.3
lambda = 2
  
bino <- rbinom(n, size = 1, prob = prob)
y <- 0 * bino + (1-bino) * rpois(n, lambda)
data = data.frame(y=y, x = rnorm(n))
```

Now we fit the distribution and access the fitted parameters

``` r
mod <- deepregression(y, 
                      list_of_formulae = list(rate = ~ 1 + x, 
                                              prob = ~1),
                      data = data,
                      list_of_deep_models = NULL, 
                      family = "zip")
```

    ## Warning in deepregression(y, list_of_formulae = list(rate = ~1 + x, prob =
    ## ~1), : No deep model specified

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
# fit the model
mod %>% fit(epochs = 50, view_metrics=FALSE, verbose=FALSE)

# get distribution
mydist <- mod %>% get_distribution()

# rate for Poisson
as.matrix(mydist$components[[0]]$rate + 0) %>% head()
```

    ##          [,1]
    ## [1,] 1.823505
    ## [2,] 1.778853
    ## [3,] 1.946170
    ## [4,] 1.862115
    ## [5,] 1.753633
    ## [6,] 1.852537

``` r
# probability for inflation / non-inflation
as.array(mydist$cat$probs + 0)[,1,] %>% head()
```

    ##           [,1]      [,2]
    ## [1,] 0.7306829 0.2693171
    ## [2,] 0.7306829 0.2693171
    ## [3,] 0.7306829 0.2693171
    ## [4,] 0.7306829 0.2693171
    ## [5,] 0.7306829 0.2693171
    ## [6,] 0.7306829 0.2693171

## Real World Application

### Deep Mixed Model for Wage Panel Data

This example applies deep distributional regression to the ‘Cornwell and
Rupert’ data, a balanced panel dataset with 595 individuals and 4165
observations, where each individual is observed for 7 years. This data
set is also used in Tran et al. (2018) for within subject prediction of
the log of wage in the years 6 and 7 after training on years 1 to 5.
They report an MSE of 0.05.

``` r
library(dplyr)
data <- read.csv("http://people.stern.nyu.edu/wgreene/Econometrics/cornwell&rupert.csv")
data$ID <- as.factor(data$ID)

train <- data %>% dplyr::filter(YEAR < 6)
test <- data %>% dplyr::filter(YEAR >= 6)

deep_mod <- function(x) x %>% 
  layer_dense(units = 5, activation = "relu", use_bias = FALSE) %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

# expanding window CV
cv_folds <- list(#year1 = list(train = which(train$YEAR==1),
                #              test = which(train$YEAR>1 & train$YEAR<4)),
                 year2 = list(train = which(train$YEAR<=2),
                              test = which(train$YEAR>2 & train$YEAR<5)),
                 year3 = list(train = which(train$YEAR<=3),
                              test = which(train$YEAR>3 & train$YEAR<6)))

# initialize model
mod <- deepregression(y = train$LWAGE,
                      data = train[,c(1:11, 14, 16)], 
                      list_of_formulae = list(~ 1 + s(ID, bs="re") + 
                                                d(EXP, WKS, OCC, IND, SOUTH, YEAR,
                                                  SMSA, MS, FEM, UNION, ED, BLK),
                                              ~ 1),
                      list_of_deep_models = list(d = deep_mod),
                      family = "normal",
                      cv_folds = cv_folds
                        )
```

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
#cvres <- mod %>% cv(epochs = 200)
bestiter <- 10 # stop_iter_cv_result(cvres)
mod %>% fit(epochs = bestiter, view_metrics=FALSE, verbose=FALSE)
pred <- mod %>% predict(test)

mean((pred-test$LWAGE)^2)
```

    ## [1] 0.05144408

### Mixture of Normal Distributions for Acidity Modeling

We here estimate a mixture of three normal distributions for the acidity
data, a data set showing the acidity index for 155 lakes in the
Northeastern United States.

``` r
# load data
library("gamlss.data")
data(acidity)

# softmax function
logsumexp <- function (x) {
  y = max(x)
  y + log(sum(exp(x - y)))
}
softmax <- function (x) {
  exp(x - logsumexp(x))
}



mod <- deepregression(acidity$y-mean(acidity$y), 
                      list_of_formulae = list(~ 1, #mixtures
                                              ~1, ~1, ~1, # means
                                              ~1, ~1, ~1 # sds
                      ),
                      data = acidity,
                      list_of_deep_models = NULL, 
                      mixture_dist = 3,
                      dist_fun = mix_dist_maker())
```

    ## Warning in deepregression(acidity$y - mean(acidity$y), list_of_formulae =
    ## list(~1, : No deep model specified

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
# cvres <- mod %>% cv(epochs = 500, cv_folds = 5)
bestiter <- 49 # stop_iter_cv_result(cvres)
mod %>% fit(epochs = bestiter, 
            validation_split = NULL, 
            view_metrics = FALSE,
            verbose = FALSE)
coefinput <- unlist(mod$model$get_weights())
(means <- coefinput[c(2:4)])
```

    ## [1] -0.04558545 -0.85112649  1.36544180

``` r
(stds <- exp(coefinput[c(5:7)]))
```

    ## [1] 0.8233951 0.2804566 0.3782972

``` r
(pis <- softmax(coefinput[8:10]*coefinput[1]))
```

    ## [1] 0.3105731 0.4275322 0.2618947

``` r
library(distr)

mixDist <- UnivarMixingDistribution(Norm(means[1],stds[1]),
                                    Norm(means[2],stds[2]),
                                    Norm(means[3],stds[3]),
                                    mixCoeff=pis)

plot(mixDist, to.draw.arg="d", ylim=c(0,1.4)) 
with(acidity, hist(y-mean(y), breaks = 100, add=TRUE, freq = FALSE))
```

![](tutorial_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

### Unstructured Data Examples

#### MNIST Multinomial

The non-binarized MNIST example demonstrates the capabilities of the
framework to handle multinomial (or in general multivariate) responses.

``` r
mnist <- keras::dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
dim(x_train) <- c(nrow(x_train), 784)
dim(x_test) <- c(nrow(x_test), 784)
# rescale
x_train <- x_train / 255
x_test <- x_test / 255
# convert to data.frame
x_train <- as.data.frame(x_train)
x_test <- as.data.frame(x_test)
y_train <- keras::to_categorical(y_train)
y_test <- keras::to_categorical(y_test)

# deep model
nn_model <- function(x) x %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10)


mod <- deepregression(y = y_train,
                      list_of_formulae = list(logit = ~ 0 + d(.)),
                      list_of_deep_models = list(d = nn_model),
                      data = x_train,
                      family = "multinomial")
```

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
# model does not need to have many epochs as 
# the 0 is easily detected using some specific pixels
cvres <- mod %>% fit(epochs = 10, validation_split = NULL, steps_per_epoch=1,
                     view_metrics = FALSE, verbose=FALSE)
# currenty has some issues when using actual batch training,
# see also: https://github.com/keras-team/keras/issues/11749
pred <- mod %>% predict(x_test)
table(data.frame(pred=apply(pred,1,which.max)-1, 
                 truth=apply(y_test, 1, function(x) which(x==1))-1
                 )
      )
```

    ##     truth
    ## pred    0    1    2    3    4    5    6    7    8    9
    ##    0  921    0   12    3    1   10   22    2    8   10
    ##    1    0 1116   12    2    3   10    5   27   32    8
    ##    2    6    4  904   25    6    8   10   18   19    2
    ##    3    3    5   34  906    1   74    0    6   57   16
    ##    4    1    0   12    0  866   16    5    4   21   76
    ##    5   24    0    2   26    0  700   13    0   64    6
    ##    6   23    3   18    5   18   19  900    3   19    1
    ##    7    1    1   20   16    0    9    0  909   12   15
    ##    8    1    6   15   15    8   39    3    1  710    8
    ##    9    0    0    3   12   79    7    0   58   32  867

#### Text as Input

We use IMDB Reviews for sentiment analysis to predict 1 = positive or 0
= negative reviews. The example is taken from the [Tensorflow
Blog](https://blogs.rstudio.com/tensorflow/posts/2017-12-07-text-classification-with-keras/)
but just a small example with 1000 words.

``` r
nr_words = 1000
imdb <- keras::dataset_imdb(num_words = nr_words)
train_data <- imdb$train$x
train_labels <- imdb$train$y
test_data <- imdb$test$x
test_labels <- imdb$test$y

word_index <- keras::dataset_imdb_word_index()  
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

decoded_review <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "[...]"
})
cat(decoded_review[2:87])
```

    ## this film was just brilliant casting [...] [...] story direction [...] really [...] the part they played and you could just imagine being there robert [...] is an amazing actor and now the same being director [...] father came from the same [...] [...] as myself so i loved the fact there was a real [...] with this film the [...] [...] throughout the film were great it was just brilliant so much that i [...] the film as soon as it was released for [...]

Do the actual pre-processing and model fitting

``` r
vectorize_sequences <- function(sequences, dimension = nr_words) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension) 
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1 
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# this is how a bidirectional LSTM would look like
# nn_model <- function(x) x %>%
#   layer_embedding(input_dim = nr_words, 
#                   # embedding dimension = 2
#                   # as an example -> yields
#                   # 100*nr_words parameters
#                   # to be estimated
#                   output_dim = 100) %>% 
#   bidirectional(layer = layer_lstm(units = 64)) %>%
#   layer_dense(1)
#
# -> not appropriate here, as we have a simple
# classification task

nn_model <- function(x) x %>% 
  layer_dense(units = 5, activation = "relu", input_shape = c(nr_words)) %>% 
  layer_dense(units = 5, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

mod <- deepregression(y = y_train,
                      list_of_formulae = list(logit = ~ 0 + d(.)),
                      list_of_deep_models = list(d = nn_model),
                      data = as.data.frame(x_train),
                      family = "bernoulli")
```

    ## Preparing additive formula(e)... Done.
    ## Translating data into tensors... Done.

``` r
# as an example only use 3 epochs (lstms usualy need 
# not so many epochs anyway)
mod %>% fit(epochs = 20, view_metrics=FALSE, batch_size = 250, verbose=FALSE)
pred <- mod %>% predict(as.data.frame(x_test))
boxplot(pred ~ y_test,  ylab="Predicted Probability", xlab = "True Label")
```

![](tutorial_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->
