k <- keras:::keras

# splineLayer <- R6::R6Class("splineLayer",
# 
#                            lock_objects = FALSE,
#                            inherit = KerasLayer,
# 
#                            public = list(
# 
#                              output_dim = NULL,
# 
#                              kernel = NULL,
# 
#                              initialize = function(
#                                # input_shape=NULL,
#                                shared_weights=FALSE,
#                                kernel_regularizer=NULL,
#                                use_bias=FALSE,
#                                kernel_initializer='glorot_uniform',
#                                bias_initializer='zeros'#,
#                                # ...
#                              ) {
# 
#                                # super$initialize(...)
# 
#                                # self$input_shape = input_shape
#                                self$shared_weights = shared_weights
#                                self$kernel_regularizer = tf$keras$regularizers$get(
#                                  kernel_regularizer)
#                                self$use_bias = use_bias
#                                self$kernel_initializer = tf$keras$initializers$get(
#                                  kernel_initializer)
#                                self$bias_initializer = tf$keras$initializers$get(
#                                  bias_initializer)
#                                self$input_spec = tf$keras$layers$InputSpec(min_ndim=3)
#                              },
# 
#                              build = function(input_shape) {
# 
#                                stopifnot(length(input_shape) >= 3)
# 
#                                n_bases = input_shape[[length(input_shape)]]
#                                n_features = input_shape[[length(input_shape)-1]]
# 
#                                self$inp_shape = input_shape
#                                self$n_features = n_features
#                                self$n_bases = n_bases
# 
#                                if(self$shared_weights)
#                                  use_n_features = 1 else
#                                    use_n_features = self$n_features
# 
#                                self$kernel =
#                                  self$add_weight(shape=list(n_bases, use_n_features),
#                                                  initializer=self$kernel_initializer,
#                                                  name='kernel',
#                                                  regularizer=self$kernel_regularizer,
#                                                  trainable=TRUE)
# 
#                                if(self$use_bias)
#                                  self$bias =
#                                  self$add_weight(shape = (n_features),
#                                                  # is that the correct pendant to
#                                                  # python's (n_features, )?
#                                                  initializer=self$bias_initializer,
#                                                  name='bias',
#                                                  regularizer=NULL)
# 
#                                self$built = TRUE
# 
#                                super$build(self
#                                            #,input_shape
#                                )
# 
#                              },
# 
#                              call = function(inputs, mask = NULL) {
#                                N = length(self$inp_shape)
# 
#                                if (self$shared_weights)
#                                  return(k_squeeze(k_dot(inputs, self$kernel), -1))
# 
#                                # print("output def")
#                                output =
#                                  k_permute_dimensions(inputs, list(
#                                    N - 1,
#                                    as.list(1:(N-2)), # permute is 1-based
#                                    N
#                                  )
#                                  )
# 
#                                # print("output_reshaped def")
# 
#                                sh = list(self$n_features, -1, self$n_bases)
#                                output_reshaped =
#                                  k_reshape(output, sh)
# 
#                                # print(str(output_reshaped,1))
# 
#                                axes = NULL
#                                # # added due to missing translation
#                                # # of the argument axes in k_batch_dot
#                                # # -_____-
#                                # tKernel = k_transpose(self$kernel)
#                                # x_shape = k_int_shape(output_reshaped)
#                                # y_shape = k_int_shape(tKernel)
#                                # x_ndim = length(x_shape)
#                                # y_ndim = length(y_shape)
#                                # if(y_ndim == 2)
#                                #   axes = list(x_ndim - 1, y_ndim - 1) else
#                                #     axes = list(x_ndim - 1, y_ndim - 2)
#                                # #####################################
# 
#                                # print("bd_output def")
# 
#                                bd_output =
#                                  k_batch_dot(output_reshaped,
#                                              k_transpose(self$kernel),
#                                              axes = axes)
# 
#                                # this one is tricky to translate
#                                # from python. Not sure if this is
#                                # correct
#                                sh = list(self$n_features, -1)
#                                if(length(self$inp_shape) > 3){
#                                  sh = append(sh, self$inp_shape[2:(N - 2)])
#                                }
# 
#                                # print("output def 1")
# 
#                                output = k_reshape(bd_output, sh)
#                                # move axis 0 (features) to back
# 
#                                # print("output def 2")
# 
#                                # print(str(output,1))
# 
# 
#                                output =
#                                  k_permute_dimensions(output,
#                                                       c(as.list(2:(N-1)), list(1)))
# 
#                                # print("done")
# 
#                                # permute is 1-based
#                                if(self$use_bias)
#                                  output = k_bias_add(output,
#                                                      self$bias, data_format="channels_last")
#                                return(output)
#                              },
# 
#                              compute_output_shape = function(input_shape) {
#                                input_shape[-length(input_shape)]
#                              },
# 
#                              get_config = function()
#                              {
#                                config = list(
#                                  'shared_weights' = self$shared_weights,
#                                  'kernel_regularizer' = tf$keras$regularizers$serialize(
#                                    self$kernel_regularizer),
#                                  'use_bias' = self$use_bias,
#                                  'kernel_initializer' =
#                                    tf$keras$initializers$serialize(self$kernel_initializer),
#                                  'bias_initializer' =
#                                    tf$keras$initializers$serialize(self$bias_initializer)
#                                )
#                                base_config = super$get_config()
#                                # is this the correct analogous way to
#                                # python's combining dicts?
#                                return(c(base_config, config))
#                              }
# 
#                            )
# )

### instantiate spline layer


#' Wrapper function to create spline layer
#'
#' @param object or layer object.
#' @param name An optional name string for the layer (should be unique).
#' @param trainable logical, whether the layer is trainable or not.
#' @param input_shape Input dimensionality not including samples axis.
#' @param lambdas vector of smoothing parameters
#' @param Ps list of penalty matrices
#' @param use_bias whether or not to use a bias in the layer. Default is FALSE.
#' @param kernel_initializer function to initialize the kernel (weight). Default
#' is "glorot_uniform".
#' @param bias_initializer function to initialize the bias. Default is 0.
#' @param variational logical, if TRUE, priors corresponding to the penalties
#' and posteriors as defined in \code{posterior_fun} are created
#' @param diffuse_scale diffuse scale prior for scalar weights
#' @param posterior_fun function defining the variational posterior
#' @param output_dim the number of units for the layer
#' @param ... further arguments passed to \code{args} used in \code{create_layer}
#' 
#' 
layer_spline <- function(object,
                         name = NULL,
                         trainable = TRUE,
                         input_shape,
                         lambdas,
                         Ps,
                         use_bias = FALSE,
                         kernel_initializer = 'glorot_uniform',
                         bias_initializer = 'zeros',
                         variational = FALSE,
                         # prior_fun = NULL,
                         posterior_fun = NULL,
                         diffuse_scale = 1000,
                         output_dim = 1L,
                         ...) {
  
  if(variational){
    bigP = bdiag(lapply(1:length(Ps), function(j){ 
      # return vague prior for scalar
      if(length(Ps[[j]])==1) return(diffuse_scale^2) else
        return(chol2inv(chol(lambdas[j] * Ps[[j]])))})) 
  }else{
      bigP = bdiag(lapply(1:length(Ps), function(j) lambdas[j] * Ps[[j]]))
  }
  
  if(sum(lambdas)==0 | variational)
    regul <- NULL else
      regul <- function(x)
        k_sum(k_batch_dot(x, k_dot(
          # tf$constant(
          sparse_mat_to_tensor(bigP),
          # dtype = "float32"),
          x),
          axes=2) # 1-based
        )
    
    args <- c(list(input_shape = input_shape),
              name = name,
              units = output_dim,
              trainable = trainable,
              kernel_regularizer=regul,
              use_bias=use_bias,
              list(...))
    
    if(variational){
      
      class <- tfprobability::tfp$layers$DenseVariational 
      args$make_posterior_fn = posterior_fun
      args$make_prior_fn = function(kernel_size,
                                    bias_size = 0L,
                                    dtype) prior_pspline(kernel_size = kernel_size,
                                                         bias_size = bias_size,
                                                         dtype = 'float32',
                                                         P = as.matrix(bigP))
      args$regul <- NULL
      
    }else{
      
      class <- k$layers$Dense
      args$kernel_initializer=kernel_initializer
      args$bias_initializer=bias_initializer
    
    }

    create_layer(layer_class = class,
                 object = object,
                 args = args
    )
}

# layer_spline <- function(object,
#                          name = NULL,
#                          trainable = TRUE,
#                          input_shape,
#                          shared_weights=FALSE,
#                          lambdas,
#                          Ps,
#                          use_bias = FALSE,
#                          kernel_initializer = 'glorot_uniform',
#                          bias_initializer = 'zeros',
#                          ...) {
#   
#   bigP = bdiag(lapply(1:length(Ps), function(j) lambdas[j] * Ps[[j]]))
#   
#   if(sum(lambdas)==0)
#     regul <- NULL else
#       regul <- function(x)
#         k_mean(k_batch_dot(x, k_dot(
#           # tf$constant(
#           sparse_mat_to_tensor(bigP),
#           # dtype = "float32"),
#           x),
#           axes=2) # 1-based
#         )
#     
#     create_layer(layer_class = splineLayer,
#                  object = object,
#                  args = c(list(input_shape = input_shape),
#                           name = name,
#                           trainable = trainable,
#                           shared_weights=shared_weights,
#                           kernel_regularizer=regul,
#                           use_bias=use_bias,
#                           kernel_initializer=kernel_initializer,
#                           bias_initializer=bias_initializer,
#                           list(...))
#     )
# }

#### get layer based on smoothCon object
get_layers_from_s <- function(this_param, nr=NULL, variational=FALSE,
                              posterior_fun=NULL, trafo=FALSE, #, prior_fun=NULL
                              output_dim = 1
                              )
{

  if(is.null(this_param)) return(NULL)

  lambdas <- c()
  Ps = list()
  params = 0

  # create vectors of lambdas and list of penalties
  if(!is.null(this_param$linterms)){
    lambdas = c(lambdas, rep(0, ncol_lint(this_param$linterms)))
    Ps = c(Ps, list(0)[rep(1, ncol_lint(this_param$linterms))])
    params = ncol(this_param$linterms)
  }
  if(!is.null(this_param$smoothterms)){
    these_lambdas = unlist(sapply(this_param$smoothterms, function(x) x[[1]]$sp))
    lambdas = c(lambdas, these_lambdas)
    these_Ps = lapply(this_param$smoothterms, function(x) if(length(x)==1) x[[1]]$S else 
      list(Matrix::bdiag(lapply(x,function(y)y$S[[1]]))))
    is_TP <- sapply(these_Ps, length) > 1
    if(any(is_TP))
      these_Ps[which(is_TP)] <- lapply(these_Ps[which(is_TP)],
                                       function(x){

                                         return(
                                          # isotropic smoothing
                                          # TODO: Allow f anisotropic smoothing
                                           x[[1]] + x[[2]]
                                          # kronecker(x[[1]],
                                          #           diag(ncol(x[[2]]))) +
                                          #   kronecker(diag(ncol(x[[1]])), x[[2]])
                                         )
                                       })
    s_as_list <- sapply(these_Ps, class)=="list"
    if(any(s_as_list)){
      for(i in which(s_as_list)){
        if(length(these_Ps[i])> 1)
          stop("Can not deal with penalty of smooth term ", names(these_Ps)[i])
        these_Ps[i] <- these_Ps[i][[1]]
      }
    }
    Ps = c(Ps, these_Ps)
    params = params + sum(sapply(these_Ps, NCOL))
  }else{
    # only linear terms
    # name <- "linear"
    # if(!is.null(nr)) name <- paste(name, nr, sep="_")
    return(
      # layer_dense(input_shape = list(params),
      #             units = 1,
      #             use_bias = FALSE,
      #             name = name)
      NULL
    )
  }

  name <- "structured_nonlinear"
  if(!is.null(nr)) name <- paste(name, nr, sep="_")
  
  if(trafo){
    
    return(bdiag(lapply(1:length(Ps), function(j) lambdas[j] * Ps[[j]])))
    
  }

  layer_spline(input_shape = list(as.integer(params)),
               # the one is just an artifact from concise
               name = name,
               lambdas = lambdas,
               Ps = Ps,
               variational = variational,
               posterior_fun = posterior_fun,
               output_dim = output_dim)

}