eval_bsp <- function(y, order = 3, supp = range(y)) {
  
  # Evaluate a Bernstein Polynom (bsp) with a predefined order on a predefined 
  # support that is used for scaling since a Beta distribution is defined on (0,1).
  # MLT Vignette p. 9
  #
  # y: numeric vector of length n
  # order: postive integer which is called M in the literature
  # supp: numeric vector of length 2 that is used to scale y down to (0,1)
  #
  # returns a numeric matrix (n x (order + 1))
  
  y <- (y - supp[1]) / diff(supp)
  sapply(0:order, function(m) dbeta(y, m + 1, order + 1 - m) / (order + 1))
  
}

eval_bsp_prime <- function(y, order = 3, supp = range(y)) {
  
  # Evaluate the first derivative of the bsp with a predefined order on a predefined 
  # support that is used for scaling since a Beta distribution is defined on (0,1).
  # MLT Vignette p. 9. Note that "order" cancels out and that (1/diff(y_var$support)^deriv)
  # is multiplied afterwards. This is only due to numerical reasons to get the 
  # exact same quantities as mlt::mlt. Furthermore, order/(order - 1 + 1) cancels
  # out in the mlt::mlt implementation which is not as stated on p. 9.
  #
  # y: numeric vector of length n
  # order: postive integer which is called M in the literature
  # supp: numeric vector of length 2 that is used to scale y down to (0,1)
  #
  # returns a numeric matrix (n x (order + 1))
  
  y <- (y - supp[1]) / diff(supp)
  sapply(0:order, function(m) {
    
    first_t <- dbeta(y, m, order - m + 1) / order
    sec_t <- dbeta(y, m + 1, order - m) / order
    
    first_t[is.infinite(first_t)] <- 0L
    sec_t[is.infinite(sec_t)] <- 0L
    
    (first_t - sec_t) * order
  })
}

# TensorFlow repeat function which is not available for TF 2.0
tf_repeat <- function(a, dim)
  tf$reshape(tf$tile(tf$expand_dims(a, axis = -1L),  c(1L, 1L, dim)), shape = list(-1L, ncol(a)[[1]]*dim))

# Row-wise tensor product using TensorFlow
tf_row_tensor <- function(a,b)
{
  tf$multiply(
    tf_repeat(a, ncol(b)[[1]]),  
    tf$tile(b, c(1L, ncol(a)[[1]]))
    
  )
}

###############################################################################################
# for trafo with interacting features

mono_trafo_multi <- function(w, bsp_dim) 
{
  
  w_res <- tf$reshape(w, shape = list(bsp_dim, as.integer(nrow(w)/bsp_dim)))
  w1 <- tf$slice(w_res, c(0L,0L), size=c(1L,ncol(w_res)))
  wrest <- tf$math$softplus(tf$slice(w_res, c(1L,0L), size=c(as.integer(nrow(w_res)-1),ncol(w_res))))
  w_w_cons <- tf$cumsum(k_concatenate(list(w1,wrest), axis = 1L), axis=0L)
  return(tf$reshape(w_w_cons, shape = list(nrow(w),1L)))
  
}

MonoMultiLayer <- R6::R6Class("MonoMultiLayer",
                              
                              inherit = KerasLayer,
                              
                              public = list(
                                
                                output_dim = NULL,
                                
                                kernel = NULL,
                                
                                dim_bsp = NULL,
                                
                                initialize = function(output_dim, dim_bsp) {
                                  self$output_dim <- output_dim
                                  self$dim_bsp <- dim_bsp
                                },
                                
                                build = function(input_shape) {
                                  self$kernel <- self$add_weight(
                                    name = 'kernel', 
                                    shape = list(input_shape[[2]], self$output_dim),
                                    initializer = initializer_random_normal(),
                                    trainable = TRUE
                                  )
                                },
                                
                                call = function(x, mask = NULL) {
                                  tf$matmul(x, mono_trafo_multi(self$kernel, self$dim_bsp))
                                },
                                
                                compute_output_shape = function(input_shape) {
                                  list(input_shape[[1]], self$output_dim)
                                }
                              )
)

# define layer wrapper function
layer_mono_multi <- function(object, 
                             input_shape = NULL,
                             output_dim = 1L,
                             dim_bsp = NULL,
                             name = "constraint_mono_layer_multi", 
                             trainable = TRUE
) {
  create_layer(MonoMultiLayer, object, list(
    name = name,
    trainable = trainable,
    input_shape = input_shape,
    output_dim = as.integer(output_dim),
    dim_bsp = as.integer(dim_bsp)
  ))
}

# to retrieve the weights on their original scale
softplus <- function(x) log(exp(x)+1)
reshape_softplus_cumsum <- function(x, order_bsp_p1)
{
  
  x <- matrix(x, nrow = order_bsp_p1, byrow=T)
  x[2:nrow(x),] <- softplus(x[2:nrow(x),])
  apply(x, 2, cumsum)
  
}