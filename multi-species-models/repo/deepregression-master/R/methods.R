#' @title Generic functions for deepregression models
#'
#' @param x deepregression object
#' @param which which effect to plot, default selects all.
#' @param which_param integer of length 1.
#' Corresponds to the distribution parameter for 
#' which the effects should be plotted.
#' @param plot logical, if FALSE, only the data for plotting is returned
#' @param use_posterior logical; if \code{TRUE} it is assumed that
#' the strucuted_nonlinear layer has stored a list of length two
#' as weights, where the first entry is a vector of mean and sd
#' for each network weight. The sd is transformed using the \code{exp} function.
#' The plot then shows the mean curve +- 2 times sd.
#' @param grid_length the length of an equidistant grid at which a two-dimensional function
#' is evaluated for plotting.
#' @param ... further arguments, passed to fit, plot or predict function
#'
#' @method plot deepregression
#' @export
#' @rdname methodDR
#'
plot.deepregression <- function(
  x,
  which = NULL,
  # which of the nonlinear structured effects
  which_param = 1, # for which parameter
  plot = TRUE,
  use_posterior = FALSE,
  grid_length = 40,
  ... # passed to plot function
)
{
  this_ind <- x$init_params$ind_structterms[[which_param]]
  if(all(this_ind$type!="smooth")) return("No smooth effects. Nothing to plot.")
  if(is.null(which)) which <- 1:length(which(this_ind$type=="smooth"))
  plus_number_lin_eff <- sum(this_ind$type=="lin")
  
  plotData <- vector("list", length(which))
  org_feature_names <- 
    names(x$init_params$l_names_effets[[which_param]][["smoothterms"]])
  phi <- x$model$get_layer(paste0("structured_nonlinear_",
                                  which_param))$get_weights()
  if(length(phi)>1){
    if(use_posterior){
      phi <- matrix(phi[[1]], ncol=2, byrow=F)
    }else{
      phi <- as.matrix(phi[[2]], ncol=1)
    }
  }else{
    phi <- phi[[1]]
  }
  
  for(w in which){
    
    nam <- org_feature_names[w]
    this_ind_this_w <- do.call("Map",
                               c(":", as.list(this_ind[w+plus_number_lin_eff,
                                                       c("start","end")])))[[1]]
    BX <- 
      x$init_params$parsed_formulae_contents[[
        which_param]]$smoothterms[[nam]][[1]]$X
    if(use_posterior){
      
      # get the correct index as each coefficient has now mean and sd
      phi_mean <- phi[this_ind_this_w,1]
      phi_sd <- log(exp(log(expm1(1)) + phi[this_ind_this_w,2])+1)
      plotData[[w]] <- 
        list(org_feature_names = nam,
             value = unlist(x$init_params$data[strsplit(nam,",")[[1]]]),
             design_mat = BX,
             coef = phi[this_ind_this_w,],
             mean_partial_effect = BX%*%phi_mean,
             sd_partial_effect = sqrt(diag(BX%*%diag(phi_sd^2)%*%t(BX))))
    }else{
      plotData[[w]] <-
        list(org_feature_name = nam,
             value = sapply(strsplit(nam,",")[[1]], function(xx)
               x$init_params$data[[xx]]),
             design_mat = BX,
             coef = phi[this_ind_this_w,],
             partial_effect = BX%*%phi[this_ind_this_w,])
    }
    if(plot){
      nrcols <- pmax(NCOL(plotData[[w]]$value), length(unlist(strsplit(nam,","))))
      if(nrcols==1)
      {
        if(use_posterior){
          plot(plotData[[w]]$mean_partial_effect[order(plotData[[w]]$value)] ~ 
                 sort(plotData[[w]]$value),
               main = paste0("s(", nam, ")"),
               xlab = nam,
               ylab = "partial effect",
               ylim = c(min(plotData[[w]]$mean_partial_effect - 
                              2*plotData[[w]]$sd_partial_effect),
                        max(plotData[[w]]$mean_partial_effect + 
                              2*plotData[[w]]$sd_partial_effect)),
               ...)
          with(plotData[[w]], {
            points((mean_partial_effect + 2 * sd_partial_effect)[order(plotData[[w]]$value)] ~
                     sort(plotData[[w]]$value), type="l", lty=2)
            points((mean_partial_effect - 2 * sd_partial_effect)[order(plotData[[w]]$value)] ~
                     sort(plotData[[w]]$value), type="l", lty=2)
          })
        }else{
          plot(partial_effect[order(value),1] ~ sort(value[,1]),
               data = plotData[[w]][c("value", "partial_effect")],
               main = paste0("s(", nam, ")"),
               xlab = nam,
               ylab = "partial effect",
               ...)
        }
      }else if(nrcols==2){
        sTerm <- x$init_params$parsed_formulae_contents[[which_param]]$smoothterms[[w]][[1]]
        this_x <- do.call(seq, c(as.list(range(plotData[[w]]$value[,1])),
                                 list(l=grid_length)))
        this_y <- do.call(seq, c(as.list(range(plotData[[w]]$value[,2])),
                                 list(l=grid_length)))
        df <- as.data.frame(expand.grid(this_x,
                                        this_y))
        colnames(df) <- sTerm$term
        pred <- PredictMat(sTerm, data = df)%*%phi[this_ind_this_w,]
        #this_z <- plotData[[w]]$partial_effect
        filled.contour(
          this_x,
          this_y,
          matrix(pred, ncol=length(this_y)),
          ...,
          xlab = colnames(df)[1],
          ylab = colnames(df)[2],
          # zlab = "partial effect",
          main = sTerm$label
        )
        # warning("Plotting of effects with ", nrcols, " 
        #         covariate inputs not supported yet.")
      }else{
        warning("Plotting of effects with ", nrcols, 
                " covariate inputs not supported.")
      }
    }
  }
  
  invisible(plotData)
}

#' @title Generic functions for deeptrafo models
#'
#' @param x deepregression object
#' @param which which effect to plot, default selects all.
#' @param which_param integer, either 1 or 2.
#' 1 corresponds to the shift term, 2 to the interaction term.
#' @param plot logical, if FALSE, only the data for plotting is returned
#' @param grid_length the length of an equidistant grid at which a two-dimensional function
#' is evaluated for plotting.
#' @param ... further arguments, passed to fit, plot or predict function
#'
#' @method plot deeptrafo
#' @export
#' @rdname methodTrafo
#'
plot.deeptrafo <- function(
  x,
  which = NULL,
  # which of the nonlinear structured effects
  which_param = 1, # for which parameter
  plot = TRUE,
  grid_length = 40,
  ... # passed to plot function
)
{
  this_ind <- x$init_params$ind_structterms[[which_param]]
  if(all(this_ind$type!="smooth")) return("No smooth effects. Nothing to plot.")
  if(is.null(which)) which <- 1:length(which(this_ind$type=="smooth"))
  plus_number_lin_eff <- sum(this_ind$type=="lin")
  
  plotData <- vector("list", length(which))
  org_feature_names <- 
    names(x$init_params$l_names_effets[[which_param]][["smoothterms"]])
  if(which_param==1){
    phi <- matrix(get_shift(x), ncol=1)
  }else{
    phi <- t(get_theta(x))
  }
  
  for(w in which){
    
    nam <- org_feature_names[w]
    this_ind_this_w <- do.call("Map",
                               c(":", as.list(this_ind[w+plus_number_lin_eff,
                                                       c("start","end")])))[[1]]
    BX <- 
      x$init_params$parsed_formulae_contents[[
        which_param]]$smoothterms[[nam]][[1]]$X
    
    plotData[[w]] <-
      list(org_feature_name = nam,
           value = sapply(strsplit(nam,",")[[1]], function(xx)
             x$init_params$data[[xx]]),
           design_mat = BX,
           coef = phi[this_ind_this_w,],
           partial_effects = BX%*%phi[this_ind_this_w,])
    
    if(plot){
      if(which_param==1){
        nrcols <- pmax(NCOL(plotData[[w]]$value), length(unlist(strsplit(nam,","))))
        if(nrcols==1)
        {
          
          plot(partial_effects[order(value),1] ~ sort(value[,1]),
               data = plotData[[w]][c("value", "partial_effects")],
               main = paste0("s(", nam, ")"),
               xlab = nam,
               ylab = "partial effect",
               ...)
          
        }else if(nrcols==2){
          sTerm <- x$init_params$parsed_formulae_contents[[which_param]]$smoothterms[[w]][[1]]
          this_x <- do.call(seq, c(as.list(range(plotData[[w]]$value[,1])),
                                   list(l=grid_length)))
          this_y <- do.call(seq, c(as.list(range(plotData[[w]]$value[,2])),
                                   list(l=grid_length)))
          df <- as.data.frame(expand.grid(this_x,
                                          this_y))
          colnames(df) <- sTerm$term
          pred <- PredictMat(sTerm, data = df)%*%phi[this_ind_this_w,]
          #this_z <- plotData[[w]]$partial_effect
          filled.contour(
            this_x,
            this_y,
            matrix(pred, ncol=length(this_y)),
            ...,
            xlab = colnames(df)[1],
            ylab = colnames(df)[2],
            # zlab = "partial effect",
            main = sTerm$label
          )
          # warning("Plotting of effects with ", nrcols, " 
          #         covariate inputs not supported yet.")
        }else{
          warning("Plotting of effects with ", nrcols, 
                  " covariate inputs not supported.")
        }
      }else{ # plot effects in theta
        
        matplot(
          #sort(plotData[[w]]$value[,1]),
          #1:ncol(plotData[[w]]$partial_effects),
          x=sort(plotData[[w]]$value[,1]),
          y=(plotData[[w]]$partial_effects[order(plotData[[w]]$value[,1]),]),
          ...,
          xlab = plotData[[w]]$org_feature_name,
          ylab = paste0("partial effects ", plotData[[w]]$org_feature_name),
          # zlab = "partial effect",
          type = "l"
        )
        
      }
    }
  }
  
  invisible(plotData)
}


#' Function to prepare data for deepregression use
#' 
#' @export
#' 
#' @param x a deepregression object
#' @param data a data.frame or list
#' @param pred logical, where the data corresponds to a prediction task
#' 
#' @rdname methodDR
#'
prepare_data <- function(
  x,
  data,
  pred=TRUE
)
{
  if(length(data)>1 & is.list(data) & !is.null(x$init_params$offset))
  {
    
    message("Using the second list item of data as offset.")
    newdata_processed <- prepare_newdata(
      x$init_params$parsed_formulae_contents,
      data[[1]], pred=pred)
    if(is.list(data[[2]]))
      data[[2]] <- unlist(data[[2]], recursive = FALSE)
    newdata_processed <- c(newdata_processed,
                           tf$constant(matrix(data[[2]], ncol = 1), 
                                       dtype="float32"))
    
  }else{
    
    newdata_processed <- prepare_newdata(
      x$init_params$parsed_formulae_contents,
      data, pred=pred)
    
  }
  return(newdata_processed)
}

#' Predict based on a deepregression object
#' 
#' @param object a deepregression model
#' @param newdata optional new data, either data.frame or list
#' @param apply_fun which function to apply to the predicted distribution,
#' per default \code{tfd_mean}, i.e., predict the mean of the distribution
#' @param convert_fun how should the resulting tensor be converted,
#' per default \code{as.matrix}
#'
#' @export
#' @rdname methodDR
#'
predict.deepregression <- function(
  object,
  newdata = NULL,
  apply_fun = tfd_mean,
  convert_fun = as.matrix,
  ...
)
{
  
  if(is.null(newdata)){
    yhat <- object$model(unname(object$init_params$input_cov))
  }else{
    # preprocess data
    if(is.data.frame(newdata)) newdata <- as.list(newdata)
    newdata_processed <- prepare_data(object, newdata, pred=TRUE)
    yhat <- object$model(newdata_processed)
  }
  
  
  if(!is.null(apply_fun))
    return(convert_fun(apply_fun(yhat))) else
      return(convert_fun(yhat))
  
}

#' Predict based on a deeptrafo object
#' 
#' @param object a deeptrafo model
#' @param newdata optional new data, either data.frame or list
#' @param ... not used atm
#' @return returns a function with two parameters: the actual response
#' and \code{type} in \code{c('trafo', 'pdf', 'cdf')} determining the
#' returned value
#'
#' @export
#' @rdname methodDR
#'
predict.deeptrafo <- function(
  object,
  newdata = NULL,
  ...
)
{
  
  if(is.null(newdata)){
    inpCov <- unname(object$init_params$input_cov)
  }else{
    # preprocess data
    if(is.data.frame(newdata)) newdata <- as.list(newdata)
    inpCov <- prepare_data(object, newdata, pred=TRUE)
    inpCov <- c(inpCov, list(NULL), list(NULL))
  }
  
  trafo_fun <- function(y, type = c("trafo", "pdf", "cdf"), grid = FALSE)
  {
    type <- match.arg(type)
    
    ay <- tf$cast(object$init_params$y_basis_fun(y), tf$float32)
    aPrimey <- tf$cast(object$init_params$y_basis_fun_prime(y), tf$float32)
    inpCov[length(inpCov)-c(1,0)] <- list(ay, aPrimey)
    mod_output <- object$model(list(inpCov, tf$cast(matrix(y, ncol=1), tf$float32)))
    w_eta <- mod_output[, 1, drop = FALSE]
    aTtheta <- mod_output[, 2, drop = FALSE]
    ytransf <- aTtheta + w_eta
    yprimeTrans <- mod_output[, 3, drop = FALSE]
    theta <- get_theta(object)
    if(grid)
    {
      
      grid_eval <- t(as.matrix(
        tf$matmul(inpCov[[2]],
                  tf$transpose(
                    tf$matmul(ay,
                              tf$cast(theta, tf$float32)
                    )))))
      grid_eval <- grid_eval + 
        t(as.matrix(w_eta)[,rep(1,nrow(grid_eval))])
      
      if(type=="pdf") 
        grid_prime_eval <- t(as.matrix(
          tf$matmul(inpCov[[2]],
                    tf$transpose(
                      tf$matmul(aPrimey,
                                tf$cast(theta, tf$float32)
                      )))))
      
      
    }
    
    if(grid) type <- paste0("grid_",type)
    
    ret <- switch (type,
                   trafo = (ytransf %>% as.matrix),
                   pdf = ((tfd_normal(0,1) %>% tfd_prob(ytransf) %>% 
                             as.matrix)*as.matrix(yprimeTrans)),
                   cdf = (tfd_normal(0,1) %>% tfd_cdf(ytransf) %>% 
                            as.matrix),
                   grid_trafo = grid_eval,
                   grid_pdf = ((tfd_normal(0,1) %>% tfd_prob(grid_eval) %>% 
                                  as.matrix)*as.matrix(grid_prime_eval)),
                   grid_cdf = (tfd_normal(0,1) %>% tfd_cdf(grid_eval) %>% 
                                 as.matrix)
    )
    
    return(ret)
    
  }
  
  return(trafo_fun)
  
}


#' Function to extract fitted distribution
#' 
#' @param object a deepregression object
#' @param apply_fun function applied to fitted distribution,
#' per default \code{tfd_mean}
#' @param ... further arguments passed to the predict function
#' 
#' @export
#' @rdname methodDR
#'
fitted.deepregression <- function(
  object, apply_fun = tfd_mean, ...
)
{
  return(
    predict.deepregression(object, apply_fun=apply_fun, ...)
  )
}

#' Generic fit function
#' 
#' @param x object
#' @param ... further arguments passed to the class-specific function
#' 
#' @export
fit <- function (x, ...) {
  UseMethod("fit", x)
}

#' Fit a deepregression model
#'
#' @param x a deepregresison object.
#' @param early_stopping logical, whether early stopping should be user.
#' @param verbose logical, whether to print losses during training.
#' @param view_metrics logical, whether to trigger the Viewer in RStudio / Browser.
#' @param patience integer, number of rounds after which early stopping is done.
#' @param save_weights logical, whether to save weights in each epoch.
#' @param auc_callback logical, whether to use a callback for AUC
#' @param val_data optional specified validation data
#' @param callbacks a list of callbacks for fitting
#' @param ... further arguments passed to 
#' \code{keras:::fit.keras.engine.training.Model}
#' 
#' 
#' @method fit deepregression
#' @export
#' @rdname methodDR
#'
fit.deepregression <- function(
  x,
  early_stopping = FALSE,
  verbose = TRUE, 
  view_metrics = FALSE,
  patience = 20,
  save_weights = FALSE,
  auc_callback = FALSE,
  val_data = NULL,
  callbacks = list(),
  ...
)
{
  
  # make callbacks 
  if(save_weights){
    weighthistory <- WeightHistory$new()
    callbacks <- append(callbacks, weighthistory)
  }
  if(early_stopping)
    callbacks <- append(callbacks, 
                        callback_early_stopping(patience = patience))
  
  if(auc_callback){
    
    if(is.null(val_data)) stop("Must provide validation data via argument val_data.")
    if(is.data.frame(val_data[[1]])) val_data[[1]] <- as.list(val_data[[1]])
    val_data_processed <- prepare_data(x, val_data[[1]], pred=TRUE)
    
    auc_cb <- auc_roc$new(training = list(unname(x$init_params$input_cov), x$init_params$y),
                          validation = list(unname(val_data_processed), val_data[[2]]))
    callbacks <- append(callbacks,
                        auc_cb)
    verbose <- FALSE
  }
  # if(monitor_weights){
  #   # object$history <- WeightHistory$new()
  #   weight_callback <- callback_lambda(
  #     on_epoch_begin = function(epoch, logs) 
  #       coef_prior_to_epoch <<- unlist(coef(object)),
  #     on_epoch_end = function(epoch, logs) 
  #       print(sum(abs(coef_prior_to_epoch-unlist(coef(object)))))
  #     )
  #   this_callbacks <- append(this_callbacks, weight_callback)
  # }
  
  if(is.null(x$init_params$input_cov)){
    
    input_x <- prepare_newdata(x$init_params$parsed_formulae_contents,
                               x$init_params$data,
                               pred = FALSE)
    if(!is.null(x$init_params$offset))
      input_x <- c(input_x, unlist(lapply(x$init_params$offset, function(yy)
        tf$constant(matrix(yy, ncol = 1), dtype="float32")), recursive = FALSE))
    
  }else{
    
    input_x <- x$init_params$input_cov
    
  }
  
  
  args <- list(...)
  input_list_model <- 
    list(object = x$model,
         x = input_x,
         y = x$init_params$y,
         validation_split = x$init_params$validation_split,
         validation_data = x$init_params$validation_data,
         callbacks = callbacks,
         verbose = verbose,
         view_metrics = view_metrics
    )
  args <- append(args, 
                 input_list_model[!names(input_list_model) %in% 
                                    names(args)])
  if(length(x$init_params$ellipsis)>0)
    args <- append(args, 
                   x$init_params$ellipsis[
                     !names(x$init_params$ellipsis) %in% names(args)])
  
  
  ret <- do.call(fit_fun,
                 args)
  if(save_weights) ret$weighthistory <- weighthistory$weights_last_layer
  invisible(ret)
}

#' Extract layer weights / coefficients from model
#'
#' @param object a deepregression model
#' @param variational logical, if TRUE, the function takes into account
#' that coefficients have both a mean and a variance
#'
#' @method coef deepregression
#' @export
#' @rdname methodDR
#'
coef.deepregression <- function(
  object,
  variational = FALSE,
  ...
)
{
  nrparams <- length(object$init_params$parsed_formulae_contents)
  layer_names <- sapply(object$model$layers, "[[", "name")
  lret <- vector("list", nrparams)
  names(lret) <- object$init_params$param_names
  for(i in 1:nrparams){
    sl <- paste0("structured_linear_",i)
    slas <- paste0("structured_lasso_",i)
    snl <- paste0("structured_nonlinear_",i)
    lret[[i]] <- list(structured_linear = NULL,
                      structured_lasso = NULL,
                      structured_nonlinear = NULL)
    
    lret[[i]]$structured_linear <-
      if(sl %in% layer_names)
        object$model$get_layer(sl)$get_weights()[[1]] else
          NULL
    lret[[i]]$structured_lasso <-
      if(slas %in% layer_names)
        object$model$get_layer(slas)$get_weights()[[1]] else
          NULL
    if(snl %in% layer_names){
      cf <- object$model$get_layer(snl)$get_weights()
      if(length(cf)==2 & variational){
        lret[[i]]$structured_nonlinear <-  cf[[1]]
      }else{
        lret[[i]]$structured_nonlinear <- cf[[length(cf)]]
      }
    }else{
      lret[[i]]$structured_nonlinear <- NULL
    }
    
  }
  return(lret)
  
}

#' Print function for deepregression model
#' 
#' @export
#' @rdname methodDR
#' @param x a \code{deepregression} model
#' @param ... unused
#' 
#' @method print deepregression
#'
print.deepregression <- function(
  x,
  ...
)
{
  print(x$model)
}

#' @title Cross-validation for deepgression objects
#' @param ... further arguments passed to 
#' \code{keras:::fit.keras.engine.training.Model}
#' @param x deepregression object
#' @param verbose whether to print training in each fold
#' @param patience number of patience for early stopping
#' @param plot whether to plot the resulting losses in each fold
#' @param print_folds whether to print the current fold
#' @param mylapply lapply function to be used; defaults to \code{lapply}
#' @param save_weights logical, whether to save weights in each epoch.
#' @param cv_folds see \code{deepregression}
#' @param stop_if_nan logical; whether to stop CV if NaN values occur
#' @export
#' @rdname methodDR
#' 
#' @return Returns an object \code{drCV}, a list, one list element for each fold
#' containing the model fit and the \code{weighthistory}.
#' 
#'
#' 
cv <- function(
  x,
  verbose = FALSE, 
  patience = 20,
  plot = TRUE,
  print_folds = TRUE,
  cv_folds = NULL,
  stop_if_nan = TRUE,
  mylapply = lapply,
  save_weights = FALSE,
  ...
)
{
  
  if(is.null(cv_folds)){ 
    cv_folds <- x$init_params$cv_folds
  }else if(!is.list(cv_folds) & is.numeric(cv_folds)){
    cv_folds <- make_cv_list_simple(
      data_size = NROW(x$init_params$data[[1]]), 
      cv_folds)
  }else{
    stop("Wrong format for cv_folds.")
  }
  if(is.null(cv_folds)){
    warning("No folds for CV given, using k = 10.\n")
    cv_folds <- make_cv_list_simple(
      data_size = NROW(x$init_params$data[[1]]), 10)
  }
  nrfolds <- length(cv_folds)
  old_weights <- x$model$get_weights()
  
  if(print_folds) folds_iter <- 1
  
  # subset fun
  if(NCOL(x$init_params$y)==1)
    subset_fun <- function(y,ind) y[ind] else
      subset_fun <- function(y,ind) subset_array(y,ind)
  
  res <- mylapply(cv_folds, function(this_fold){
    
    if(print_folds) cat("Fitting Fold ", folds_iter, " ... ")
    st1 <- Sys.time()
    
    # does not work?
    # this_mod <- clone_model(x$model)
    this_mod <- x$model
    
    train_ind <- this_fold[[1]]
    test_ind <- this_fold[[2]]
    
    # data
    if(is.data.frame(x$init_params$data)){
      train_data <- x$init_params$data[train_ind,, drop=FALSE] 
      test_data <- x$init_params$data[test_ind,,drop=FALSE]
    }else if(class(x$init_params$data)=="list"){
      train_data <- lapply(x$init_params$data, function(x) 
        subset_array(x, train_ind))
      test_data <- lapply(x$init_params$data, function(x) 
        subset_array(x, test_ind))
    }else{
      stop("Invalid input format for CV.")
    }
    
    # make callbacks 
    this_callbacks <- list()
    if(save_weights){
      weighthistory <- WeightHistory$new()
      this_callbacks <- append(this_callbacks, weighthistory)
    }
    
    args <- list(...)
    args <- append(args,
                   list(object = this_mod,
                        x = prepare_newdata(
                          x$init_params$parsed_formulae_contents,
                          train_data,
                          pred = FALSE,
                          index = train_ind,
                          cv = TRUE),
                        y = subset_fun(x$init_params$y,train_ind),
                        validation_split = NULL,
                        validation_data = list(
                          prepare_newdata(
                            x$init_params$parsed_formulae_contents,
                            test_data,
                            pred = FALSE,
                            index = test_ind,
                            cv = TRUE),
                          subset_fun(x$init_params$y,test_ind)
                        ),
                        callbacks = this_callbacks,
                        verbose = verbose,
                        view_metrics = FALSE
                   )
    )
    args <- append(args, x$init_params$ellipsis)
    
    ret <- do.call(fit_fun, args)
    if(save_weights) ret$weighthistory <- weighthistory$weights_last_layer
    
    if(stop_if_nan && any(is.nan(ret$metrics$validloss)))
      stop("Fold ", folds_iter, " with NaN's in ")
    
    if(print_folds) folds_iter <<- folds_iter + 1
    
    this_mod$set_weights(old_weights)
    td <- Sys.time()-st1
    if(print_folds) cat("\nDone in", as.numeric(td), "", attr(td,"units"), "\n")
    
    return(ret)
    
  })
  
  class(res) <- c("drCV","list")
  
  if(plot) try(plot(res))
  
  x$model$set_weights(old_weights)
  
  invisible(return(res))
  
}

#' mean of model fit
#' 
#' @export
#' @rdname methodDR
#'
#' @param x a deepregression model
#' @param data optional data, a data.frame or list
#' @param ... arguments passed to the predict function
#' 
#' @method mean deepregression
#' 
#'
mean.deepregression <- function(
  x,
  data,
  ...
)
{
  predict.deepregression(x, newdata = data, apply_fun = tfd_mean, ...)
}


#' Generic sd function
#' 
#' @param x object
#' @param ... further arguments passed to the class-specific function
#' 
#' @export
sd <- function (x, ...) {
  UseMethod("sd", x)
}

#' Standard deviation of fit distribution
#' 
#' @param x a deepregression object
#' @param data either NULL or a new data set
#' @param ... arguments passed to the \code{predict} function
#' 
#' @export
#' @rdname methodDR
#'
sd.deepregression <- function(
  x,
  data,
  ...
)
{
  predict.deepregression(x, newdata = data, apply_fun = tfd_stddev, ...)
}

#' Generic quantile function
#' 
#' @param x object
#' @param ... further arguments passed to the class-specific function
#' 
#' @export
quantile <- function (x, ...) {
  UseMethod("quantile", x)
}

#' Calculate the distribution quantiles
#' 
#' @param x a deepregression object
#' @param data either \code{NULL} or a new data set
#' @param value the quantile value(s) 
#' @param ... arguments passed to the \code{predict} function
#' 
#' @export
#' @rdname methodDR
#'
quantile.deepregression <- function(
  x,
  data,
  value,
  ...
)
{
  predict.deepregression(x, 
                         newdata = data, 
                         apply_fun = function(x) tfd_quantile(x, value=value),
                         ...)
}

#' Function to return the fitted distribution
#' 
#' @param x the fitted deepregression object
#' @param data an optional data set
#' 
#' @export
#' 
get_distribution <- function(
  x,
  data=NULL
)
{
  if(is.null(data)){
    disthat <- x$model(unname(x$init_params$input_cov))
  }else{
    # preprocess data
    if(is.data.frame(data)) data <- as.list(data)
    newdata_processed <- prepare_data(x, data, pred=TRUE)
    disthat <- x$model(newdata_processed)
  }
  return(disthat)
}


log_score <- function(
  x,
  data=NULL,
  this_y=NULL,
  ind_fun = function(x) tfd_independent(x,1),
  convert_fun = as.matrix,
  summary_fun = function(x) x
)
{
  is_trafo <- x$init_params$family=="transformation_model"
  
  if(is.null(data)){
    this_data <- unname(x$init_params$input_cov)
    this_data <- lapply(this_data, function(x) tf$cast(x, tf$float32))
    if(is_trafo)
      this_data <- list(this_data, tf$constant(matrix(x$init_params$y, 
                                                      ncol=1), 
                                               dtype = "float32"))
    disthat <- x$model(this_data)
  }else{
    # preprocess data
    if(is.data.frame(data)) data <- as.list(data)
    newdata_processed <- prepare_data(x, data, pred=TRUE)
    if(is_trafo){
      if(missing(this_y)) stop("Must provide this_y for transformation models and new data.")
      newdata_processed <- list(unname(
        lapply(c(newdata_processed,
                 list(x$init_params$y_basis_fun(this_y)),
                 list(x$init_params$y_basis_fun_prime(this_y))),
               function(y) tf$cast(y, tf$float32))),
        tf$constant(matrix(this_y, 
                           ncol=1), 
                    dtype = "float32")
      )
    }
    disthat <- x$model(newdata_processed)
  }
  
  if(is_trafo)
    return(summary_fun(
      convert_fun(
        tfd_normal(loc = 0, scale = 1) %>% 
          tfd_log_prob(disthat[,2,drop=F] + 
                         disthat[,1,drop=F])
      ))
    )
  
  if(is.null(this_y)){
    this_y <- x$init_params$y
  }
  return(summary_fun(convert_fun(
    disthat %>% ind_fun() %>% tfd_log_prob(this_y)
  )))
}

get_shift <- function(x)
{
  
  stopifnot("deeptrafo" %in% class(x))
  nrtw <- length(x$model$trainable_weights)
  c(-1* (as.matrix(x$model$weights[[nrtw-1]] + 0)))
  
}

get_theta <- function(x)
{
  
  stopifnot("deeptrafo" %in% class(x))
  nrtw <- length(x$model$trainable_weights)
  reshape_softplus_cumsum(
    as.matrix(x$model$weights[[nrtw]] + 0), 
    order_bsp_p1 = x$init_params$order_bsp + 1
  )
  
}
