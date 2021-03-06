orthog_structured <- function(S,L)
{
  qrL <- qr(L)
  Q <- qr.Q(qrL)
  XtXinvXt <- tcrossprod(Q)
  Sorth <- S - XtXinvXt%*%S
  return(Sorth)
}

orthog_smooth <- function(pcf){
  
  nml <- attr(pcf$linterms, "names")
  nms <- attr(pcf$smoothterms, "names")
  if(!is.null(nms) && grepl(",by_",nms)){
    warning("Orthogonalization for s-terms with by-Term currently not supported.")
  }
  L <- NULL
  for(nm in nms){
    
    if("(Intercept)" %in% nml)
      L <- matrix(rep(1,NROW(pcf$linterms)), ncol=1)
    
    if(nm %in% nml){
      
      if(!is.null(L))
        L <- cbind(L, pcf$linterms[,nm]) else
          L <- pcf$linterms[,nm]
      
    }
    
    if(!is.null(L))
      pcf$smoothterms[[nm]][[1]]$X <- 
        orthog_structured(pcf$smoothterms[[nm]][[1]]$X, L) 
    
  }
  
  return(pcf)
}



make_orthog <- function(
  pcf,
  retcol = FALSE
)
{

  if(is.null(pcf$deepterms)) return(NULL)
  n_obs <- nROW(pcf)
  if(n_obs==0){
    if(!is.null(pcf$smoothterms))
      n_obs <- NROW(pcf$smoothterms[[1]][[1]]$X) else
        n_obs <- nROW(pcf$deepterms[[1]])
  }
  nms <- lapply(pcf[c("linterms","smoothterms")], function(x)attr(x,"names"))
  nmsd <- lapply(pcf$deepterms, function(x) attr(x,"names"))
  if(!is.null(nms$smoothterms))
    struct_nms <- c(nms$linterms, #unlist(strsplit(nms$smoothterms,",")),
                    nms$smoothterms) else
      struct_nms <- nms$linterms
  if(is.null(pcf$linterms) & is.null(pcf$smoothterms))
    return(NULL)
  
  qList <- lapply(nmsd, function(nn){
      
    # number of columns removed due to collinearity
    rem_cols <- 0
    # if there is any smooth 
    X <- matrix(rep(1,n_obs), ncol=1) 
    # Ps <- list()
    # lambdas <- c()
    if(length(intersect(nn, struct_nms)) > 0){
      
      for(nm in nn){
        
        if(nm %in% nms$linterms) X <- cbind(X,pcf$linterms[,nm,drop=FALSE])
        if(nm %in% nms$smoothterms){ 
          
          Z_nr <- drop_constant(pcf$smoothterms[[
            grep(paste0("\\b",nm,"\\b"),nms$smoothterms)
            ]][[1]]$X)
          
          X <- cbind(X,Z_nr[[1]])
          rem_cols <- rem_cols + Z_nr[[2]]
          
        }  
      }
      
      # check for TP
      if(any(length(nn)>1 &  grepl(",", nms$smoothterms))){ 
        
        tps_index <- grep(",", nms$smoothterms)
        for(tpi in tps_index){
          
          if(length(setdiff(unlist(strsplit(nms$smoothterms[tpi],",")), nn))==0){
            
            X <- cbind(X, pcf$smoothterms[[tpi]][[1]]$X)
            
          }
        }
      } 
      
    }else{
      return(NULL)
    }
    
    qrX <- qr(X)
    # if(qrX$rank<ncol(X)){
    #   warning("Collinear features in X")
    #   # qrX <- qr(qrX$qr[,1:qrX$rank])
    # }
    
    Q <- qr.Q(qrX)
    # coefmat <- tcrossprod(Q)
    if(retcol) return(NCOL(Q)) else 
      return(Q)
    
  })
  
  return(qList)
  
}

# for P-Splines
# Section 2.3. of Fahrmeir et al. (2004, Stat Sinica)
centerxk <- function(X,K) tcrossprod(X, K) %*% solve(tcrossprod(K))

orthog <- function(Y, Q)
{
  
  # print(Y)
  # print(XtXinvXt)
  XtXinvXt <- tf$linalg$matmul(Q,tf$linalg$matrix_transpose(Q))
  Yorth <- Y - tf$linalg$matmul(XtXinvXt, Y)
  # print(Yorth)
  return(Yorth)
  
  # tfcrossprodx <- function(x) tf$linalg$matmul(tf$linalg$matrix_transpose(x),x)
  # 
  # if(is.null(XtXinv))
  # 
  # if(!is.null(pwr)){ 
  #   # X must be same size but something invertible in the case of
  #   # prediction, for prediction pwr = 0
  #   # pwr <- tf$reshape(
  #   pwr <- tf$squeeze(tfcrossprodx(pwr), 0)
  #   #                   list(tf$constant(1)))
  #   neg_pwr <- 1-pwr
  #   pwr_ncolX <- tf$linalg$tensor_diag(tf$tile(pwr, ncol(X)))
  #   neg_pwr_ncolX <- tf$linalg$tensor_diag(tf$tile(neg_pwr, ncol(X)))
  #   pwr_ncolY <- tf$linalg$tensor_diag(
  #     tf$tile(pwr, ncol(Y))
  #   )
  #   XtX <- tf$linalg$matmul(tfcrossprodx(X), pwr_ncolX) + neg_pwr_ncolX
  #   XtXinv <- tf$linalg$inv(XtX + tf$linalg$tensor_diag(rep(1e-8,ncol(X))))
  #   XXtXinv <- tf$linalg$matmul(X,XtXinv)
  #   XtY <- tf$linalg$matmul(tf$linalg$matrix_transpose(X), Y)
  #   Y - tf$linalg$matmul(tf$linalg$matmul(XXtXinv, XtY), pwr_ncolY)
  # }else{
  #   Y - tf$linalg$matmul(tf$linalg$matmul(X, tf$linalg$inv(
  #     tf$linalg$matmul(tf$linalg$matrix_transpose(X),X)
  #   )), tf$linalg$matmul(tf$linalg$matrix_transpose(X), Y))
  # }
}

orthog_nt <- function(Y,X) Y <- X%*%solve(crossprod(X))%*%crossprod(X,Y)

split_model <- function(model, where)
{

  fun_as_string <- Reduce(paste, deparse(body(model)))
  split_fun <- strsplit(fun_as_string, "%>%")[[1]]
  length_model <- length(split_fun) - 1

  if(where < 0) where <- length_model + where
  # as input is also part of split_fun
  where <- where + 1

  # define functions as strings
  first_part <- paste(split_fun[1:where], collapse = "%>%")
  second_part <- paste(split_fun[c(1,(where+1):(length_model+1))], collapse = "%>%")

  # define functions with strings
  first_part <- eval(parse(text = paste0('function(x) ', first_part)))
  second_part <- eval(parse(text = paste0('function(x) ', second_part)))

  return(list(first_part, second_part))

}

### R6 class, not used atm

if(FALSE){

  Orthogonalizer <- R6::R6Class("Orthogonalizer",

                                lock_objects = FALSE,
                                inherit = KerasLayer,

                                public = list(

                                  output_dim = NULL,

                                  kernel = NULL,

                                  initialize = function(inputs) {

                                    self$inputs <- inputs

                                  },

                                  call = function(inputs, training=NULL) {
                                    if(is.null(training))
                                      return(inputs[[1]]) else
                                        return(orthog(inputs[[1]],inputs[[2]]))
                                  }
                                )
  )

  layer_orthog <- function(inputs, ...) {
    create_layer(layer_class = Orthogonalizer,
                 args = list(inputs = inputs)
    )
  }


}

combine_model_parts <- function(deep, deep_top, struct, ox, orthog_fun)
{
  
  if(is.null(deep)){
    
    return(struct)
  
  }else if(is.null(struct)){
    
    if(length(deep)==1) return(deep_top[[1]](deep[[1]]))
    
    return(layer_add(lapply(1:length(deep), 
                            function(j) deep_top[[j]](deep[[j]]))))
    
  }else{
    
    if(is.null(ox)){
      
      return(
        layer_add( append(lapply(1:length(deep), 
                                           function(j) deep_top[[j]](deep[[j]])),
                          list(struct))
        )
        )
      
    }else{
      
      if(length(deep) > 1) 
        warning("Applying orthogonalization for more than ", 
                "one deep model in each predictor.")
      
      return(
        layer_add( append(lapply(1:length(deep), 
                                 function(j){
                                   
                                   if(is.null(ox[[j]]))
                                     return(deep_top[[j]](deep[[j]])) else
                                       deep_top[[j]](orthog_fun(deep[[j]], 
                                                                ox[[j]]))
                                   }), 
                          struct))
      )
    }
  }
}

drop_constant <- function(X){ 
  
  this_notok <- apply(X, 2, function(x) var(x, na.rm=TRUE)==0)
  return(list(X[,!this_notok], 
              sum(this_notok))
  )

}