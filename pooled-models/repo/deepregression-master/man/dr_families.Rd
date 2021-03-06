% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/families.R
\name{make_tfd_dist}
\alias{make_tfd_dist}
\title{Families for deepregression}
\usage{
make_tfd_dist(
  family,
  add_const = 1e-08,
  return_nrparams = FALSE,
  trafo_list = NULL
)
}
\arguments{
\item{family}{character vector}

\item{add_const}{small positive constant to stabilize calculations}

\item{return_nrparams}{logical, if TRUE, only the number of distribution parameters is
returned; else (FALSE) the \code{dist_fun} required in \code{deepregression}}

\item{trafo_list}{list of transformations for each distribution parameter. 
Per default the transformation listed in details is applied.}
}
\description{
Families for deepregression
}
\details{
To specify a custom distribution, define the a function as follows
\code{
function(x) do.call(your_tfd_dist, lapply(1:ncol(x)[[1]], 
                                    function(i)
                                     your_trafo_list_on_inputs[[i]](
                                       x[,i,drop=FALSE])))
}
and pass it to \code{deepregression} via the \code{dist_fun} argument.
Currently the following distributions are supported 
with parameters (and corresponding inverse link function in brackets):

\itemize{
 \item{"normal": }{normal distribution with location (identity), scale (exp)}
 \item{"bernoulli": }{bernoulli distribution with logits (identity)}
 \item{"bernoulli_prob": }{bernoulli distribution with probabilities (sigmoid)}
 \item{"beta": }{beta with concentration 1 = alpha (exp) and concentration 
 0 = beta (exp)}
 \item{"betar": }{beta with mean (sigmoid) and scale (sigmoid)}
 \item{"cauchy": }{location (identity), scale (exp)}
 \item{"chi2": }{cauchy with df (exp)}
 \item{"chi": }{cauchy with df (exp)}
 \item{"exponential": }{exponential with lambda (exp)}
 \item{"gamma": }{gamma with concentration (exp) and rate (exp)}
 \item{"gammar": }{gamma with location (exp) and scale (exp)}
 \item{"gumbel": }{gumbel with location (identity), scale (exp)}
 \item{"half_cauchy": }{half cauchy with location (identity), scale (exp)}
 \item{"half_normal": }{half normal with scale (exp)}
 \item{"horseshoe": }{horseshoe with scale (exp)}
 \item{"inverse_gamma": }{inverse gamma with concentation (exp) and rate (exp)}
 \item{"inverse_gaussian": }{inverse Gaussian with location (exp) and concentation 
 (exp)}
 \item{"laplace": }{Laplace with location (identity) and scale (exp)}
 \item{"log_normal": }{Log-normal with location (identity) and scale (exp) of 
 underlying normal distribution}
 \item{"logistic": }{logistic with location (identity) and scale (exp)}
 \item{"negbinom": }{neg. binomial with mean (exp) and st.dev.(exp)}
 \item{"pareto": }{Pareto with concentration (exp) (and if modeled scale (exp), 
 else scale = 1)}
 \item{"poisson": }{poisson with rate (exp)}
 \item{"poisson_lograte": }{poisson with lograte (identity))}
 \item{"student_t": }{Student's t with df (exp)}
 \item{"student_t_ls": }{Student's t with df (exp), location (identity) and 
 scale (exp)}
 \item{"uniform": }{uniform with upper and lower (both identity)}
 \item{"zip":  }{Zero-inflated poisson distribution with }
}
}
