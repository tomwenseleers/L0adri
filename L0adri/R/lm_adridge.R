#' Function to fit a single adaptive ridge regression with or without nonnegativity or box constraints
#' 
#' This function fits a least squares adaptive ridge regression model using squared L2 norm regularization on each coefficient of lam / beta^2, where value lam and the initial coefficient estimates beta can both be specified.
#' An efficient linear solver is used that adapts to the different arguments given, e.g. whether or not covariate matrix X is sparse, whether or not nonnegativity or box constraints are required,
#' and the size of the covariate matrix. Three different methods to deal with nonnegativity or box constraints are provided (osqp=quadratic programming, clipping or scaling).
#' When covariate matrix X is sparse efficient sparse solvers from the Eigen package are used.
#' 
#' @param X  the covariate matrix
#' @param y  vector of values of the dependent variable
#' @param weights  observation weights of the IRLS GLM fit in each iteration
#' @param beta coefficient values from which the adaptive ridge penalty weights=1/beta^2 are calculated; each coefficients will receive an L2-penalty of lam * the adaptive penalty weight.
#' @param lam the value of tuning parameter lambda for the ridge
#'  regression
#' @param nonnegative a vector of TRUE/FALSE representing whether
#' to apply nonnegative constraints on each of the coefficients
#' @param algo the algorithm used to apply the constraints. Can be 
#' osqp (quadratic programming), clipping or scaling
#' @param thresh coefficients with absolute value below this threshold will be
#'   set to zero
#' 
#' @return vector with coefficients 

lm_adridge = function( X, y, weights, lambda = 1, 
                       beta = rep(1, ncol( X)), 
                       thresh = 1E-4,
                       lower=rep(-.Machine$double.xmax,ncol(X)),
                       upper=rep(.Machine$double.xmax,ncol(X)), 
                       algo=c('osqp','linearsolver')){ 
  # TODO: remove scaling, change nonnegative to lower & upper to allow box constaints
  
  require( Matrix)
  require( osqp)
  require( Rcpp)
  require( RcppEigen)
  require( pracma)
  
  # this part only functions for OSQP now
  intercept = (sum(X[1,]==1)==nrow(X))
  
  #beta[ beta< lower+thresh]=lower
  #nonzero =(beta!=lower)
  beta[ abs(beta) < thresh] = 0
  nonzero = (beta != 0)
  
  # the beta0,lower0 and upper0 will pass the linear slover function
  beta0=beta
  lower0=lower
  upper0=upper
  
  if (sum(nonzero)==0) {
    beta = rep(0, ncol(X))
    return (beta)
  }
  
  # this part only functions for OSQP now
  Xw = X[, nonzero, drop=F]*sqrt( weights) 
  yw = y*sqrt( weights)
  beta = beta[ nonzero]
  
  lower = lower[nonzero]
  upper = upper[nonzero]
  
  # Get dimensionality
  nvars <- ncol(Xw)
  nobs <-  nrow(Xw)
  
  # this part only functions for OSQP now
  penweights = 1/beta^2 # adaptive penalty weights, cf Frommlet paper
  lam = lambda*penweights # beta specific lambdas  changed lam to lambda also in the function
  if (intercept & nonzero[ 1]) lam[ 1] = 0 #no penalty on intercept

  #using OSQP
  if (algo == "osqp"){
    
    # TO DO: put this into in OSQP -> make it constrainedRidgeOSQP
    ywaug <- c( yw, rep( 0, nvars)) 
    Xwaug <- rbind( Xw, as(sqrt( lam)*.sparseDiagonal( nvars, x=1), "dgCMatrix") )
    
    osqp_fit <- constrainedLS_osqp( Xwaug, ywaug, lower,upper)
    beta <- osqp_fit$x
    
  }  else if(algo=="linearsolver"){
    

    lambda <- lambda
    linear_fit <- linearSolver(X,y,weights,lambda,beta0,lower0,upper0)


    beta <- linear_fit
    
  }
  
  beta_out = rep(0, length = ncol( X))
  beta_out[ nonzero] = beta 
  
  return ( beta_out)
  
}


