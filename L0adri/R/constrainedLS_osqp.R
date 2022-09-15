#' OSQP quadratic programming solver to solve a constrained least squares problem
#' 
#' This function utilizes the OSQP package to carry out a constrained least squares regression.
#' 
#' @param X the covariate (design) matrix
#' @param y vector of values of the dependent variable
#' @param nonnegative a vector of TRUE/FALSE representing whether
#' to apply nonnegative constraints on each of the model's coefficients
#' @param x.start the starting value for x
#' @param y.start the starting value for the expected value of y
#' 
#' @return object retuned by OSQP Solve()

constrainedLS_osqp <- function(X, y, 
                               lower = rep(-.Machine$double.xmax,ncol(X)), upper=rep(.Machine$double.xmax,ncol(X)),
                               x.start = NULL, y.start = NULL) {
  
  XtX = Matrix::crossprod(X, X)
  Xty = Matrix::crossprod(X, y)
  
  settings <- osqpSettings(verbose = FALSE, eps_abs = 1e-8, eps_rel = 1e-8, linsys_solver = 0L,
                           warm_start = FALSE) 
  # ospq fits each coefficient from the range (lower, upper), when the nonnegative is true, it should fit from (0,upper),when the 
  # nonnegative is false, it should fit from lower to upper
  

  model <- osqp(XtX, -Xty, t(as.matrix(.sparseDiagonal(ncol(X)))), l=lower,u=upper, pars=settings) 
  
  #add clipping
  if (!is.null(x.start)) model$WarmStart(x=x.start, y=y.start)
  model$Solve()
}
