#' Cross Validation to tune the regularization parameter lambda for L0 adaptive ridge models
#' 
#' This function uses k-fold cross validation for l0adri and returns the value
#' of the tuning parameter lambda that yields the lowest mean square prediction
#' error.
#' 
#' @param X  the covariate matrix
#' @param y  vector of values of the dependent variable
#' @param weights observation weights
#' @param family  desired noise model and link function; all default R families
#'   are supported, e.g. gaussian(identity), poisson(identity), Gamma(identity),
#'   inv.gaussian(identity) or binomial(logit)
#' @param lambdas vector with values of the regularization parameter lambda for
#'   which to perform cross-validation
#' @param nfolds the number of folds to use during cross validation
#' @param nonnegative a vector of TRUE/FALSE representing whether
#' to apply nonnegative constraints for each of the coefficients
#' @param algo the algorithm used to apply the constraints. Can be 
#' osqp (quadratic programming), clipping or scaling
#' @param maxit the maximum number of times to iterate the adaptive ridge fits
#' @param tol  convergence tolerance
#' @param thresh coefficients with absolute value below this threshold will be
#'   set to zero
#' @param seed random seed
#' @param Plot whether or not to plot the cross validation error in function of the lambda values
#' 
#' @return list with elements lam.min (the optimal value of lambda that gives
#'   the minimal weighted mean square error), cv.error (weighted mean square error for different lambda values) and error
#'   (weighted total sums of squares for different folds and lambda values)
#' 
#' @examples 
#' lam1=cv.l0adri(X=X,y=y,lam=l,family=gaussian(identity),nonnegative = rep(TRUE,ncol(X))
#' 
#' 


cv.l0adri <- function(X, y, weights = rep( 1, nrow( X)), 
                      family = c("gaussian(identity)", "poisson(identity)", "Gamma(identity)", "inv.gaussian(identity)", "binomial(logit)"), 
                      lambdas,
                      nfolds = 10,  maxit = 100, seed = 1, lower=rep(-.Machine$double.xmax,ncol(X)),upper=rep(.Machine$double.xmax,ncol(X)),
                      algo = c('osqp','linearsolver'), tol=1E-8, thresh=1E-3, Plot=TRUE) { 
  if (!is.null(lambdas) && length(lambdas) < 2) {
    stop("Need a sequence of lambda values")
  }
  if (nfolds < 3) {
    stop("nfolds must be equal or larger than 3")
  }
  if (!missing(seed)) {
    set.seed(seed)
  }
  if (length(lower) == 1){
    lower = rep(lower, ncol(X))
  }else if(length(lower) != ncol(X)){
    stop("Lower constraint length conflicts")
  }
  if (length(upper) == 1){
    upper = rep(upper, ncol(X))
  }else if(length(upper) != ncol(X)){
    stop("Upper constraint length conflicts")
  }
  if (!is(family, "family")) family = family()
  variance = family$variance
  linkinv = family$linkinv
  mu.eta = family$mu.eta
  
  n <- nrow(X)
  p <- ncol(X)
  error <- matrix(NA, length(lambdas), nfolds)     # preallocate error matrix with NA
  id <- sample(rep(1:nfolds, length = n))          # randomly sample cross validation folds

  compute.ss <- function(mu, y, w, fam){
    eta <- fam$linkfun(mu)
    w <- w * as.vector(fam$mu.eta(eta)^2 / fam$variance(mu)) # IWLS weights from last iteration
    z.res <- (y - mu)/fam$mu.eta(eta) # the residuals on the adjusted scale = z.res = z - z.fit = eta + (y - mu)/g'(eta) - eta = (y - mu)/g'(eta)
    #print(length(z.res))
    ss <- sum(w * z.res^2) # sums of squares
    return(ss)
  }
  
  for (i in 1:nfolds){
    test <- which(id == i)
    training <- which(id != i)
    y_train <- y[training]
    X_train <- X[training,,drop=FALSE]
    for (j in 1:length(lambdas)){
      l0fit <- l0adri(X=X_train, y=y_train, family=family, 
                     lam=lambdas[j], algo=algo, lower=lower, upper=upper, maxit=maxit, 
                     tol=tol, thresh=thresh)
      pred_test <- X[test,,drop=FALSE] %*% l0fit$beta
      
      mu_test = linkinv(pred_test)
      varg_test = variance(mu_test)
      gprime_test = mu.eta(pred_test)
      weights_test = weights[test] * as.vector(gprime_test^2 / varg_test) 
    
      error[j,i] <- compute.ss(mu=pred_test, y=y[test], w=weights_test, fam=family)
    }
  }

  cv.error <- rowMeans(error)
  if (Plot) plot(log10(lambdas), cv.error, type="l", xlab="Log10(lambda)", ylab="Cross-validation error (WMSE)", col="red")
  lam.min <- lambdas[which.min(cv.error)]

  return(list(lam.min=lam.min,cv.error=cv.error,error=error))
}
