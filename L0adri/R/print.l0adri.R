

print.l0adri <- function(y, X, x, weights, family, lam, lower, upper, algo, maxit, tol, boot.repl = 100, ctrl.fit, ctrl.iwls, ctrl.l0) {
  
  require( boot)
  output <- list()
  #plot(x,fit$fitted.values,title(main="Response function"),ylab="Values fitted by l0adri")
 # se <- sqrt(diag(vcov(fit)))
  
 # beta_nonzero <- fit$beta[fit$beta!=0]
 # df <- df()
  
  #print(df)
 # return(list(beta=fit$beta,fitted.values=as.vector(fit$mu),weights=fit$W,eta=fit$eta,))
  # output$Listbetas <- beta_nonzero
  
  #make a function that will be used for the boot function of bootstrapping
  #what is indices????
  #start = a vector of starting values for the coefficients to estimate
  
  bootstrap_statistic <- function(X, y, weights, family, lam, lower, upper, algo, miniter, maxit, tol, thresh){
    
    out_l0adri <- l0adri(X=X, y=y, weights = weights, family = family, lam=lam, lower=lower, upper=upper, algo=algo, miniter=miniter, maxit=maxit, tol=tol, thresh=thresh)
   
    out_l0adri_beta <- out_l0adri$beta
    return(out_l0adri_beta)
  }
  
  fitobj <- l0adri(X=X, y=y, weights = weights, family = family, lam=lam, lower=lower, upper=upper, algo=algo, maxit=maxit, tol=tol)
  beta_nonzero_fitboj <- fitobj$beta[fitobj$beta!=0]
  output$Listbetas <- beta_nonzero_fitboj
  
  #bootstrap:
  #resample the rows of covariate matrix and accordingly with y
  #command sample (with option resample)
  #post.filter.fn = fit$post.filter.fn is this needed as input for the boot?
  output$bootstrap <- boot(data = y, X = X, wts = fitobj$weights, family = fitobj$family,
                           lambda = fitobj$lambda, start = NULL,
                           control.l0 = ctrl.l0, control.iwls = ctrl.iwls,
                           control.fit = ctrl.fit, 
                           # boot arguments
                           statistic = bootstrap_statistic, R = boot.repl, stype = "i")
  
  output$estimates_boot <- boot.result$t0
  
  output$CI.lower <- apply(out$boot.result$t, 2, quantile, probs = alpha/2)
  output$CI.upper <- apply(out$boot.result$t, 2, quantile, probs = 1-alpha/2)
  
  return(output)
  #p-values: whether the coeff. are 0 or not: 1-proportion of coeff. non-zero. => p-value x 2 
  
}
