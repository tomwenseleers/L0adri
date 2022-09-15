#' Bootstrap for l0adri model
#'
#' @param X  the covariate (design) matrix (can be dense or sparse, i.e. of class dgCMatrix)
#' @param y  vector of values of the dependent variable
#' @return
#' 
bootstrap = function(X,y, family, lam, lower, upper, algo, miniter, maxit, tol, thresh,nr.boot=100){
  
  # create matrix with all coefficients as rows, and their different bootstrap rounds in the columns
  boot.matrix <- matrix(0,nrow = nr.boot,ncol = nrow(X))
  
  #resample the rows of covariate matrix and accordingly with y
  # fit every iteration and a to the boot.matrix
  for(i in 1:nr.boot){
    
    train=sample(x = nrow(X),size = nrow(X),replace=TRUE)
    X.train = X[train,]
    y.train = y[train,]
    
    fit <- l0adri(X.train, y.train, family, algo, lam, lower,upper, maxit, tol)
    boot.matrix[i,] = fit$beta
    
  }
  
  # retrieve confidence intervals and p-values from the boot.matrix
  CI <- matrix(0,nrow = ncol(boot.matrix),3)
  colnames(CI) = c("2,5%","97.5%","p-value")
  for(i in 1:ncol(boot.matrix)) {
    ci <- quantile(boot.matrix[,i], c(.025, .975))
    CI[i,1:2] <- ci
    
    #p-values: whether the coeff. are 0 or not: 1-proportion of coeff. non-zero. => p-value x 2 
    pvalue <- 1-(sum(length(which(boot.matrix[,i] != 0)))/nrow(boot.matrix))
    CI[i,3] <- pvalue
  }
  return(list(CI))
}
 


