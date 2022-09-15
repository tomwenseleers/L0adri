# Method for Linear Solver

linearSolver <- function(X, y, weights, lambda = 1, 
                         beta = rep(1,ncol(X)),
                         lower = rep(-.Machine$double.xmax,ncol(X)), 
                         upper=rep(.Machine$double.xmax,ncol(X)),
                         x.start = NULL, y.start = NULL) {
  require(Matrix)
   # Check the zero's
  nonzero = (beta != 0)
  
  # Add weights
  Xw = X[, nonzero, drop=F]*sqrt( weights) 
  yw = y*sqrt( weights)
  
  # Adapt to nonzero (THIS CAN GO)
  beta = beta[nonzero]
  
  # Intercept
  intercept = (sum(X[1,]==1)==nrow(X))
  
  # Update Lambda
  penweights = 1/beta^2 
  lam = lambda*penweights 
  if (intercept & nonzero[ 1]) lam[ 1] = 0 #no penalty on intercept 
  
  # Get dimensions
  nvars <- ncol(Xw)
  nobs <-  nrow(Xw)
  
  # Actual Solver
  
  #change it back to class with the real definition of sparse matrix
  if( class(X)=="dsCMatrix"||class(X)=="dgCMatrix"){   #if matrix is sparse 
    
    if (nobs > nvars) {#high dim
      # PS also for the sparse case there is a more efficient way for when nvars>nobs, cf Liu & Li 2016 
      # I added this below
      library(L0adri1.0)
      ywaug <- c( yw, rep( 0, nvars)) 
      Xwaug <- rbind( Xw, as(sqrt( lam)*.sparseDiagonal( nvars, x=1),"dgCMatrix") )
      beta <- solve_sparse_lsconjgrad( list( Xwaug, ywaug, rep( 0, nvars)))$res
    } else {
      
      # optimisation for nvars >= nobs case, cf. Liu & Li 2016
      Xwt <- (beta^2) * Xw # cf optimisation for nvars >= nobs case, cf. Liu & Li 2016 from Woodbury identity, cf https://darrenho.github.io/SML2015/woodburyidentity.pdf
      beta <- as.vector(Matrix::t( Xwt) %*% solve(Matrix::tcrossprod( Xwt) + lambda*.sparseDiagonal(nobs), yw, tol=2*.Machine$double.eps)) # default solve worked best here
    }
  }else{
    #if matrix is dense
    beta <- solve( crossprod( Xw) + diag(lam), crossprod( Xw, yw), tol = .Machine$double.eps/4) 
    beta <- as.vector(beta)
  
  
  if (( lambda == 0)&( nobs > nvars)) { beta = solve( crossprod( Xw), crossprod( Xw, yw), tol = .Machine$double.eps)}
  
  if (( lambda == 0)&( nobs <= nvars)) { beta = as.vector(pinv(as.matrix(crossprod(Xw))) %*% crossprod( Xw, yw)) }
  }
  
  lower = lower[nonzero]
  upper = upper[nonzero]
  beta[beta < lower ] = lower[beta < lower]
  beta[beta > upper ] = upper[beta > upper]
  
  return(beta)
  
}










