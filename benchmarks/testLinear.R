sim <- simulate_spike_train()
X <- sim$X
y <- sim$y
sim$a
ncol(X)
beta = sim$a
lambda = 1
lower = rep(0,ncol(X))
upper=rep(100000,ncol(X))
weights = rep(1,nrow(X))

Xw = X[, nonzero, drop=F]*sqrt( weights) 
yw = y*sqrt( weights)

dim(Xw)
dim(diag(lambda))
diag(lambda)
linear <- linearSolver(X = X, y = y, lambda = 1, beta=sim$a,
                                   lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), weights =  rep( 1, nrow( X)))
linear

lm <- lm_adridge(X = X, y = y,algo = "linearsolver", lambda = 1, beta = sim$a,
                 lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), 
                 thresh = 1E-4, weights = rep( 1, nrow( X)))

# CODE
nonzero = (beta != 0)
Xw = X[, nonzero, drop=F]*sqrt( weights) 
yw = y*sqrt( weights)

#
dim(Xw)
nvars <- ncol(Xw)
nobs <-  nrow(Xw)
#

beta = beta[ nonzero]
penweights = 1/beta^2 
lam = lambda*penweights 
dim(Xw)
dim(diag(lam))
lam


( nobs > nvars) & lambda > 0
beta <- solve( crossprod( Xw) + diag(lam),
                                crossprod( Xw, yw), tol = .Machine$double.eps/4) #  weg met trycatch

beta <- as.vector(beta)
length(beta)


lower <- lower[nonzero]
upper <- upper[nonzero]
length(lower)

beta[beta < lower ] = lower[beta < lower]
beta
beta[beta > upper ] = upper[beta > upper]
beta

#test linear solver for sparse matrix

