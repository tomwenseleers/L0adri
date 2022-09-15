# TESTS
# a few tests to check if L0adri functions work as expected

source("./R/utils.R")
library(devtools)
require(glmnet)
# devtools::install_github("tomwenseleers/L0glm")
library(L0glm)
# source("L0adri1.0/R/l0adri.R")
# source("L0adri1.0/R/lm_adridge.R")
# source("L0adri1.0/R/constrainedLS_osqp.R")

par(mfrow=c(1,2))
sim <- simulate_spike_train(n=500, npeaks=50, 
                            peakhrange = c(10,1E3),
                            seed = 123, Plot = TRUE)
X <- sim$X
y <- sim$y
sum(X==0)

ctrl.fit <- control.fit.gen() # default
ctrl.iwls <- control.iwls.gen(maxit = 1)
ctrl.l0 <- control.l0.gen() # default

#make a sparse matrix to test solver (especially the linear solver)
X_sparse <- Matrix(X, sparse=TRUE)
class(X_sparse)

# L0glm
L0glm.out <- L0glm.fit(X = X, y = y, family = poisson(identity), 
                       lambda = 1, nonnegative = TRUE,
                       control.iwls = ctrl.iwls, control.l0 = ctrl.l0,
                       control.fit = ctrl.fit)
plot(sim$a, L0glm.out$coefficients, pch=16)
plot_L0glm_benchmark(x = sim$x, y = y, fit = L0glm.out, a.true = sim$a, # TO DO: include function like this?
                     main="Ground truth vs L0 penalized L0glm estimates")

# L0adri
par(mfrow=c(2,1))
# poisson - osqp
L0adri.nnpois.out <- l0adri(X = sim$X, y = sim$y, family = poisson(identity),
                            algo = "osqp", lam = 1,
                            lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), maxit = 1000, tol = 1E-8)
plot(sim$a, L0adri.nnpois.out$beta, pch=16)

# gaussian - oqsp
L0adri.nnwLS.out <- l0adri(X = sim$X, y = sim$y, weights = 1/(sim$y+0.1),
                           family = gaussian(identity), 
                           algo = "osqp", lam = 1, 
                           lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), maxit = 1000, tol = 1E-8)
plot(sim$a, L0adri.nnwLS.out$beta, pch=16)



# print output of the gaussian oqsp with the print.l0adri.R script
# printing <- print.l0adri(y, X, x, weights = 1/(sim$y+0.1), family = gaussian(identity), 
#              algo = "osqp", lam = 1, 
#              lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), maxit = 1000, tol = 1E-8, boot.repl = 100, ctrl.fit, ctrl.iwls, ctrl.l0) 


# poisson - clipping
L0adri.ccpois.out <- l0adri(X = sim$X, y = sim$y,
                            family = poisson(identity), 
                            algo = "linearsolver", lam = 1, 
                            lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), maxit = 1000, tol = 1E-8)
plot(sim$a, L0adri.ccpois.out$beta, pch=16)


# gaussion - linear solver

# add benchmark for clipping
L0adri.gauss.linear <- l0adri(X = sim$X, y = sim$y, 
                 family = gaussian(identity), 
                 algo = "linearsolver", lam = 1, 
                 lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), maxit = 1000, tol = 1E-8)

plot(sim$a, L0adri.gauss.linear$beta, pch=16)

L0adri.gauss.linear$beta == L0adri.nnwLS.out$beta

# add benchmark for clipping, linear solver with sparse matrix
L0adri.gauss.linear2 <- l0adri(X_sparse, y, 
                              family = gaussian(identity), 
                              algo = "linearsolver", lam = 1, 
                              lower=rep(0,ncol(X_sparse)),
                              upper=rep(1000,ncol(X_sparse)), 
                              maxit = 1000, tol = 1E-8)
sum(L0adri.gauss.linear2$beta !=0)
plot(sim$a, L0adri.gauss.linear2$beta, pch=16)


L0adri.gauss.linear3 <- l0adri(X_sparse, y, 
                               family = gaussian(identity), 
                               algo = "linearsolver", lam = 1, 
                               lower=rep(0,ncol(X_sparse)),
                               upper=rep(1000,ncol(X_sparse)), 
                               maxit = 1000, tol = 1E-8)

##### PROBLEM = clipping and osqp give same result - maybe expected given regularisation that is used? ####

par(mfrow=c(1,1))
plot(L0adri.nnpois.out$beta, L0adri.ccpois.out$beta, pch=16)
L0adri.nnpois.out$beta
L0adri.ccpois.out$beta

## CHECK IF PROBLEM IS IN lm_adridge ##

# lm_adridge - osqp
lmadr_osqp <- lm_adridge(X = sim$X, y = sim$y,algo = "osqp", lam = 1, beta = sim$a,
                    lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), 
                    thresh = 1E-4, weights = rep( 1, nrow( X)))

plot(lmadr_osqp,sim$a)
length(lmadr_osqp)
length(lmadr_osqp[lmadr_osqp !=0])

par(mfrow=c(3,1))
plot(lmadr_osqp,type="h")
plot(lmadr_clipping, type="h")
plot(y,type="h")

# lm_adridge - clipping
lmadr_clipping <- lm_adridge(X = sim$X, y = sim$y,algo = "linearsolver", lambda = 1, beta = rep(1,ncol(X)),
                    lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), 
                    thresh = 1E-4, weights = rep( 1, nrow( X)))
lmadr_clipping
length(lmadr_clipping)
length(lmadr_clipping[lmadr_clipping!=0])
plot(lmadr_clipping,sim$a)

# test for linear solver
linear <- linearSolver(X = sim$X, y = sim$y, lambda = 1,
                       lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), weights =  rep( 1, nrow( X)))
linear


# test sparse matrix for linear solver
linear2 <- linearSolver(X = X_sparse, y = y, lambda = 1,
                       lower=rep(0,ncol(X_sparse)),upper=rep(1000,ncol(X_sparse)), weights =  rep( 1, nrow( X_sparse)))
linear2
lmadr_clipping2 <- lm_adridge(X = X_sparse, y = y,algo = "linearsolver", lambda = 1, beta = rep(1,ncol(X_sparse)),
                             lower=rep(0,ncol(X_sparse)),upper=rep(1000,ncol(X_sparse)), 
                             thresh = 1E-4, weights = rep( 1, nrow( X_sparse)))
lmadr_clipping2
plot(lmadr_clipping2, sim$a) # error


# Compare clipping w/ osqp
plot(lmadr_osqp,lmadr_clipping, xlab = "OSQP coefficients", ylab = "CLIPPING coefficients", title(main = "Single adaptive ridge regression"))

lmadr_clipping
lmadr_osqp

plot(x=lmadr_clipping,y=sim$a, col="2")
plot(x=lmadr_osqp,y=sim$a, col="2")

# Compare benchmark plots
par(mfrow=c(1,1))
plot_L0adri_benchmark(x = sim$x, y = y, fit = L0adri.nnpois.out, a.true = sim$a, 
                     main="Ground truth vs L0 penalized L0adri estimates")
plot_L0glm_benchmark(x = sim$x, y = y, fit = L0glm.out, a.true = sim$a, # TO DO: include function like this?
                     main="Ground truth vs L0 penalized L0glm estimates")
plot_L0adri_benchmark(x = sim$x, y = y, fit = L0adri.nnwLS.out, a.true = sim$a, 
                      main="Ground truth vs L0 penalized L0adri estimates")

# clipping benchmark
plot_L0adri_benchmark(x = sim$x, y = y, fit =L0adri.gauss.linear , a.true = sim$a, 
                      main="Ground truth vs L0 penalized L0adri estimates")


# compare the number of true peaks and
# weighed R^2 (look up formula): 1 - true value
# number of false positives rate

