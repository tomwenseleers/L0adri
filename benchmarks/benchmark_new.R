#' @param X  the covariate (design) matrix (can be dense or sparse, i.e. of class dgCMatrix)

library(glmnet)
library(ggplot2)
library(devtools)
library(L0glm)
library(microbenchmark)
library("ncvreg")
library("ordinis")
library("L0Learn")
source("L0adri1.0/R/utils.R")
source("L0adri1.0/R/l0adri.R")
source("L0adri1.0/R/lm_adridge.R")
source("L0adri1.0/R/constrainedLS_osqp.R")
source("L0adri1.0/R/linearSolver.R")

sim <- simulate_spike_train()
X <- sim$X
y <- sim$y

ctrl.fit <- control.fit.gen() # default
ctrl.iwls <- control.iwls.gen(maxit = 1)
ctrl.l0 <- control.l0.gen() # default

# Ridge regression using L0adri, L0glm and glmnet

set.seed(123)
n <- 100
p <- 20
x <- matrix(rnorm(n*p), nrow = n, ncol = p)
beta <- runif(p)
y0 <- x %*% beta
y <- y0 + rnorm(n, mean = 0, sd = 2.5)
h<-sum(beta==beta)/p
k=sum(beta!=0)


#add to the script lower as 0 (not as a vector)
cv.l0adri(X=x, y, weights = 1/(y+0.1), family = gaussian(identity), lower =rep(0, ncol(x)), algo="osqp", lambdas= 10^seq(-1,2,length.out = 100), maxit = 100)
cv.l0adri(X=x, y, weights = 1/(y+0.1), family = poisson(identity), lower =rep(0, ncol(x)), algo="osqp", lambdas= 10^seq(-1,2,length.out = 100), maxit = 100)
cv.l0adri(X=x, y,weights = 1/(y+0.1), family = gaussian(identity),lower=rep(0, ncol(x)), algo="linearsolver", lambdas= 10^seq(-1, 1, length.out=100), maxit = 100)

L0adri.nnwLS.out <- l0adri(X = sim$X, y = sim$y, weights = 1/(sim$y+0.1),
                           family = gaussian(identity),
                           algo = "osqp", lam = 1, 
                           lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), maxit = 1000, tol = 1E-8)

L0adri.nnwLS.out2 <- l0adri(X = sim$X, y = sim$y, weights = 1/(sim$y+0.1),
                           family = gaussian(identity), 
                           algo = "osqp", lam = 10, 
                           lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), maxit = 1000, tol = 1E-8)
plot(L0adri.nnwLS.out$beta, L0adri.nnwLS.out2$beta)

bench200<-microbenchmark(
  # Ridge regression using glmnet
  "glmnet" = {
    glmnet_fit <- glmnet(x = x, y = y, family = "gaussian", alpha = 0,
                         standardize = FALSE, thresh = .Machine$double.eps,
                         lambda = 10^seq(10,0), intercept = FALSE)
    # Note: best lambda was tuned with 3-fold cv on sequence 10^seq(-10, 10)
  },
  # L0glm fitting (using glm settings)
  "L0glm" = {
    L0glm_fit <- L0glm(y ~ 0 + ., data = data.frame(y = y, x),
                       family = gaussian(),
                       lambda = 10, tune.meth = "none", nonnegative = TRUE,
                       control.iwls = list(maxit = 25, thresh = .Machine$double.eps),
                       control.l0 = list(maxit = 1),
                       control.fit = list(maxit = 1), verbose = FALSE)
    # Note: best lambda was tuned with 3-fold cv on sequence 10^seq(-10, 10)
  },
  #L0adri fitting using osqp
  "L0adriosqppoisson" = {
    L0adri_osqp_fit_poisson <- l0adri(X=x, y, family = poisson(identity), lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)),
                                      lam = 1, algo = "osqp",maxit = 1000, tol = 1E-8)
  },
  #L0adri fitting using linear slover
  " L0adrilspoisson" = {
    L0adri_ls_fit_poisson <- l0adri(x, y, family = poisson(identity), lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)),
                                    lam = 1, algo = "linearsolver",maxit = 1000, tol = 1E-8)
  },
  #L0adri fitting using osqp
  "L0adriosqpgaussian" = {
    L0adri_osqp_fit_gaussian <- l0adri(x, y, family = gaussian(identity),lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)) ,
                                       lam = 1, algo = "osqp",maxit = 1000, tol = 1E-8)
  },
  #L0adri fitting using linear slover
  "L0adrilsgaussian" = {
    L0adri_ls_fit_gaussian <- l0adri(x, y, family = gaussian(identity), lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)),
                                     lam = 1, algo = "linearsolver",maxit = 1000, tol = 1E-8)
  },
  "nnALASSO_ordinis" = { nnALASSO_fit <- ordinis(x=x,
                                                 y=y,
                                                 penalty = "alasso",
                                                 lower.limits = rep(0, p),
                                                 alpha = 0.95,
                                                 intercept = FALSE,
                                                 standardize = FALSE)
  beta_nnALASSO_ordinis <- nnALASSO_fit$beta[,which.min(BIC(nnALASSO_fit))][-1] },
  "nnMCP_ordinis" = { nnMCP_fit <- ordinis(x=x,
                                           y=y,
                                           weights=1/(y+1),
                                           penalty = "mcp",
                                           lower.limits = rep(0, ncol(x)),
                                           penalty.factor = c(0, 0, rep(1, p-2)),
                                           alpha = 0.95,
                                           intercept = FALSE,
                                           standardize = FALSE)
  beta_nnMCP <- nnMCP_fit$beta[,which.min(BIC(nnMCP_fit))][-1] },
  "nnSCAD_ordinis" = { nnSCAD_fit <- ordinis(x=x,
                                             y=y,
                                             penalty = "scad",
                                             lower.limits = rep(0, ncol(x)),
                                             alpha = 0.95,
                                             intercept = FALSE,
                                             standardize = FALSE)
  beta_nnSCAD <- nnSCAD_fit$beta[,which.min(BIC(nnSCAD_fit))][-1] },
  
  "l0learn"={
    cvfit = L0Learn.cvfit(x=x, y=y, nFolds=5, seed=1, penalty="L0L2", nGamma=5, gammaMin=0.0001, gammaMax=0.1, maxSuppSize=50)
    lapply(cvfit$cvMeans, min)
    optimalGammaIndex = 5 # index of the optimal gamma identified previously
    optimalLambdaIndex = which.min(cvfit$cvMeans[[optimalGammaIndex]])
    optimalLambda = cvfit$fit$lambda[[optimalGammaIndex]][optimalLambdaIndex]
    optimalLambda
    l0learnbeta1<-coef(cvfit, lambda=optimalLambda, gamma=cvfit$fit$gamma[1])
    nnL0Learn_fit1 <- L0Learn.fit(x, y,
                                  loss = "SquaredError",
                                  penalty="L0L2", algorithm="CDPSI",
                                  maxSuppSize=round(200/2), intercept=FALSE,
                                  activeSetNum = 10, maxSwaps = 1000,)
    l0learnbeta1<-as.vector(l0learnbeta1) },
  "ncvreg"={
    cvfit <- cv.ncvreg(x, y)
    plot(cvfit)
    coef(cvfit)
    
    fit <- ncvreg(x, y)
    ncvregbeta<-coef(fit, lambda=cvfit$lambda.min)},
  
  times = 25
)




df <- data.frame(glmnet = coef(glmnet_fit, s = 1)[-1], # first element is an empty intercept
                 L0glm = coef(L0glm_fit),
                 L0adri.osqp.poi = L0adri_osqp_fit_poisson$beta,
                 L0adri.ls.poi = L0adri_ls_fit_poisson$beta,
                 L0adri.osqp.gau = L0adri_osqp_fit_gaussian$beta,
                 L0adri.ls.gau = L0adri_ls_fit_gaussian$beta,
                 l0learn=l0learnbeta1[-1],
                 ncvregco=as.vector(ncvregbeta[-1]),
                 nnALASSO=beta_nnALASSO_ordinis,
                 nnMCP=beta_nnMCP,
                 nnSCAD=beta_nnSCAD
                 
)


index <- c(1:20)


plt1 <- ggplot(df,aes(index))
plt2 <- plt1 + geom_line(aes(y=coef.glmnet, colour="coef.glmnet"))+
  geom_line(aes(y=coef.L0glm, colour="coef.L0glm"))+
  geom_line(aes(y=coef.L0adri.osqp, colour="coef.L0adri.osqp"))+
  geom_line(aes(y=coef.L0adri.ls, colour="coef.L0adri.ls"))

plt3 <- plt2 + ylab("Estimates")
plt4 <- plt3 +  scale_color_manual(name="Algorithm", 
                                   values=c("grey","blue","red","pink"))
plt4  




plta1 <- ggplot(df,aes(index))
plta2 <- plta1 + 

  geom_line(aes(y=coef.L0adri.osqp, colour="coef.L0adri.osqp"))+
  geom_line(aes(y=coef.L0adri.ls, colour="coef.L0adri.ls"))+
  geom_line(aes(y=coef.true, colour="coef.true"))+
 
  geom_line(aes(y= ncvregco,colour=" ncvregco"))+
  geom_line(aes(y= coef.nnALASSO_ordinis,colour=" coef.nnALASSO_ordinis"))+
  geom_line(aes(y=  coef.nnMCP,colour=" coef.nnMCP"))+
  geom_line(aes(y=  coef.nnSCAD,colour="  coef.nnSCAD"))

plta3 <- plta2 + ylab("Estimates")
plta4 <- plta3 +  scale_color_manual(name="Algorithm", 
                                   values=c("grey","blue","green","black","yellow","mistyrose4","gold1"))
plta4  




betas<-df



#increase the number of coefficients to test the sensityivity and specifiicty

set.seed(123)
sim <- simulate_spike_train()
x <- sim$X
y <- sim$y
beta=sim$a
p=200


#copy from before

bench200<-microbenchmark(
  # Ridge regression using glmnet
  "glmnet" = {
    glmnet_fit <- glmnet(x = x, y = y, family = "gaussian", alpha = 0,
                         standardize = FALSE, thresh = .Machine$double.eps,
                         lambda = 10^seq(10,0), intercept = FALSE)
    # Note: best lambda was tuned with 3-fold cv on sequence 10^seq(-10, 10)
  },
 
  #L0adri fitting using osqp
  "L0adriosqppoisson" = {
    cvlam1<-cv.l0adri(X=x, y, weights = 1/(y+0.1), family = poisson(identity), lower =rep(0, ncol(x)), algo="osqp", lambdas= 10^seq(-1,2,length.out = 10), maxit = 10)
    L0adri_osqp_fit_poisson <- l0adri(X=x, y, family = poisson(identity), lower=rep(0,ncol(x)),
                                      lam = cvlam1$lam.min, algo = "osqp",maxit = 1000, tol = 1E-8)
  },
  #L0adri fitting using linear slover
  " L0adrilspoisson" = {
    cvlam2<-cv.l0adri(X=x, y, weights = 1/(y+0.1), family = poisson(identity), lower =rep(0, ncol(x)), algo="linearsolver", lambdas= 10^seq(-1,2,length.out = 10), maxit = 10)
    L0adri_ls_fit_poisson <- l0adri(X=x, y, family = poisson(identity), lower=rep(0,ncol(x)),
                                    lam = cvlam2$lam.min, algo = "linearsolver",maxit = 1000, tol = 1E-8)
  },
  #L0adri fitting using osqp
  "L0adriosqpgaussian" = {
    cvlam3<-cv.l0adri(X=x, y, weights = 1/(y+0.1), family = gaussian(identity), lower =rep(0, ncol(x)), algo="osqp", lambdas= 10^seq(-1,2,length.out = 10), maxit = 10)
    L0adri_osqp_fit_gaussian <- l0adri(X=x, y, family = gaussian(identity),lower=rep(0,ncol(x)),
                                       lam = cvlam3$lam.min, algo = "osqp",maxit = 1000, tol = 1E-8)
  },
  #L0adri fitting using linear slover
  "L0adrilsgaussian" = {
    cvlam4<-cv.l0adri(X=x, y, weights = 1/(y+0.1), family = gaussian(identity), lower =rep(0, ncol(x)), algo="linearsolver", lambdas= 10^seq(-1,2,length.out = 10), maxit = 10)
    L0adri_ls_fit_gaussian <- l0adri(X=x, y, family = gaussian(identity), lower=rep(0,ncol(x)),
                                     lam = cvlam4$lam.min, algo = "linearsolver",maxit = 1000, tol = 1E-8)
  },
  "nnALASSO_ordinis" = { nnALASSO_fit <- ordinis(x=x,
                                                 y=y,
                                                 penalty = "alasso",
                                                 lower.limits = rep(0, p),
                                                 alpha = 0.95,
                                                 intercept = FALSE,
                                                 standardize = FALSE)
  beta_nnALASSO_ordinis <- nnALASSO_fit$beta[,which.min(BIC(nnALASSO_fit))][-1] },
  "nnMCP_ordinis" = { nnMCP_fit <- ordinis(x=x,
                                           y=y,
                                           weights=1/(y+1),
                                           penalty = "mcp",
                                           lower.limits = rep(0, ncol(x)),
                                           penalty.factor = c(0, 0, rep(1, p-2)),
                                           alpha = 0.95,
                                           intercept = FALSE,
                                           standardize = FALSE)
  beta_nnMCP <- nnMCP_fit$beta[,which.min(BIC(nnMCP_fit))][-1] },
  "nnSCAD_ordinis" = { nnSCAD_fit <- ordinis(x=x,
                                             y=y,
                                             penalty = "scad",
                                             lower.limits = rep(0, ncol(x)),
                                             alpha = 0.95,
                                             intercept = FALSE,
                                             standardize = FALSE)
  beta_nnSCAD <- nnSCAD_fit$beta[,which.min(BIC(nnSCAD_fit))][-1] },
 
  "l0learn"={
    cvfit = L0Learn.cvfit(x=x, y=y, nFolds=5, seed=1, penalty="L0L2", nGamma=5, gammaMin=0.0001, gammaMax=0.1, maxSuppSize=50)
    lapply(cvfit$cvMeans, min)
    optimalGammaIndex = 5 # index of the optimal gamma identified previously
    optimalLambdaIndex = which.min(cvfit$cvMeans[[optimalGammaIndex]])
    optimalLambda = cvfit$fit$lambda[[optimalGammaIndex]][optimalLambdaIndex]
    optimalLambda
    l0learnbeta1<-coef(cvfit, lambda=optimalLambda, gamma=cvfit$fit$gamma[1])
    nnL0Learn_fit1 <- L0Learn.fit(x, y,
                                 loss = "SquaredError",
                                 penalty="L0L2", algorithm="CDPSI",
                                 maxSuppSize=round(200/2), intercept=FALSE,
                                 activeSetNum = 10, maxSwaps = 1000,)
    l0learnbeta1<-as.vector(l0learnbeta1) },
  "ncvreg"={
    cvfit <- cv.ncvreg(x, y)
    plot(cvfit)
    coef(cvfit)
    
    fit <- ncvreg(x, y)
    ncvregbeta<-coef(fit, lambda=cvfit$lambda.min)},
  
  times =5
)


df <- data.frame(glmnet = coef(glmnet_fit, s = 1)[-1], # first element is an empty intercept
                 L0adri.osqp.poi = L0adri_osqp_fit_poisson$beta,
                 L0adri.ls.poi = L0adri_ls_fit_poisson$beta,
                 L0adri.osqp.gau = L0adri_osqp_fit_gaussian$beta,
                 L0adri.ls.gau = L0adri_ls_fit_gaussian$beta,
                 l0learn=l0learnbeta1[-1],
                 ncvregco=as.vector(ncvregbeta[-1]),
                 nnALASSO=beta_nnALASSO_ordinis,
                 nnMCP=beta_nnMCP,
                 nnSCAD=beta_nnSCAD
                 
)










betas<-df
p=200



#anothe way to calculate the fp, from the l0glm
#Number of non-zeros in true vector of coefficients
k<-sum(beta!=0)

FP2 <- colSums(betas[(k+1):p,] != 0)
TP2 <- colSums(betas[1:k,] != 0)
FN2 <- colSums(betas[1:k,] == 0)
TN2 <- colSums(betas[(k+1):p,] == 0)

ACC2 = (TP2+TN2)/p # ACCURACY
SENS2 = TP2/(TP2 + FN2) # sensitivity
SPEC2 = TN2/(TN2 + FP2) # specificity


install.packages("export")
library(export)

barplot(ACC2*100, main="Accuracy of different models",
        ,  col="#69b3a2",xlab = "Accuarcy %",horiz = TRUE)


par(las=2) # make label text perpendicular to axis
par(mar=c(5,8,4,2))

barplot(SENS2*100, main="Sensitivity ",col="RED",
        Xlab = "Sensitivity %" ,horiz = TRUE)
par(las=2) # make label text perpendicular to axis
par(mar=c(5,8,4,2))

barplot(SPEC2*100, main="Specificity",col="YELLOW",
        Xlab = "Specificity %",horiz = TRUE)
par(las=2) # make label text perpendicular to axis
par(mar=c(5,8,4,2))



name1<-as.vector(bench200$expr)
mtime<-as.vector(bench200$time)

barplot(mtime,names.arg=name1,main="computing time/minisecons",col="pink",xlab = "models")
