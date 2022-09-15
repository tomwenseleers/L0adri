library(L0glm)
library(l0ara)
library(Matrix)
library(L0Learn)
library(nnls)
library(ggplot2)
library(glmnet)
library(ncvreg)
 devtools::install_github("jaredhuling/ordinis")
library(ordinis)
library(bestsubset)
library(informR)
# Simulate some data
n <- 700
p <- 500
s <- 0.1 # sparsity as proportion of p that have nonzero coefficients // p*s == #of peaks
#sim$beta_true== the actual peak
k <- round(p*s) # nr of nonzero covariates
sim <- simulate_spike_train(n = n, p = p, k = k,
                            mean_beta = 1000, sd_logbeta = 1,
                            family = "poisson", seed = 122223, Plot = TRUE)
X = sim$X
y <- sim$y
weights <- rep(1,nrow(X))
#X <- Matrix(X,sparse = T)
l <- seq(0.5,5,by=0.5)

lam1=cv.l0adri(X=X,y=y,lam=l,family=gaussian(identity),nfolds=3,maxit=100,algo='clipping',lower=0)
family <- gaussian(identity)



library(microbenchmark)
microbenchmark("gaussian_clipping" = {gaussian_clipping <- l0adri(X, y, weights=rep(1,nrow(X)),
                                family=gaussian(identity), lam=lam1$lam.min, algo="clipping",maxit=200, tol=1E-8, thresh=1E-3,intercept=F)},
              
               "gaussian_scaling" = {gaussian_scaling <- l0adri(X, y, weights=rep(1,nrow(X)), family=gaussian(identity),
                                                        lam=lam1$lam.min, algo="scaling",maxit=200, tol=1E-8, thresh=1E-3,intercept=F)},
               "poisson_clipping" = {poisson_clipping <- l0adri(X, y, weights=rep(1,nrow(X)),
                                                                  family=poisson(identity), lam=1, algo="clipping",maxit=200, tol=1E-8,intercept=F, thresh=1E-3)},
               "gaussian_osqp" = {gaussian_osqp <- l0adri(X, y, weights=rep(1,nrow(X)), family=poisson(identity), lam=lam1$lam.min,lower=0, algo="osqp",intercept=F,maxit=100, tol=1E-8, thresh=1E-3)},
               "poisson_scaling" = {poisson_scaling <- l0adri(X, y, weights=rep(1,nrow(X)), family=poisson(identity),
                                                                lam=1, algo="scaling",maxit=200, tol=1E-8, thresh=1E-3,intercept=F)},
               "poisson_osqp" = {poisson_osqp <- l0adri(X, y, weights=rep(1,nrow(X)), family=poisson(identity),
                                                          lam=1,lower=0, algo="osqp",maxit=200, tol=1E-8, thresh=1E-3,intercept=F)},
               "L0glm_pois" = { L0glm_pois_fit <- L0glm.fit(X=sim$X, y=sim$y,
                                                            family = poisson(identity),
                                                            lambda = 1, nonnegative = TRUE, normalize = FALSE,
                                                            control.l0 = list(maxit = 25, rel.tol = 1e-04,
                                                                              delta = 1e-05, gamma = 2, warn = FALSE),
                                                            control.iwls = list(maxit = 1, rel.tol = 1e-04, thresh = 1e-03, warn = FALSE),
                                                            control.fit = list(maxit = 1, block.size = NULL, tol = 1e-07)) },
               "L0glm_wgaus" = { L0glm_wgaus_fit <- L0glm.fit(X=sim$X, y=sim$y,
                                                              weights=1/(sim$y+1),
                                                              family = gaussian(identity),
                                                              lambda = 1, nonnegative = TRUE, normalize = FALSE,
                                                              control.l0 = list(maxit = 25, rel.tol = 1e-04,
                                                                                delta = 1e-05, gamma = 2, warn = FALSE),
                                                              control.iwls = list(maxit = 1, rel.tol = 1e-04, thresh = 1e-03, warn = FALSE),
                                                              control.fit = list(maxit = 1, block.size = NULL, tol = 1e-07)) },times=3)
               
weighted.rmse <- function(actual, predicted, weight){                                 #calculates the weights
  sqrt(sum((predicted-actual)^2*weight)/sum(weight))
}
weightedrmse_betas <- function(betas) apply(betas, 2,   function (fitted_coefs) {
  weighted.rmse(actual=sim$y_true,
                predicted=sim$X %*% fitted_coefs,
                weight=1/sim$y_true) } )
library(microbenchmark)
# library(l0ara)
library(L0glm)
library(nnls)

betas = data.frame(beta_true=sim$beta_true,
                   gaussian_osqp = gaussian_osqp$beta,
                   gaussian_clipping = gaussian_clipping$beta,
                   gaussian_scaling = gaussian_scaling$beta,
                   poisson_osqp = poisson_osqp$beta,
                   poisson_clipping = poisson_clipping$beta,
                   poisson_scaling = poisson_scaling$beta,
                   L0glm_gaus = coef(L0glm_wgaus_fit),
                   L0glm_pois = coef(L0glm_pois_fit))
betas[betas<0] = 0                                            #?
TP=colSums(betas[sim$beta_true!=0,]>0) # TPs
tptrack <- betas[sim$beta_true!=0,]>0
FP=colSums(betas[sim$beta_true==0,]>0) # FPs
FN=colSums(betas[sim$beta_true>0,]==0) # FNs
TN=colSums(betas[sim$beta_true==0,]==0) # TNs
ACC = (TP+TN)/p # ACCURACY
SENS = TP/(TP + FN) # sensitivity
SPEC = TN/(TN + FP) # specificity
RELABSBIAS = colMeans(100*abs(betas[sim$beta_true!=0,]-sim$beta_true[sim$beta_true!=0])/sim$beta_true[sim$beta_true!=0])
ABSBIAS = colMeans(abs(betas[sim$beta_true!=0,]-sim$beta_true[sim$beta_true!=0]))

test <- data.frame(wrmse=weightedrmse_betas(betas), TP=TP, FP=FP, FN=FN, TN=TN,
           ACC=ACC, SENS=SENS, SPEC=SPEC, RELABSBIAS=RELABSBIAS, ABSBIAS=ABSBIAS)
###############################################################################################################################
betas[betas<0] = 0                                            #?
TP=colSums(betas[sim1$beta_true!=0,]>0) # TPs
FP=colSums(betas[sim1$beta_true==0,]>0) # FPs
FN=colSums(betas[sim1$beta_true>0,]==0) # FNs
TN=colSums(betas[sim1$beta_true==0,]==0) # TNs
ACC = (TP+TN)/p # ACCURACY
SENS = TP/(TP + FN) # sensitivity
SPEC = TN/(TN + FP) # specificity
RELABSBIAS = colMeans(100*abs(betas[sim1$beta_true!=0,]-sim1$beta_true[sim1$beta_true!=0])/sim1$beta_true[sim1$beta_true!=0])
ABSBIAS = colMeans(abs(betas[sim1$beta_true!=0,]-sim1$beta_true[sim1$beta_true!=0]))

test1 <- data.frame(wrmse=weightedrmse_betas(betas), TP=TP, FP=FP, FN=FN, TN=TN,
                   ACC=ACC, SENS=SENS, SPEC=SPEC, RELABSBIAS=RELABSBIAS, ABSBIAS=ABSBIAS)
#############################################################################################################################
lambda_L0Learn = (nnL0Learn_fit$lambda[[1]]*(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2))[which.min(wrmse_betas_L0Learn)]
nnL0Learn_fit <- L0Learn.fit(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                             loss = "SquaredError",
                             penalty="L0", algorithm="CDPSI",
                             maxSuppSize=round(p/2), intercept=FALSE, nLambda=1,
                             activeSetNum = 10, maxSwaps = 1000,
                             autoLambda=TRUE)
# lambda of 1 is scaled by 1/(norm(weighted y,"2")) because L0Learn scales y by L2 norm of y
beta_nnL0Learn <- as.vector(coef(nnL0Learn_fit))[1:p]
# Set up the parameters for controlling the algorithm
ctrl.fit <- control.fit.gen() # default
ctrl.iwls <- control.iwls.gen(maxit = 1)
ctrl.l0 <- control.l0.gen() # default

# compare solution quality of some different popular sparse learners
library(microbenchmark)
bench_test<- microbenchmark("lmfit" = { lmfit <- lm.fit(x = sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)))
beta_lmfit <- lmfit$coefficients
beta_lmfit[is.na(beta_lmfit)] <- 0
beta_lmfit[beta_lmfit<0] <- 0 },
"nnls" = { beta_nnls <- nnls(A = sim$X*sqrt(1/(sim$y+1)), b=sim$y*sqrt(1/(sim$y+1)))$x },
"gaussian_osqp" = {gaussian_osqp <- l0adri(X, y, weights=rep(1,nrow(X)), family=gaussian(identity), lower=0,intercept = F,lam=lam1$lam.min, algo="osqp",maxit=100, tol=1E-8, thresh=1E-3)},
"poisson_osqp" = {poisson_osqp <- l0adri(X, y, weights=rep(1,nrow(X)), family=poisson(identity),
                                         lam=1, algo="osqp",maxit=200, tol=1E-8, thresh=1E-3,intercept = F,lower=0)},
"nnL0Learnfit" = { nnL0Learn_fit <- L0Learn.fit(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                                                loss = "SquaredError",
                                                penalty="L0", algorithm="CDPSI",
                                                maxSuppSize=round(p/2), intercept=FALSE, nLambda=1,
                                                activeSetNum = 10, maxSwaps = 1000,
                                                autoLambda=TRUE)  # lambda of 1 is scaled by 1/(norm(weighted y,"2")) because L0Learn scales y by L2 norm of y
beta_nnL0Learn <- as.vector(coef(nnL0Learn_fit))[1:p] },
#"doublennL0Learnfit" = { beta_doublennL0Learn <- rep(0,p)
#doublennL0Learn_fit <- L0Learn.fit(x=(sim$X*sqrt(1/(sim$y+1)))[,beta_nnL0Learn>0],
#                                  y=sim$y*sqrt(1/(sim$y+1)),
#                                   loss = "SquaredError",
#                                   penalty="L0", algorithm="CDPSI",
#                                   maxSuppSize=round(p/2), intercept=FALSE, nLambda=1,
#                                   activeSetNum = 10, maxSwaps = 1000,
#                                   autoLambda=TRUE)
#                                   #ambdaGrid=list(1/(norm(sim$y*sqrt(1/(sim$y+1)),"2")^2)) )  # lambda of 1 is scaled by 1/(norm(weighted y,"2")) because L0Learn scales y by L2 norm of y
#beta_doublennL0Learn[beta_nnL0Learn>0] <- coef(doublennL0Learn_fit) },
"nnl0arafit" = { nnl0ara_fit <- l0ara(x = sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)),
                                      family="gaussian", lam=1, standardize=FALSE, maxit=1E4, eps=1E-4)
beta_nnl0ara <- coef(nnl0ara_fit) },
"nnLASSO_glmnet" = { lam.max <- function (X, y) max( abs(crossprod(X,y)) ) / nrow(X) # largest lambda value for LASSO so that no variables would be selected, cf https://stats.stackexchange.com/questions/166630/glmnet-compute-maximal-lambda-value & https://stats.stackexchange.com/questions/292147/how-to-find-the-smallest-lambda-such-that-all-lasso-elastic-net-coefficient?noredirect=1&lq=1
lammax <- lam.max(X=X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)))
nnLASSO_cvfit <- cv.glmnet(x=X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)), family="gaussian",
                           lambda = exp(seq(log(lammax),-20,length.out=50)),
                           intercept=F, standardize=F, nfolds=3, lower.limits=0)
beta_nnLASSO_glmnet <- coef(nnLASSO_cvfit, s = nnLASSO_cvfit$lambda.1se)[-1] }, # lambda.1se or lambda.min give similar results
"nnALASSO_ordinis" = { nnALASSO_fit <- ordinis(x=X,
                                               y=sim$y,
                                               weights=1/(sim$y+1),
                                               penalty = "alasso",
                                               lower.limits = rep(0, p),
                                               alpha = 1,
                                               intercept = FALSE,
                                               standardize = FALSE)
beta_nnALASSO_ordinis <- nnALASSO_fit$beta[,which.min(BIC(nnALASSO_fit))][-1] },
"nnMCP_ordinis" = { nnMCP_fit <- ordinis(x=X,
                                         y=sim$y,
                                         weights=1/(sim$y+1),
                                         penalty = "mcp",
                                         lower.limits = rep(0, p),
                                         alpha = 1,
                                         intercept = FALSE,
                                         standardize = FALSE)
beta_nnMCP <- nnMCP_fit$beta[,which.min(BIC(nnMCP_fit))][-1] },
"nnSCAD_ordinis" = { nnSCAD_fit <- ordinis(x=X,
                                           y=sim$y,
                                           weights=1/(sim$y+1),
                                           penalty = "scad",
                                           lower.limits = rep(0, p),
                                           alpha = 1,
                                           intercept = FALSE,
                                           standardize = FALSE)
beta_nnSCAD <- nnSCAD_fit$beta[,which.min(BIC(nnSCAD_fit))][-1] },
"gaussian_clipping" = {gaussian_clipping <- l0adri(X, y, weights=rep(1,nrow(sim$X)),lower=0,
                                                   family=gaussian(identity), lam=lam1$lam.min,intercept = F, algo="clipping",maxit=200, tol=1E-8, thresh=1E-3)},
#"gaussian_scaling" = {gaussian_scaling <- l0adri(X, y, weights=rep(1,nrow(sim$X)), family=gaussian(identity),lower=0,
 #                                                lam=lam1$lam.min, algo="scaling",maxit=200, tol=1E-8, thresh=1E-3,intercept = F)},
"poisson_clipping" = {poisson_clipping <- l0adri(X, y, weights=rep(1,nrow(sim$X)),lower=0,
                                                 family=poisson(identity), lam=1, algo="clipping",maxit=200,intercept = F, tol=1E-8, thresh=1E-3)},
#"poisson_scaling" = {poisson_scaling <- l0adri(X, y, weights=rep(1,nrow(sim$X)), family=poisson(identity),lower=0,
 #                                              lam=1, algo="scaling",maxit=200, tol=1E-8, thresh=1E-3,intercept = F)},
# "bestsubset" = { bestsubset_fit <- bs(x=sim$X*sqrt(1/(sim$y+1)), y=sim$y*sqrt(1/(sim$y+1)), intercept=FALSE,
#                               k = k) # set k to true value, since tuning it is super slow...
#                  beta_bestsubset <- as.vector(coef(bestsubset_fit))
#                  beta_bestsubset[beta_bestsubset<0] <- 0
#                   },
# often returns error Error 10020: Q matrix is not positive semi-definite (PSD), In lsfit(x[, ids[1:k]], y, int = FALSE) : 'X' matrix was collinea
#"relaxed_LASSO" = { relLASSO_fit <- lasso(x=sim$X*sqrt(1/(sim$y+1)),
#                                         y=sim$y*sqrt(1/(sim$y+1)),
#                                          intercept=FALSE, nrelax=5, nlam=50)
#betas_relLASSO <- coef(relLASSO_fit)
#betas_relLASSO[betas_relLASSO<0] <- 0
#beta_relLASSO <- betas_relLASSO[,which.min(weightedrmse_betas(betas_relLASSO))]
#},
# "stepwise" = { stepwise_fit <- fs(x=sim$X*sqrt(1/(sim$y+1)),
#                   y=sim$y*sqrt(1/(sim$y+1)), intercept=FALSE)
# # returns error Error in `[<-`(`*tmp*`, A, k, value = backsolve(R, t(Q1) %*% y)) : subscript out of bounds
#                betas_stepwise <- coef(stepwise_fit)
#                beta_stepwise <- betas_stepwise[,which.min(weightedrmse_betas(betas_stepwise))] },
  times=1) #


betas <- data.frame(lmfit = beta_lmfit,
                    nnls = beta_nnls,
                    nnL0Learn = beta_nnL0Learn, # beta_L0Learn_bestlambda, # beta_nnL0Learn,
                   # doublennL0Learn = beta_doublennL0Learn,
                    nnl0ara = beta_nnl0ara,
                    nnLASSO_glmnet = beta_nnLASSO_glmnet,
                    nnALASSO_ordinis = beta_nnALASSO_ordinis,
                    nnMCP = beta_nnMCP,
                    nnSCAD = beta_nnSCAD,
                    gaussianclipping = gaussian_clipping$beta,
                   #gaussianscaling = gaussian_scaling$beta,
                   gaussianosqp = gaussian_osqp$beta,
                   poissonclipping = poisson_clipping$beta,
                   #poissonscaling = poisson_scaling$beta,
                   poissonosqp = poisson_osqp$beta
                    # bestsubset = beta_bestsubset,
                   # relLASSO = beta_relLASSO#,
                    # stepwise = beta_stepwise
)

thresh=1
betas[betas<thresh] <- 0
for (col in 1:ncol(betas)) {
  plot(sim$beta_true, betas[,col], pch=16, col="steelblue", ylab="Fitted coefficients", xlab="True coefficients", main=colnames(betas)[col])
}
#plot_L0glm_benchmark(x = sim$x, y = sim$y, fit = nnL0glm_fit, beta_true = sim$beta_true,
#               # inference = L0glm_infer_out,
               main = "Estimated spike train (red=ground truth,\nblue/green=L0 penalized L0glm estimate, green=significant (1-sided p < 0.05)"


wrmse_betas = weightedrmse_betas(betas)
wrmse_betas
wrmse_betas[which.min(wrmse_betas)] # nnL0glm_nnL0Learn = 5.153607e-09
# see https://en.wikipedia.org/wiki/Sensitivity_and_specificity
FP <- colSums((betas>0)&(sim$beta_true==0))
FP
FPgausscal <- colSums((betas$gaussianscaling>0)&(sim$beta_true==0))
FP[FP==0]
TP <- colSums((betas>0)&(sim$beta_true>0))
TP[TP==k]
FN <- colSums((betas==0)&(sim$beta_true>0))
FN[FN==0]
TN <- colSums((betas==0)&(sim$beta_true==0))
TN[TN==(p-k)]
TPR = TP/(TP + FN) # sensitivity = recall = hit rate = true positive rate TPR = power = 1-FNR
TPR
TPR[which.max(TPR)] # nnls = 0.94
FPR = FP/(FP+TN) # false positive rate = fall-out = 1-TNR = Type I error rate
FPR
FPR[which.min(FPR)] # nnL0glm_nnL0Learn = 0.01333333
TNR = TN/(TN + FP) # specificity = selectivity = true negative rate TNR
TNR
TNR[which.max(TNR)] # nnL0glm_nnL0Learn = 0.9866667
TP/(TP+FP) # precision or positive predictive value
FP/(FP+TP) # false discovery rate FDR
FNR = FN/(FN+TP) # false negative rate FNR or miss rate = Type II error rate
FNR
FNR[which.min(FNR)] # nnls = 0.06
PLR = TPR / FPR # positive likelihood ratio (LR+)
PLR
PLR[which.max(PLR)] # nnL0glm_nnL0Learn = 69
NLR = FNR / TNR  # negative likelihood ratio (LR-)
NLR
NLR[which.min(NLR)] # nnL0Learn = 0.06221198
ACC = (TP+TN)/p # accuracy
ACC
ACC[which.max(ACC)] # nnL0glm_nnL0Learn = 0.98
final <- as.data.frame(ACC)
final$TPR <- TPR
final$TP <- TP
final$TN <- TN
final$FN <- FN
final$FP <- FP
final$TNR <- TNR
final$FPR <- FPR
final$FNR <- FNR
final$wrmse <- wrmse_betas
bench_test$time <- bench_test$time/1000000
final$time <- bench_test$time
final$method <- row.names(final)
par(mfrow=c(2,2))
plot(bench_test,units='s') 
p <- ggplot(data=final,aes(x=method,y=ACC),xlim=1) + geom_bar(stat="identity",color="steelblue",fill="steelblue",width = 0.8, position = position_dodge(width = 0.9)) + coord_flip() + scale_y_continuous(limits = c(0,1)) + ggtitle("ACCURACY")
p
q <- ggplot(data=final,aes(x=method,y=TPR),xlim=1) + geom_bar(stat="identity",color="green",fill="green",width = 0.8, position = position_dodge(width = 0.9)) + coord_flip() + scale_y_continuous(limits = c(0,1)) + ggtitle("SENSITIVITY")
q
r <- ggplot(data=final,aes(x=method,y=TNR),xlim=1) + geom_bar(stat="identity",color="red",fill="red",width = 0.8, position = position_dodge(width = 0.9)) + coord_flip() + scale_y_continuous(limits = c(0,1)) + ggtitle("SPECIFICITY")
r
s <- ggplot(data=final,aes(x=method,y=time),xlim=1) + geom_bar(stat="identity",color='black',fill="black",width = 0.8, position = position_dodge(width = 0.9)) + coord_flip() + ggtitle("TIME") + ylab("Seconds")
s
t <- ggplot(data=final,aes(x=method,y=wrmse_betas),xlim=1) + geom_bar(stat="identity",color='black',fill="black",width = 0.8, position = position_dodge(width = 0.9)) + coord_flip() + ggtitle("WRMSE") +scale_y_continuous(limits = c(0,0.1))
t
library(gridExtra)
grid.arrange(p,q,r,s,ncol=2)

microbenchmark(
  # L0 penalized regression using L0Learn
  "L0Learn" = {
    L0Learn_fit <- L0Learn.fit(x = x_norm, y = y_norm, penalty="L0", maxSuppSize = ncol(x),
                               nGamma = 0, autoLambda = FALSE, lambdaGrid = list(0.0015593),
                               tol = 1E-7)
  },
  # L0 penalized regression using L0ara
  "L0ara" = {
    L0ara_fit <- l0ara(x = x_norm, y = y_norm, family = "gaussian", lam = 0.001258925,
                       standardize = FALSE, eps = 1E-7)
  },
  # Best subset regression using bestsubset
  "bestsubset" = {
    bs_fit <- bs(x = x_norm, y = y_norm, k = k, intercept = FALSE,
                 form = ifelse(nrow(x) < ncol(x), 2, 1), time.limit = 5, nruns = 50,
                 maxiter = 1000, tol = 1e-7, polish = TRUE, verbose = FALSE)
  },
  # L0 penalized regression using L0glm
  "L0glm" = {
    L0glm_fit <- L0glm(y_norm ~ 0 + ., data = data.frame(y_norm = y_norm, x_norm),
                       family = gaussian(),
                       lambda = 0.0002511886, tune.meth = "none", nonnegative = FALSE,
                       normalize = FALSE,
                       control.iwls = list(maxit = 100, thresh = 1E-7),
                       control.l0 = list(maxit = 100, rel.tol = 1E-7),
                       control.fit = list(maxit = 1), verbose = FALSE)
  },
  times = 1
)
# Note that bestsubset is optimized using the true number of nonzero coefficient
# because tuning it was much to slow. The algorithm check solution using
# Gurobi's mixed integer program solver which is very slow for k = 10, so time
# limit was set to 5 s which dramatically overestimates to true time
# performance of bestsubset

# Check results
df <- data.frame(coef.L0Learn = coef_scaling*as.numeric(L0Learn_fit$beta[[1]]),
                 coef.L0ara = coef_scaling*L0ara_fit$beta,
                 coef.bestsubset = coef_scaling*as.vector(bs_fit$beta),
                 coef.L0glm = coef_scaling*coef(L0glm_fit),
                 coef.true = beta)
all(df[(k+1):p,] == 0)
# No false positives !
abs(df$coef.L0Learn - df$coef.bestsubset)[1:k]
abs(df$coef.L0glm - df$coef.L0ara)[1:k]
abs(df$coef.L0glm - df$coef.L0Learn)[1:k]

data <- data.frame(y = unlist(df[1:k,]),
                   x = rep(1:k, ncol(df[1:k,])),
                   type = rep(c("L0Learn", "L0ara", "bestsubset", "L0glm", "true"), each = k))
pl <- ggplot(data = data, aes(x = x, y = y, color = type)) +
  geom_point() + geom_line() +
  ggtitle("Compare true coefficients with coefficient estimated \nusing bestsubset, L0ara, L0glm, L0Learn") +
  ylab("Estimate") + xlab("Index") +
  scale_colour_manual(name = "Algorithm",
                      values = c(L0Learn = "red3", L0ara = "orange2",
                                 bestsubset = "purple", L0glm = "green4", true = "grey40"),
                      labels = c(L0Learn = "L0Learn", L0ara = "L0ara",
                                 bestsubset = "bestsubset", L0glm = "L0glm", true = "True"))


gsc <- betas$gaussianscaling
for(i in 1:500)
  new[i] <- ifelse(gsc[i]==0 && sim$beta_true[i]==0,0,1)
  tester[i] <- ifelse(sim$beta_true[i]==0,0,1)
  
gaussian_scaling <- l0adri(X, y, weights=rep(1,nrow(sim$X)), family=gaussian(identity),lower=0,
                                                                            lam=lam1$lam.min, algo="scaling",maxit=200, tol=1E-8, thresh=1E-3,intercept = F)
                             