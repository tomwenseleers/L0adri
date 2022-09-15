#test with other methods

options(java.parameters = c("-XX:+UseConcMarkSweepGC", "-Xmx8192m")) # increase JAVA memory to 8 Gb
library(rJava)
library(glmnet)
library(caret)
# install.packages("~/My drive/stats_courses/Advanced Biological Data Analysis 2021/lectures 2021/statistical machine learning/machine learning/tutorial_lecture/L0adri1.0_1.0.tar.gz", repos = NULL, type = "source")
library(L0adri1.0) # install from source from provided .tar.gz archive
library(mgcv)
library(nnet)
library(e1071)
library(randomForest)
library(xgboost)

library(Metrics)
library(dplyr)
library(effects)
library(ggplot2)
library(ggthemes)

#vs rand

devtools::install_github("jaredhuling/ordinis")

library("ordinis")
install_github("hazimehh/L0Learn")
library("L0Learn")


par(mfrow=c(1,2))
sim <- simulate_spike_train()
X <- sim$X
y <- sim$y

set.seed(1) # set random number generator seed to make script reproducible
data_index = createDataPartition(y, p = .75, list = FALSE)

dim(X) # 322 obs x 2663 variables
X_train = X[data_index, ] # training set covariate values
X_test = X[-data_index, ] # validation test set covariate values

y_train = y[data_index] # training set outcome variable
y_test = y[-data_index] # validation test set outcome variable

set.seed(1)
lasso_cv = cv.glmnet(x = X, y = y, 
                     family = "gaussian", 
                     alpha = 1, # to get LASSO penalty (ie with L1 norm penalty, 0<alpha<1 gives elastic net, with weighted L1 and L2 norm penalty, which can be better with many collinear variables)
                     type.measure = "deviance", # loss to use from cross-validation, "deviance" more general and also works for binomial & Poisson data
                     standardize = TRUE,
                     nfolds = 5, # we do 5-fold cross validation
                     intercept = TRUE) 
optlambda = lasso_cv$lambda.min # optimal lambda that gives minimum out-of-sample MSE
optlambda 

plot(lasso_cv) # MSE ifo lambda sequence
graph2ppt(file="LASSO_cv_lambda_MSE.pptx", width=4, height=4)

# now again fit LASSO model over sequence of lambda values (these are automatically set)
lassomodel = glmnet(X_train, y_train, family = "gaussian", 
                    alpha = 1, # to get LASSO penalty
                    standardize = TRUE,
                    intercept = TRUE) 
plot(lassomodel, xvar="lambda") # coefficients ifo lambda
abline(v=log(optlambda), col="green4", lty=3, lwd=3) # optimal lambda
# higher lambda (greater penalty on the L1 norm of the coefficients) results in some coefficients being penalized down to zero
# and also in the other coefficients being biased a bit towards zero

#graph2ppt(file="LASSO_coefficient paths.pptx", width=8, height=6)


beta = coef(lassomodel, s=optlambda) # model coefficients at optimal lambda value determined using 5-fold cross validation
sum(beta!=0) # nr nonzero coefficients: 52


# show results & validate on test set:
pred_test_lasso <- yhat_test <- predict(lassomodel, newx=X_test, s=optlambda) # predictions test set
pred_train_lasso <- predict(lassomodel, newx=X_train, s=optlambda) # predictions training set
lassoset = which(beta!=0) # set of variables included in the model
rownames(beta)[lassoset] # variables included in the model
# benchmarks calculated on test set
benchm_lasso = c(RMSE = RMSE(yhat_test, y_test), # root mean squared error, sqrt(mean((pred - obs)^2))
                 Rsq = 1 - (sum((yhat_test - y_test) ^ 2))/(sum((y_test - mean(y_test)) ^ 2)), # R squared 
                 MAE = MAE(yhat_test, y_test), # mean absolute error
                 MdAE = median(abs(yhat_test-y_test)), # median absolute error
                 maxdev = max(abs(yhat_test-y_test))) # maximum deviation
benchm_lasso
#  RMSE        Rsq        MAE       MdAE     maxdev 
#15.5357329  0.9966234 10.8926707  9.4236546 66.4439952 

set.seed(1)
ridge_cv = cv.glmnet(x = X, y = y, family = "gaussian", # normally distributed noise, can also be "binomial", "poisson", "multinomial" or "cox" (for survival data)
                     alpha = 0, # to get ridge penalty (ie with L2 norm penalty, 0<alpha<1 gives elastic net, with weighted L1 and L2 norm penalty, which can be better with many collinear variables)
                     lambda.min.ratio = 1E-5, # to put lower bound of lambda sequence used to lower value than default
                     type.measure = "deviance", # loss to use from cross-validation, "deviance" more general (2xlog likelihood) and also works for binomial & Poisson data
                     standardize = TRUE,
                     nfolds = 5, # we do 5-fold cross validation
                     intercept = TRUE) 
optlambda = ridge_cv$lambda.min # optimal lambda that gives minimum out-of-sample MSE
optlambda # 1.826785

plot(ridge_cv) # MSE ifo lambda sequence
graph2ppt(file="ridge_cv_lambda_MSE.pptx", width=4, height=4)

# now again fit ridge model over sequence of lambda values (these are automatically set)
ridgemodel = glmnet(X_train, y_train, family = "gaussian", 
                    alpha = 0, # to get ridge penalty
                    lambda.min.ratio = 1E-5, # to put lower bound of lambda sequence used to lower value than default
                    standardize = TRUE,
                    intercept = TRUE) 
plot(ridgemodel, xvar="lambda") # coefficients ifo lambda
# higher lambda (greater penalty on the L2 norm of the coefficients) results in coefficients being penalized towards zero
# but never quite hitting zero as with LASSO (so no feature selection here; ridge regression can give more stable coefficients though
# and give higher predictive performance when there are many highly collinear variables in the dataset)
abline(v=log(optlambda), col="green4", lty=3, lwd=3) # optimal lambda
#graph2ppt(file="ridge_coefficient paths.pptx", width=8, height=6)

beta = coef(ridgemodel, s=optlambda) # model coefficients at optimal lambda value determined using 5-fold cross validation
sum(beta!=0) # nr nonzero coefficients: 201, almost equal to ncol(X)=201, i.e. almost no coefficients are really penalized down to zero, unlike in LASSO

# show results & validate on test set:
pred_test_ridge <- yhat_test <- predict(ridgemodel, newx=X_test, s=optlambda) # predictions test set
pred_train_ridge <- predict(ridgemodel, newx=X_train, s=optlambda) # predictions training set
ridgeset = which(beta!=0) # set of variables included in the model
rownames(beta)[ridgeset] # variables included in the model
# benchmarks calculated on test set
benchm_ridge <- c(RMSE = RMSE(yhat_test, y_test), # root mean squared error, sqrt(mean((pred - obs)^2))
                  Rsq = 1 - (sum((yhat_test - y_test) ^ 2))/(sum((y_test - mean(y_test)) ^ 2)), # R squared 
                  MAE = MAE(yhat_test, y_test), # mean absolute error
                  MdAE = median(abs(yhat_test-y_test)), # median absolute error
                  maxdev = max(abs(yhat_test-y_test))) # maximum deviation
benchm_ridge
#RMSE        Rsq        MAE       MdAE     maxdev 
#15.6983573  0.9965524 10.1880102  6.7802133 78.7450373 


set.seed(1)
l0adri_cv <- cv.l0adri(X = X, y = y, algo = 'osqp', family = gaussian(identity),
                       lam = 10^seq(-1.5, -0.5, length.out=20), # sequence of lambda regularization parameters 
                       nfolds = 3, # we do 3-fold cross validation here (a bit faster than 5-fold)
                       maxit = 10000, seed = 1,
                       lower=rep(0,ncol(X)), # we do not want nonnegativity constraints on our fitted coefficients here
                        Plot = TRUE) # PS we determine optimal tuning variable lambda using whole dataset
optlambda = l0adri_cv$lam.min # optimal lambda that gives minimum out of sample MSE
optlambda #  0.03162278

#graph2ppt(file="L0adridge_cv_lambda_MSE.pptx", width=4, height=4)


adrimodel = l0adri(X_train, y_train, algo = 'osqp', family = gaussian(identity), 
                   lam = optlambda,   
                   lower=rep(0,ncol(X)), # we do not want nonnegativity constraints on our fitted coefficients here
                    maxit = 10000) 
range(adrimodel$beta) 
sum(adrimodel$beta!=0) # nr nonzero coefficients: 27

# show results & validate on test set:
pred_test_l0adri <- yhat <- X_test %*% adrimodel$beta # predictions test set
pred_train_l0adri <- X_train  %*% adrimodel$beta # predictions training set
L0adriset = which(adrimodel$beta!=0) # set of variables included in the model
colnames(X_train)[L0adriset] # variables included in the model
# [1] "Intercept"      "standard_236"   "standard_330"   "standard_337"   "extended_165"   "maccs_24"       "pubchem_125"    "pubchem_205"    "pubchem_261"    "lingo_94"      
# [11] "Fsp3"           "tpsaEfficiency" "XLogP"          "VP.3"           "maccs_80.1"     "lingo_26.1"     "lingo_120.1" 

# benchmarks calculated on test set
benchm_l0adri <- c(RMSE = RMSE(yhat, y_test), # root mean squared error, sqrt(mean((pred - obs)^2)) 
                   Rsquared = 1 - (sum((yhat - y_test) ^ 2))/(sum((y_test - mean(y_test)) ^ 2)), # R squared 
                   MAE = MAE(yhat, y_test), # mean absolute error
                   MdAE = median(abs(yhat-y_test)), # median absolute error
                   maxdev = max(abs(yhat-y_test))) # maximum deviation
benchm_l0adri

#     RMSE   Rsquared        MAE       MdAE     maxdev 
#13.4537899  0.9974678  9.1409833  6.8840101 53.0500854 


plot(pred_train_l0adri, y_train, 
     xlab="sim$x", ylab="sim$y", pch=16, col="grey",
     main=paste0("L0 adaptive ridge (R2=", as.character(round(benchm_l0adri["Rsquared"]*100,1)), "%)") )
points(pred_test_l0adri, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set,", line=0.8, cex.main=0.9)


graph2ppt(file="obsvspred_L0adridge.pptx", width=6, height=6)



#random forest
model_rf<-randomForest(x=X_train,y=y_train)
pred_test_rf = predict(model_rf, X_test)
pred_train_rf = predict(model_rf, X_train)
benchm_rf = postResample(pred = pred_test_rf, obs = y_test)
benchm_rf
# RMSE  Rsquared       MAE 
#36.489409  0.985573 22.728006 
plot(pred_train_rf, y_train, 
     xlab="sim$x", ylab="sim$y", pch=16, col="grey",
     main=paste0("Random forest (test R2=", as.character(round(benchm_rf["Rsquared"]*100,1)), "%)") )
points(pred_test_rf, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
#graph2ppt(file="obsvspred_rf.pptx", width=6, height=6)

y_pred<-predict(model_rf, X)

line(y_pred)

# 7.3.3 neural net using L0 adaptive ridge features : similar performance on training & test / validation set ####
model_nnet<-nnet(x=X_train,y=y_train,size=10)
pred_test_nnet = predict(model_nnet, X_test)
pred_train_nnet = predict(model_nnet, X_train)
benchm_nnet = postResample(pred = pred_test_nnet, obs = y_test)
benchm_nnet
# RMSE       Rsquared       MAE 

plot(pred_train_nnet, y_train, 
     xlab="sim$x", ylab="sim$y", pch=16, col="grey",
     main=paste0("nnet (L0adri features, test R2=", as.character(round(benchm_nnet["Rsquared"]*100,1)), "%)") )
points(pred_test_nnet, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
#graph2ppt(file="obsvspred_nnet.pptx", width=6, height=6)


# 7.3.4 support vector regression : clearly overfitted on training set & much worse on test / validation set ####

model_svm<-svm(x=X_train,y=y_train)
pred_test_svm = predict(model_svm, X_test)
pred_train_svm = predict(model_svm, X_train)
benchm_svm = postResample(pred = pred_test_svm, obs = y_test)
benchm_svm
#RMSE   Rsquared        MAE 
#32.2555667  0.9956792 23.2016486 



plot(pred_train_svm, y_train, 
     xlab="sim$x", ylab="sim$y", pch=16, col="grey",,
     main=paste0("SVR (test R2=", as.character(round(benchm_svm["Rsquared"]*100,1)), "%)") )
points(pred_test_svm, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
#graph2ppt(file="obsvspred_svr.pptx", width=6, height=6)




#competeting package
model_ncvreg<- ncvreg(X, y)

pred_test_ncvreg = predict(model_ncvreg, X_test)
pred_train_ncvreg = predict(model_ncvreg, X_train)
benchm_ncvreg = postResample(pred = pred_test_svm, obs = y_test)
benchm_ncvreg
#RMSE   Rsquared        MAE 
#32.2555667  0.9956792 23.2016486 

plot(pred_train_ncvreg, y_train, 
     xlab="sim$x", ylab="sim$y", pch=16, col="grey",,
     main=paste0("ncvreg (test R2=", as.character(round(benchm_svm["Rsquared"]*100,1)), "%)") )
points(pred_test_ncvreg, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
#graph2ppt(file="obsvspred_svr.pptx", width=6, height=6)



model_L0learn <- L0Learn.fit(x=X, y=y, penalty="L0", maxSuppSize=20)

pred_L0learn= predict(model_L0learn, X)
pred_train_L0learn= predict(model_ncvreg, X_train)
benchm_L0learn= postResample(pred = pred_test_svm, obs = y_test)
benchm_L0learn



#RMSE   Rsquared        MAE 
#32.2555667  0.9956792 23.2016486 


model_ordinis<- ordinis(x=X, y, 
               penalty = "mcp",
               lower.limits = rep(0, ncol(X)), # force all coefficients to be positive
               penalty.factor = c(0, 0, rep(1, 198)), # don't penalize first two coefficients
               alpha = 0.95)  # use elastic net with alpha = 0.95


pred_test_ordinis= predict(model_ordinis, X_test)
pred_train_ordinis= predict(model_ordinis, X_train)
benchm_ordinis= postResample(pred = pred_test_ordinis, obs = y_test)
benchm_ordinis


plot(y)
abline(model_ordinis$beta)


pred_ordinis= predict(model_ordinis, X)


plot_L0adri_benchmark(x = sim$x, y = y, fit =benchm_ordinis, a.true = sim$a, 
                      main="Ground truth vs L0 penalized L0adri estimates")

