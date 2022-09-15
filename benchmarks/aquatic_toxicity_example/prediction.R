###################################################################################
# MACHINE LEARNING TUTORIAL : PREDICTING AQUATIC TOXICITY OF  CHEMICALS ###########
# PART 2: PREDICT TOXICITY FROM CALCULATED MOLECULAR PROPERTIES & FINGERPRINTS ####
###################################################################################

# DATASET:
# aquatic toxicity (-Log(LC50)) of 233 chemicals for the fathead minnow (Pimephales promelas) reported in
# He and Jurs. Assessing the reliability of a QSAR model's predictions. Journal of Molecular Graphics and Modelling (2005) vol. 23 (6) pp. 503-523
# see also data(AquaticTox) in library(QSARdata)
# plus CAS nrs & molecule names & 4371 molecular fingerprints and properties calculated from these chemical structures
# (see Part 1 of this tutorial to see how that is done)

# PROBLEM ADDRESSED IN PART 2 OF THIS TUTORIAL:
# Make a predictive model of aquatic toxicity (-Log(LC50)) in function of these molecular fingerprints & calculated chemical properties
# to be able to predict aquatic toxicity of other similar chemicals

# load required packages ####
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
library(export)
library(Metrics)
library(dplyr)
library(effects)
library(ggplot2)
library(ggthemes)
# adjust as appropriate

# 1. Load covariate matrix with fingerprints and chemical properties calculated from molecular structures to predict outcome variable (aquatic toxicity) ####

# file with outcome variable (toxicity) and molecular structures (INCHI & SMILES)
DF = read.csv("aquatictox_molecules_withsmiles.csv")
head(DF)
nrow(DF) # 322 molecules, toxicity=outcome variable, inchi & smiles encodes chemical structure

y = DF$toxicity # dependent variable = aquatic toxicity
names(y) =  paste0("Mol_", 1:length(y))

# file with calculated molecular fingerprints and chemical properties calculated from these molecular structures to use to predict toxicity and
# be able to predict the toxicity of other similar chemicals (see Part 1 tutorial to see how this was done)
# i.e. this is our predictor / covariate matrix
X = read.csv("aquatictox_covariates.csv")


# 2. Covariate matrix preprocessing ####

# center and/or L2 norm normalize columns, to center all variables & get them all on the same scale
# function to L2 norm normalize the columns of the design matrix (i.e. to have them on the same scale, same aim as standardization) :
L2norm = function (X) apply(X, 2, function (col) col/norm(as.matrix(col),"2")) # to L2 norm normalize all columns of matrix X
centL2norm = function (X) apply(X, 2, function (col) (col-mean(col))/norm((col-mean(col)),"2")) # to center & L2 norm normalize all columns of matrix X

X = L2norm(X=as.matrix(X)) # L2 norm normalize columns of X

# remove near-invariant variables
X = X[,-nearZeroVar(X, freqCut = 99/1, uniqueCut = 1)]
dim(X) # 322 2662

# sometimes highly collinear variables are removed using
X_nocol = X[,-findCorrelation(cor(X), cutoff = 0.99, verbose = TRUE)]
dim(X_nocol) # 322 1713

# and sometimes linearly dependent variables are removed using
X_nolindep = X[, -findLinearCombos(X)$remove]
dim(X_nolindep) # 322 320

# here we will only remove the near-invariant & highly collinear variables as above, so we continue working with X

# we add a column of ones for intercept
X = cbind(Intercept=1, X)

# for more help on preprocessing using caret, e.g. to include dummy-coded factors see https://topepo.github.io/caret/pre-processing.html


# 3. Partition dataset into training & test set (fixed for all models) ####
set.seed(1) # set random number generator seed to make script reproducible
data_index = createDataPartition(y, p = .75, list = FALSE)

dim(X) # 322 obs x 2663 variables
X_train = X[data_index, ] # training set covariate values
X_test = X[-data_index, ] # validation test set covariate values

y_train = y[data_index] # training set outcome variable
y_test = y[-data_index] # validation test set outcome variable


# 4. LASSO REGRESSION USING glmnet PACKAGE (L1 norm penalized regression) ####

# given that our dataset contains more variables (2663) than observations (322) we have a problem to 
# apply normal multiple regression, as the system of equations would be underdetermined
# to resolve this we can apply regularisation techniques, whereby the objective function
# is to minimize not just the residual sums of squares, but the 
# residual sums of squares + a lambda regularisation parameter x the L1 norm (sum of the absolute value) of your coefficients
# in this way favouring models with smaller coefficients, and with some coefficients penalized down to zero, thereby
# resulting in variable selection (feature selection)
# the optimal regularization parameter is chosen to given optimal out-of-sample predictive performance using cross validation
# for theory see https://en.wikipedia.org/wiki/Lasso_(statistics)
# for details on the glmnet package see https://cran.r-project.org/web/packages/glmnet/vignettes/glmnet.pdf
# this can be done both for linear regression and generalizd linear models & both are supported by glmnet

# first determine optimal lambda regularization parameter using cross validation
set.seed(1)
lasso_cv = cv.glmnet(x = X, y = y, 
                     family = "gaussian", # normally distributed noise, can also be "binomial", "poisson", "multinomial" or "cox" (for survival data)
                     alpha = 1, # to get LASSO penalty (ie with L1 norm penalty, 0<alpha<1 gives elastic net, with weighted L1 and L2 norm penalty, which can be better with many collinear variables)
                     type.measure = "deviance", # loss to use from cross-validation, "deviance" more general and also works for binomial & Poisson data
                     standardize = TRUE,
                     nfolds = 5, # we do 5-fold cross validation
                     intercept = TRUE) 
optlambda = lasso_cv$lambda.min # optimal lambda that gives minimum out-of-sample MSE
optlambda # 0.03124134

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
graph2ppt(file="LASSO_coefficient paths.pptx", width=8, height=6)


beta = coef(lassomodel, s=optlambda) # model coefficients at optimal lambda value determined using 5-fold cross validation
sum(beta!=0) # nr nonzero coefficients: 100

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
#      RMSE  Rsquared       MAE      MdAE    maxdev 
# 0.4598485 0.8433252 0.3469563 0.2538752 1.3406620 
plot(pred_train_lasso, y_train, 
     xlab="Predicted toxicity", ylab="Observed toxicity", pch=16, col="grey", xlim=c(0,7), ylim=c(0,7),
     main=paste0("LASSO regression (test R2=", as.character(round(benchm_lasso["Rsq"]*100,1)), "%)") )
points(pred_test_lasso, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set, 2662 features, 100 features selected)", line=0.8, cex.main=0.9)
graph2ppt(file="obsvspred_LASSO.pptx", width=6, height=6)


# NOTE1: if you would be interested also in significance levels there is the selectiveInference package that provides such 
# post-seletion inference for glmnet models
# https://cran.r-project.org/web/packages/selectiveInference/index.html

# NOTE2: for very large (out-of-core) datasets (e.g. 1000 observations and 10 million variables) there is the biglasso package
# see https://github.com/YaohuiZeng/biglasso


# 5. RIDGE REGRESSION USING glmnet PACKAGE (L2 norm penalized regression) ####

# instead of using the L1 norm penalty (sum of the absolute values of the coefficients)
# we can also use the L2 norm penalty, which then adds the squared magnitude of your
# model coefficients to the residual sums of squares in your objective function
# this will also shrink coefficients towards zero, and help to stabilize coefficients 
# to avoid overfitting but it will not penalize any coefficients all the way to zero
# so it will not result in variable (feature) selection
# it will better retain highly collinear variables though compared to LASSO

# let's see how this would compare with the LASSO method in terms of predictive performance:

# first determine optimal lambda regularization parameter using cross validation
set.seed(1)
ridge_cv = cv.glmnet(x = X, y = y, family = "gaussian", # normally distributed noise, can also be "binomial", "poisson", "multinomial" or "cox" (for survival data)
                     alpha = 0, # to get ridge penalty (ie with L2 norm penalty, 0<alpha<1 gives elastic net, with weighted L1 and L2 norm penalty, which can be better with many collinear variables)
                     lambda.min.ratio = 1E-5, # to put lower bound of lambda sequence used to lower value than default
                     type.measure = "deviance", # loss to use from cross-validation, "deviance" more general (2xlog likelihood) and also works for binomial & Poisson data
                     standardize = TRUE,
                     nfolds = 5, # we do 5-fold cross validation
                     intercept = TRUE) 
optlambda = ridge_cv$lambda.min # optimal lambda that gives minimum out-of-sample MSE
optlambda # 1.787744

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
graph2ppt(file="ridge_coefficient paths.pptx", width=8, height=6)

beta = coef(ridgemodel, s=optlambda) # model coefficients at optimal lambda value determined using 5-fold cross validation
sum(beta!=0) # nr nonzero coefficients: 2660, almost equal to ncol(X)=2663, i.e. almost no coefficients are really penalized down to zero, unlike in LASSO

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
#      RMSE  Rsquared       MAE      MdAE    maxdev 
# 0.5043247 0.8115527 0.3770935 0.2695985 1.6176051 
plot(pred_train_ridge, y_train, 
     xlab="Predicted toxicity", ylab="Observed toxicity", pch=16, col="grey", xlim=c(0,7), ylim=c(0,7), 
     main=paste0("ridge regression (test R2=", as.character(round(benchm_ridge["Rsq"]*100,1)), "%)") )
points(pred_test_ridge, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set, 2660 features in model)", line=0.8, cex.main=0.9)
graph2ppt(file="obsvspred_ridge.pptx", width=6, height=6)



# 6. ITERATIVE ADAPTIVE RIDGE MODEL (BEST SUBSET APPROXIMATION, i.e. L0 norm penalized) USING L0adri1.0 PACKAGE ####

# We can also use ridge regression and iterate the algorithm, so that variables with high coefficients get penalized less
# than other variables (i.e. using unequal penalty weights lambda*penalty weight i). If we choose penalty weight for variable i
# as 1/(coefficient i obtained in previous iteration ^2 + small epsilon) it can be shown that this will eventually cause
# irrelevant variables to be removed from the model (by getting a very high penalty) and important variables to be
# retained in the model. If the lambda regularisation parameter is chosen to give optimal out of sample predictive performance
# it can be shown that the estimate thus obtained after convergence is the best subset selection, i.e. 
# L0 norm penalized regression (the penalty on the residual sums of squares now being the nr of nonzero coefficients)
# see https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0148620
# https://www.sciencedirect.com/science/article/abs/pii/S037837582030135X

# I implemented this method in a new R package L0adri1.0 (source code attached)
# In contrast to other methods to do best subset selection (e.g. using bestglm, glmulti or leaps packages), this method
# scales to large datasets (tens of thousands of variables at least, including high dimensional ones, i.e.
# with more variables than cases in our example here) and it is also implemented for any of the traditional
# generalized linear model families (ie not just for normally distributed data, but e.g. also Poisson counts or binomial data).

set.seed(1)
l0adri_cv <- cv.l0adri(X = X, y = y, algo = 'linearsolver', family = gaussian(identity),
                       lam = 10^seq(-1.5, -0.5), # sequence of lambda regularization parameters 
                       # we do 3-fold cross validation here (a bit faster than 5-fold)
                       maxit = 10, seed = 1,
                       lower  = rep(0, ncol(X)), # we do not want nonnegativity constraints on our fitted coefficients here
                        ) # PS we determine optimal tuning variable lambda using whole dataset
optlambda = l0adri_cv$lam.min # optimal lambda that gives minimum out of sample MSE
optlambda # 0.1528307

graph2ppt(file="L0adridge_cv_lambda_MSE.pptx", width=4, height=4)


adrimodel = l0adri(X_train, y_train, algo = 'linearsolver', family = gaussian(identity), 
                   lam = optlambda,   
                   lower=rep(0,ncol(X)), # we do not want nonnegativity constraints on our fitted coefficients here
                   maxit = 10) 
range(adrimodel$beta) 
sum(adrimodel$beta!=0) # nr nonzero coefficients: 17 (incl intercept)

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
# RMSE      Rsquared  MAE       MdAE      maxdev 
# 0.4810723 0.8285292 0.3689003 0.3020913 1.6127084
plot(pred_train_l0adri, y_train, 
     xlab="Predicted toxicity", ylab="Observed toxicity", pch=16, col="grey", 
     main=paste0("L0 adaptive ridge (R2=", as.character(round(benchm_l0adri["Rsquared"]*100,1)), "%)") )
points(pred_test_l0adri, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set, 2662 features, 17 features selected)", line=0.8, cex.main=0.9)
graph2ppt(file="obsvspred_L0adridge.pptx", width=6, height=6)

# note: the L0adri1.0 currently does not report p values for the selected variables
# in principle those could be calculated using a bootstrapping approach 
# (resampling the original data with replacement & fitting the model on each resampled dataset,
# after which the 95% confidence intervals on the fitted coefficients would be given by the 2.5% and 97.5% 
# percentiles of the coefficients obtained over all fits done on all resampled datasets)
# this feature is planned in a future update


# 7. Train some different machine learning models using caret ("Classification and REgression Training") ####

# for a list of all 238 models available in caret for either regression or classification see https://topepo.github.io/caret/available-models.html

# set options to tune hyperparameters using 5-fold cross validation
set.seed(13435) # random seed to make script reproducible
myFolds = createFolds(y_train, k = 5) # 5 fold for 5-fold cross validation, we will use the same folds for all methods to allow fair comparison
ctrl = trainControl(method="cv", number=5, savePredictions=T, index=myFolds)
# PS method="repeatedcv" with repeats = 5 and myFolds = createMultiFolds(y_train, k = 10, times = 5) 
# would make these folds 5 times, to get more accurate out of sample prediction performance, but slower


# 7.1 TRAIN MODELS ####

# 7.1.1 elastic net (LASSO, ridge or elastic net if alpha = 1, alpha = 0 or 0 < alpha < 1), see library(glmnet), ?glmnet and ?cv.glmnet ####

# the glmnet package not only allows one to fit models with a LASSO (L1 norm) penalty or a ridge (L2 norm) penalty but it can also
# fit models with a weighted average penalty of the two using an extra parameter alpha (0 gives LASSO, 1 gives ridge, in between is called elastic net)
# this can be better than LASSO if there are many collinear variables in the dataset
# the caret package gives an easy option to tune both alpha and the regularisation parameter lambda to give
# the best out-of-sample predictive performed using cross validation:

system.time(model_elasticnet <- try(train(X_train, y_train, method = "glmnet", trControl = ctrl, 
                                          tuneGrid = expand.grid(alpha = c(0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1), # alpha parameters (0 results in ridge penalty, 1 in lasso penalty, in between in weighted penalty of each)
                                                                 lambda = 10^seq(2, -5, length.out=100)), # lambda regularisation parameter, from large to small
                                          intercept=TRUE, # we want an intercept
                                          standardize=TRUE # standardize covariates
),
silent =T)) # 15s 
model_elasticnet$bestTune # alpha=0.8 and lambda=0.1261857 is optimal (ie this would be close to LASSO)
sum(coef(model_elasticnet$finalModel, model_elasticnet$bestTune$lambda)!=0) # 37 nonzero coefficients, i.e. more than in the L0 adaptive ridge / best subset method
elasticnetset = which(coef(model_elasticnet$finalModel, model_elasticnet$bestTune$lambda)>0) # set of selected features with nonzero coefficients
colnames(X_train)[elasticnetset] # included variables
# [1] "Intercept"      "standard_237"   "standard_246"   "standard_274"   "standard_333"   "standard_341"   "extended_12"    "extended_137"   "extended_166"   "maccs_5"       
# [11] "pubchem_113"    "pubchem_119"    "pubchem_126"    "pubchem_215"    "pubchem_262"    "lingo_61"       "WPATH"          "TopoPSA"        "C2SP2"          "ALogp2"        
# [21] "AMR"            "C"              "standard_144.1" "standard_260.1" "pubchem_43.1"   "pubchem_69.1"   "pubchem_222.1"  "graph_28.1"     "graph_142.1"    "graph_157.1"   
# [31] "MDEC.11.1"      "khs.sBr.1"      "Br.1" 

# elastic net coefficient path for increasing lambda (close to LASSO here given high alpha of 0.8, close to 1)
plot(model_elasticnet$finalModel, xvar="lambda")
abline(v=log(model_elasticnet$bestTune$lambda), col="green4", lty=3, lwd=3) # optimal lambda
graph2ppt(file="elasticnet_coefficient paths.pptx", width=8, height=6)


# 7.1.2 generalized additive spline model using features identified by L0 adaptive ridge, see library(mgcv), ?gam ####

# above we identified a best subset of variables explaining our outcome variable using the L0 iterative adaptive ridge method
# this assumed those variables to have a linear effect on the outcome variable

# we can also allow each variable selected by the L0 adaptive ridge method to have additive but nonlinear effects on the outcome variable
# one way of doing this is by allowing spline terms for each of these terms using a generalized additive spline model (gam function in library(mgcv)) :

system.time(model_gam <- try(train(X_train[,L0adriset[-1]], y_train, method = "gamSpline", trControl = ctrl, 
                                   tuneGrid = expand.grid(df = c(2, 3, 4, 5, 6))), # df : nr of spline knots per variable
                             silent =T)) # 1.4s
# effect plots showing influence of each variable (here all perfectly linear, some are binary variables, so for those no splines were fit)
plot(allEffects(model_gam$finalModel))
summary(model_gam$finalModel)


# 7.1.3 neural network (single-hidden-layer neural network) using features identified by L0 adaptive ridge, see library(nnet), ?nnet ####

# another way to allow each variable selected by the L0 adaptive ridge method to have nonlinear effects on the outcome variable toxicity would be
# by using them as features in a neural network
# neural nets also work best with far more observations than features, so prior feature selection is recommended anyway

set.seed(1)
system.time(model_nnet <- try(train(X_train[,L0adriset[-1]], # we use all variables selected by L0 adaptive ridge, except intercept
                                    y_train, method = "nnet", trControl = ctrl,
                                    preProcess = c("center", "scale"), # for this method we center & scale all our variables first
                                    tuneGrid = expand.grid(size = c(1, 2, 4, 8, 16, 32),
                                                           decay = c(0.1, 0.5, 1, 2, 4)), # weight decay - uses an L2 norm of the neural weights as a penalty as a regularization method
                                    linout=T), # linear activation function (set to F for logistic case to get sigmoidal activation function, with output constrained to [0,1] domain)
                              silent =T)) # 11s

# PS: other option is to work on a reduced set of principal components as features (see lectures Hans Jacquemyn), but this gives poor results here
# to get this to run we have to use the covariate matrix with linearly dependent variables removed (X_nolindep_train instead of X_train[,L0adriset[-1]])
# and use preProcess = c("scale","center","pca") 
system.time(model_pcannet <- try(train(X_train, # to make the PCA run we have to reduce some more near-zero variables
                                       y_train, method = "nnet", trControl = ctrl,
                                       preProcess = c("zv", "center", "scale", "pca"), # here we first do a PCA dimensionality reduction to reduce the nr of features to a smaller nr of principal components (see lectures Hans Jacquemyn)
                                       tuneGrid = expand.grid(size = c(1, 2, 4, 8),
                                                              decay = c(1, 10, 100)), # weight decay - uses an L2 norm of the neural weights as a penalty as a regularization method
                                       linout=T), # linear activation function (set to F for logistic case to get sigmoidal activation function, with output constrained to [0,1] domain)
                                 silent =T)) # 29s

# you can also use the full input set of features (nnet has inbuilt regularisation using that weight decay parameter), but this gives worse results
# (neural networks work best when you have many more rows than parameters, in our example we have p > n though, 
# so best to do feature selection or dimensionality reduction first)

# NOTE: for deep learning models you can also use library(h2o) in R, see https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html


# 7.1.4 support vector regression, see library(e1071), ?svm ####
# for introduction on theory behind this see https://towardsdatascience.com/an-introduction-to-support-vector-regression-svr-a3ebc1672c2

# support vector regression models are a machine learning technique that allow nonlinear effects to be modelled & allow all features to be used
# simultaneously

system.time(model_svm <- try(train(X_train, y_train, method = "svmLinear2", trControl = ctrl,
                                   tuneGrid = expand.grid(cost = seq(0.5,2,by=0.1))), # regularisation term - cost of constraints violation (default: 1)
                             silent =T)) # 15s


# 7.1.5 random forest regression (a decision tree based ensemble method that combines several regression tree models into one better model using bagging), see library(randomForest), ?randomForest ####
# for a simple introduction see https://www.section.io/engineering-education/introduction-to-random-forest-in-machine-learning/

# yet another method to do so it the decision tree based ensemble method of random forest regression

system.time(model_rf <- try(train(X_train, y_train, method = "rf", trControl = ctrl,
                                  tuneGrid = expand.grid(mtry=200),  # mtry = number of variables randomly sampled as candidates at each split, here I set it a bit larger than default
                                  ntree=1000 # number of trees to grow
),
silent =T)) # 15s


# 7.1.6 xgboost (an extension of gradient tree boosting, a decision tree based ensemble method that combines several regression tree models into one better model using boosting), see library(xgboost), ?xgboost ####
# for intro see https://xgboost.readthedocs.io/en/latest/tutorials/model.html
# https://www.mygreatlearning.com/blog/xgboost-algorithm/ and details in https://arxiv.org/pdf/1603.02754.pdf

# finally, there is also the xgboost ensemble decision tree based method xgboost, which is nowadays often the winning method in
# many Kaggle machine learning contests

# downside is the many hyperparameters (nrounds, max_depth, eta, gamma, colsample_bytree, subsample, min_child_weight), 
# that require careful tuning to achieve best results and which is a bit of a black art in itself
# see https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/

system.time(model_xgboost <- try(train(X_train, y_train, method = "xgbTree", trControl = ctrl,
                                       objective = "reg:squarederror"
), 
silent =T)) # 54s

# using default hyperparameters here, for more info on tuning xgboost hyperparameters see https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret
# and https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
# results with these defaults not great, but could potentially be improved by fiddling with hyperparameters



# 7.2 CHECK PREDICTION PERFORMANCE ON TRAINING SET ####

# we can look at the performance on the training sets for the different hyperparameters used as follows:

print(model_elasticnet)
print(model_gam)
print(model_nnet)
print(model_pcannet) # this is a bad model, so we skip it below
print(model_svm)
print(model_rf)
print(model_xgboost)

# we can also compare the cross validated RMSE or Rsquared calculated on the training set
mod_list = list(elasticnet = model_elasticnet, gam = model_gam, nnet = model_nnet, svr = model_svm, rf = model_rf, xgbTree = model_xgboost)
resamp = resamples(mod_list)
dotplot(resamp, metric = "RMSE", main="TRAINING SET (5x CV)")
graph2ppt(file="performance_trainingset_RMSE.pptx", width=4, height=4)
dotplot(resamp, metric = "Rsquared", main="TRAINING SET (5x CV)")
graph2ppt(file="performance_trainingset_Rsquared.pptx", width=4, height=4)

# based on cross-validated performance metrics for the training set we would conclude that a 
# neural net trained using the best subset features identified by L0 iterative adaptive ridge regression performs best


# 7.3 CHECK PREDICTION PERFORMANCE BASED ON LEFT-OUT TEST/VALIDATION SET ####

# A truly fair assessment of the predictive performance, however, can only 
# be obtained by checking how each method performs on the left out test (validation) set

# 7.3.1 elastic net / glmnet : performancy on training & test set similar ####
pred_test_elasticnet = predict(model_elasticnet, X_test)
pred_train_elasticnet = predict(model_elasticnet, X_train)
benchm_elasticnet = postResample(pred = pred_test_elasticnet, obs = y_test)
benchm_elasticnet
#      RMSE  Rsquared       MAE 
# 0.4795909 0.8297741 0.3612724
plot(pred_train_elasticnet, y_train, 
     xlab="Predicted toxicity", ylab="Observed toxicity", pch=16, col="grey", xlim=c(0,7), ylim=c(0,7),
     main=paste0("Elastic net (test R2=", as.character(round(benchm_elasticnet["Rsquared"]*100,1)),"%)") )
points(pred_test_elasticnet, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
graph2ppt(file="obsvspred_elasticnet.pptx", width=6, height=6)


# 7.3.2 generalized additive spline model using L0 adaptive ridge features : similar performance on training & test / validation set ####
pred_test_gam = predict(model_gam, X_test)
pred_train_gam = predict(model_gam, X_train)
benchm_gam = postResample(pred = pred_test_gam, obs = y_test)
benchm_gam
# RMSE       Rsquared       MAE 
# 0.5322706 0.8204108 0.3900373   
plot(pred_train_gam, y_train, 
     xlab="Predicted toxicity", ylab="Observed toxicity", pch=16, col="grey", xlim=c(0,7), ylim=c(0,7),
     main=paste0("gam (L0adri features, test R2=", as.character(round(benchm_gam["Rsquared"]*100,1)), "%)") )
points(pred_test_gam, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
graph2ppt(file="obsvspred_gam.pptx", width=6, height=6)


# 7.3.3 neural net using L0 adaptive ridge features : similar performance on training & test / validation set ####
pred_test_nnet = predict(model_nnet, X_test)
pred_train_nnet = predict(model_nnet, X_train)
benchm_nnet = postResample(pred = pred_test_nnet, obs = y_test)
benchm_nnet
# RMSE       Rsquared       MAE 
# 0.4676438 0.8496612 0.3671535  
plot(pred_train_nnet, y_train, 
     xlab="Predicted toxicity", ylab="Observed toxicity", pch=16, col="grey", xlim=c(0,7), ylim=c(0,7),
     main=paste0("nnet (L0adri features, test R2=", as.character(round(benchm_nnet["Rsquared"]*100,1)), "%)") )
points(pred_test_nnet, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
graph2ppt(file="obsvspred_nnet.pptx", width=6, height=6)


# 7.3.4 support vector regression : clearly overfitted on training set & much worse on test / validation set ####
pred_test_svm = predict(model_svm, X_test)
pred_train_svm = predict(model_svm, X_train)
benchm_svm = postResample(pred = pred_test_svm, obs = y_test)
benchm_svm
# RMSE    Rsquared         MAE 
# 0.5495187 0.7913340 0.4092727
plot(pred_train_svm, y_train, 
     xlab="Predicted toxicity", ylab="Observed toxicity", pch=16, col="grey", xlim=c(0,7), ylim=c(0,7),
     main=paste0("SVR (test R2=", as.character(round(benchm_svm["Rsquared"]*100,1)), "%)") )
points(pred_test_svm, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
graph2ppt(file="obsvspred_svr.pptx", width=6, height=6)


# 7.3.5 random forest model : performance slightly better on training than on validation set (ie slightly overfitted) ####
pred_test_rf = predict(model_rf, X_test)
pred_train_rf = predict(model_rf, X_train)
benchm_rf = postResample(pred = pred_test_rf, obs = y_test)
benchm_rf
# RMSE   Rsquared        MAE 
# 0.5498790 0.7802479 0.4193554
plot(pred_train_rf, y_train, 
     xlab="Predicted toxicity", ylab="Observed toxicity", pch=16, col="grey", xlim=c(0,7), ylim=c(0,7),
     main=paste0("Random forest (test R2=", as.character(round(benchm_rf["Rsquared"]*100,1)), "%)") )
points(pred_test_rf, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
graph2ppt(file="obsvspred_rf.pptx", width=6, height=6)


# 7.3.6 xgboost : not much overfitted ####
pred_test_xgboost = predict(model_xgboost, X_test)
pred_train_xgboost = predict(model_xgboost, X_train)
benchm_xgboost = postResample(pred = pred_test_xgboost, obs = y_test)
benchm_xgboost
# RMSE    Rsquared         MAE 
# 0.5089919 0.8116597 0.3956506
plot(pred_train_xgboost, y_train, 
     xlab="Predicted toxicity", ylab="Observed toxicity", pch=16, col="grey", xlim=c(0,7), ylim=c(0,7),
     main=paste0("xgbTree (test R2=", as.character(round(benchm_xgboost["Rsquared"]*100,1)), "%)") )
points(pred_test_xgboost, y_test, pch=16, col="steelblue" )
title("(grey=training set, blue=validation set)", line=0.8, cex.main=0.9)
graph2ppt(file="obsvspred_xgbTree.pptx", width=6, height=6)


# 7.3.7 dotplot with performance on test set for different methods ####

benchm = rbind(nnet = benchm_nnet, gam = benchm_gam, elasticnet = benchm_elasticnet, rf = benchm_rf, xgbTree = benchm_xgboost, svr = benchm_svm)
# benchm = benchm[order(benchm[,"RMSE"], decreasing=F),]
dotplot(benchm[,"RMSE"], xlab="RMSE", main="TEST SET")
graph2ppt(file="performance_testset_RMSE.pptx", width=4, height=4)
dotplot(benchm[,"Rsquared"], xlab="Rsquared", main="TEST SET")
graph2ppt(file="performance_testset_Rsquared.pptx", width=4, height=4)



# 7.4 CHECK VARIABLE IMPORTANCE ####

# we can also look at the variable importance (scaled between 0 and 100) of the features included or
# selected by each of the methods (we only show the top 20):

plot(varImp(model_elasticnet, scale = T), top=20, main="elasticnet")
graph2ppt(file="varimp_elasticnet.pptx", width=4, height=7)
plot(varImp(model_gam, scale = T), top=20, main="gam")
graph2ppt(file="varimp_gam.pptx", width=4, height=7)
plot(varImp(model_nnet, scale = T), top=20, main="nnet")
graph2ppt(file="varimp_nnet.pptx", width=4, height=7)
plot(varImp(model_svm, scale = T), top=20, main="svr")
graph2ppt(file="varimp_svr.pptx", width=4, height=7)
plot(varImp(model_rf, scale = T), top=20, main="rf")
graph2ppt(file="varimp_rf.pptx", width=4, height=7)
plot(varImp(model_xgboost, scale = T), top=20, main="xgbTree")
graph2ppt(file="varimp_xgbTree.pptx", width=4, height=7)

# variables that repeatedly appear as important are
# XlogP (octanol/water partition coefficients, i.e. lipophilicity, which is strongly predictive of uptake in the body)
# ALogP (Ghose-Crippen-Viswanadhan octanol-water partition coefficient, i.e. lipophilicity, which is strongly predictive of uptake in the body)

# VAdjMat: represents the vertex adjacency information and gives information about molecular dimension and hydrophobicity

# Fsp3 (fraction of C atoms that are SP3 hybridized, i.e. tetrahedral, e.g. dichloromethane and other organochlorides, organobromides, alcohols, amines, ethers)

# VP.3 (vapor pressure, i.e. volatility; also smaller molecules)



# 7.5 CONCLUSION ####

# A neural net (nnet) fitted on the L0 adaptive ridge selected best subset of features
# performed best and gave a predictive performance with an out-of-sample R2 of 85%
# The linear L0 iterative adaptive ridge best subset model in itself also still gave a good out-of-sample predictive performance (82.9%),
# despite being a compact and simple multiple regression model with only 16 selected variables (plus intercept).

# These models could be used to make predictions of aquatic toxicity for other similar molecules.

# The octanol+water partition coefficient (a measure of lipophilicity, facilitating uptake in the body) 
# and the fraction of C atoms that were SP3 hybridized were among the key variables that were predictive of toxicity.
