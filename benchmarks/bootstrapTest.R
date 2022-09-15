# TEST PLACE, REMOVE IF THIS WORKS
sim <- simulate_spike_train()
X <- sim$X
y <- sim$y
nr.boot = 100
boot.matrix <- matrix(0,nrow = nr.boot,ncol = nrow(X))
dim(boot.matrix)  

for(i in 1:nr.boot){
  
  train=sample(x = nrow(X),size = nrow(X),replace=TRUE)
  X.train = X[train,]
  y.train = y[train]
  
  fit <- l0adri(X.train, y.train, family = poisson(identity),
                algo = "osqp", lam = 1,
                lower=rep(0,ncol(X)),upper=rep(1000,ncol(X)), maxit = 1000, tol = 1E-8)
  boot.matrix[i,] = fit$beta
  
}

dim(boot.matrix)

CI <- matrix(0,nrow = ncol(boot.matrix),3)
colnames(CI) = c("2,5%","97.5%","p-value")
for(i in 1:ncol(boot.matrix)) {
  ci <- quantile(boot.matrix[,i], c(.025, .975))
  CI[i,1:2] <- ci
  
  #p-values: whether the coeff. are 0 or not: 1-proportion of coeff. non-zero. => p-value x 2 
  pvalue <- 1-(sum(length(which(boot.matrix[,i] != 0)))/nrow(boot.matrix))
  CI[i,3] <- pvalue
}
CI