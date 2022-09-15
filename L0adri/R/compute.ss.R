compute.ss <- function(mu, y, w, fam){
  eta <- fam$linkfun(mu)
  w <- w * as.vector(fam$mu.eta(eta)^2 / fam$variance(mu)) # IWLS weights from last iteration
  z.res <- (y - mu)/fam$mu.eta(eta) # the residuals on the adjusted scale = z.res = z - z.fit = eta + (y - mu)/g'(eta) - eta = (y - mu)/g'(eta)
  #print(length(z.res))
  ss <- sum(w * z.res^2) # sums of squares
  return(ss)
}