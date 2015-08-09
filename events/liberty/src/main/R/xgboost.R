# R XGBOOST starter script
# Thanks Maher Harb (madcap) https://scholar.google.se/citations?user=WBAzJTYAAAAJ&hl=en
#
# The readr library is the best way to read and write CSV files in R
library(readr)
library(reshape2)
library(xgboost)

# The competition datafiles are in the directory ../input
# Read competition data files:
#X <- read.csv("../input/train.csv")
#X.test <- read.csv("../input/test.csv")
X <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/train.csv')
X.test <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/test.csv')

# extract id
id.test <- X.test$Id
X.test$Id <- NULL
X$Id <- NULL
n <- nrow(X)

# extarct target
y <- X$Hazard
X$Hazard <- NULL

# replace factors with level mean hazard
for (i in 1:ncol(X)) {
  if (class(X[,i])=="factor") {
    mm <- aggregate(y~X[,i], data=X, mean)
    levels(X[,i]) <- as.numeric(mm[,2])
    levels(X.test[,i]) <- mm[,2]
    X[,i] <- as.numeric(as.character(X[,i]))
    X.test[,i] <- as.numeric(as.character(X.test[,i]))
  }
}
X <- as.matrix(X)
X.test <- as.matrix(X.test)

# train & tune --skipped--
logfile <- data.frame(shrinkage=c(0.04, 0.03, 0.03, 0.03, 0.02),
                      rounds = c(140, 160, 170, 140, 180),
                      depth = c(8, 7, 9, 10, 10),
                      gamma = c(0, 0, 0, 0, 0),
                      min.child = c(5, 5, 5, 5, 5),
                      colsample.bytree = c(0.7, 0.6, 0.65, 0.6, 0.85),
                      subsample = c(1, 0.9, 0.95, 1, 0.6))

# generate final prediction -- bag of 50 models --
models <- 5
repeats <- 10
yhat.test  <- rep(0,nrow(X.test))
for (j in 1:repeats) {
  for (i in 1:models){
    set.seed(j*1000 + i*100)
    xgboost.mod <- xgboost(data = X, label = y, max.depth = logfile$depth[i], eta = logfile$shrinkage[i],
                           nround = logfile$rounds[i], nthread = 4, objective = "reg:linear", subsample=logfile$subsample[i],
                           colsample_bytree=logfile$colsample.bytree[i], gamma=logfile$gamma[i], min.child.weight=logfile$min.child[i])
    yhat.test  <- yhat.test + predict(xgboost.mod, X.test)
  }
}
yhat.test <-  yhat.test/(models*repeats)
write.csv(data.frame(Id=id.test, Hazard=yhat.test),"R_xgboost_benchmark.csv",row.names=F, quote=FALSE)
