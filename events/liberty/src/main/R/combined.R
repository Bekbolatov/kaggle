library(readr)
library(xgboost)
library(data.table)
library(Matrix)
library(caret)


# build Gini functions for use in custom xgboost evaluation metric
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = (1:nrow(df))/nrow(df)
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
  df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
  df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
  return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

train <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/train.csv')
test <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/test.csv')

# extract id
id.test <- test$Id
test$Id <- NULL
train$Id <- NULL
n <- nrow(train)

# shuffle
train <- train[sample(n),]

# extarct target
y <- log(train$Hazard)*25
train$Hazard <- NULL

test <- sparse.model.matrix(~., data = test)
xgtest <- xgb.DMatrix(data = test)

offset <- 0 #5000 #15000
num_rounds <- 20000
logfile <- data.frame(shrinkage=c(0.01, 0.03, 0.03, 0.03, 0.02),
                      depth = c(5, 7, 9, 10, 10),
                      gamma = c(0, 0, 0, 0, 0),
                      min.child = c(5, 5, 5, 5, 5),
                      colsample.bytree = c(0.3, 0.6, 0.65, 0.6, 0.85),
                      subsample = c(1, 0.9, 0.95, 1, 0.6))
#logfile <- data.frame(shrinkage=c(0.03, 0.03, 0.03, 0.03, 0.02),
#                      depth = c(8, 7, 9, 10, 10),
#                      gamma = c(0, 0, 0, 0, 0),
#                      min.child = c(5, 5, 5, 5, 5),
#                      colsample.bytree = c(0.7, 0.6, 0.65, 0.6, 0.85),
#                      subsample = c(1, 0.9, 0.95, 1, 0.6))



# this will use default evaluation metric = rmse which we want to minimise
models <- 1 #5, 5
repeats <- 50 #10, 20
yhat.test  <- rep(0,nrow(xgtest))
for (j in 1:repeats) {
  for (i in 1:models) {
    print(j)
    print(i)
    set.seed(j*1187 + i*83 + 30001)
    
    shuf = sample(1:n)
    train_train <- sparse.model.matrix(~., data = train[shuf[offset:n],])
    train_validate <- sparse.model.matrix(~., data = train[shuf[1:offset],])
    xgtrain <- xgb.DMatrix(data = train_train, label= y[shuf[offset:n]])
    xgval <-  xgb.DMatrix(data = train_validate, label= y[shuf[1:offset]])
    watchlist <- list(val=xgtrain) # list(val=xgval, train=xgtrain)
    
    #bst1 <- xgb.train(params = param, data = xgtrain, nround=num_rounds, print.every.n = 100, watchlist=watchlist, early.stop.round = 50, maximize = FALSE)
    xgboost.mod <- xgb.train(data = xgtrain, nround = 2600, print.every.n = 500,  #label = y[shuf[offset:n]], #early.stop.round = 400, maximize = FALSE,
                           watchlist=watchlist, 
                           nthread = 8, #8,
                           max.depth = 5, #logfile$depth[i],
                           objective = "reg:linear",
                           eta = 0.01, #logfile$shrinkage[i],
                           min.child.weight= 5, #logfile$min.child[i],
                           subsample= 0.85, #logfile$subsample[i],
                           colsample_bytree=0.25, #logfile$colsample.bytree[i],
                           gamma = 0)
    #scale_pos_weight = 1.0, # from chippy
    yhat.test  <- yhat.test + predict(xgboost.mod, xgtest)
    #validateNumber <- data.frame(label=y[shuf[1:offset]])
    #validateNumber$pred <- predict(xgboost.mod, xgval)
    #score.new <- NormalizedGini(validateNumber$label, validateNumber$pred)
    #print(score.prev)
    #print(score.new)
    #score.prev <- score.new
    
  }
}
yhat.test <-  yhat.test/(models*repeats)

write.csv(data.frame(Id=id.test, Hazard=yhat.test),"/Users/rbekbolatov/data/kaggle/liberty/subms/chippy_behar_50num_increasenumrounds.csv",row.names=F, quote=FALSE)

### ################################################ ### ################################################

finalSubmision <- data.frame(Id=test_Id)
finalSubmision$Hazard <- predict(bst1, test)
write_csv(finalSubmision, "submissionlscriptKaggle.csv")


#  Alternative method


train <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/train.csv')
test <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/test.csv')

# extract id
id.test <- test$Id
test$Id <- NULL
train$Id <- NULL
n <- nrow(train)

# shuffle
train <- train[sample(n),]

# extarct target
y <- log(train$Hazard)*25
train$Hazard <- NULL

# replace factors with level mean hazard
for (i in 1:ncol(train)) {
  if (class(train[,i])=="factor") {
    mm <- aggregate(y~train[,i], data=train, mean)
    levels(train[,i]) <- as.numeric(mm[,2])
    levels(test[,i]) <- as.numeric(mm[,2])
    train[,i] <- as.numeric(as.character(train[,i]))
    test[,i] <- as.numeric(as.character(test[,i]))
  }
}

# Using all
test <- sparse.model.matrix(~., data = test)
train_train <- sparse.model.matrix(~., data = train)


# Set xgboost test and training and validation datasets
xgtest <- xgb.DMatrix(data = test)
xgtrain <- xgb.DMatrix(data = train_train, label= y)

logfile <- data.frame(shrinkage=c(0.04, 0.03, 0.03, 0.03, 0.02),
                      rounds = c(140, 160, 170, 140, 180),
                      depth = c(8, 7, 9, 10, 10),
                      gamma = c(0, 0, 0, 0, 0),
                      min.child = c(5, 5, 5, 5, 5),
                      colsample.bytree = c(0.7, 0.6, 0.65, 0.6, 0.85),
                      subsample = c(1, 0.9, 0.95, 1, 0.6))

# generate final prediction -- bag of 50 models --
models <- 5 #5, 5
repeats <- 20 #10, 20
yhat.test  <- rep(0,nrow(xgtest))
#yhat.val  <- rep(0,nrow(xgval))
for (j in 1:repeats) {
  for (i in 1:models){
    print(j)
    print(i)
    set.seed(j*1187 + i*83 + 30000)
    xgboost.mod <- xgboost(data = xgtrain, label = y,  print.every.n = 50, max.depth = logfile$depth[i], eta = logfile$shrinkage[i],
                           nround = logfile$rounds[i], nthread = 8, objective = "reg:linear", subsample=logfile$subsample[i],
                           colsample_bytree=logfile$colsample.bytree[i], gamma=logfile$gamma[i], min.child.weight=logfile$min.child[i])
    yhat.test  <- yhat.test + predict(xgboost.mod, xgtest)
    #yhat.val  <- yhat.val + predict(xgboost.mod, xgval)
  }
}
yhat.test <-  yhat.test/(models*repeats)
#yhat.val <-  yhat.val/(models*repeats)

#validateNumber <- data.frame(label=y[1:offset])
#validateNumber$pred <- yhat.val
#NormalizedGini(validateNumber$label, validateNumber$pred)
write.csv(data.frame(Id=id.test, Hazard=yhat.test),"/Users/rbekbolatov/data/kaggle/liberty/subms/R_xgboost_benchmark_behar5.csv",row.names=F, quote=FALSE)

yhat.test.total <- yhat.test * (models*repeats)

