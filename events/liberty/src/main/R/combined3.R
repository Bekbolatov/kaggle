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

evalgini <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- NormalizedGini(as.numeric(labels),as.numeric(preds))
  return(list(metric = "Gini", value = err))
}

train <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/train.csv')
test <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/test.csv')

# extract id
id.test <- test$Id
test$Id <- NULL
train$Id <- NULL
n <- nrow(train)

y <- log(train$Hazard)*25
train$Hazard <- NULL

for (i in 1:ncol(train)) {
  if (class(train[,i])=="factor") {
    mm <- aggregate(y~train[,i], data=train, mean)
    levels(train[,i]) <- as.numeric(mm[,2])
    levels(test[,i]) <- as.numeric(mm[,2])
    train[,i] <- as.numeric(as.character(train[,i]))
    test[,i] <- as.numeric(as.character(test[,i]))
  }
}

train <- as.matrix(train)
train_y <- as.matrix(y)
test <- as.matrix(test)

xgtest <- xgb.DMatrix(data = test)


offset <- 5000

logfile <- data.frame(shrinkage=c(0.04, 0.03, 0.03, 0.03, 0.02),
                      rounds = c(140, 160, 170, 140, 180),
                      depth = c(8, 7, 9, 10, 10),
                      gamma = c(0, 0, 0, 0, 0),
                      min.child = c(5, 5, 5, 5, 5),
                      colsample.bytree = c(0.7, 0.6, 0.65, 0.6, 0.85),
                      subsample = c(1, 0.9, 0.95, 1, 0.6))


# this will use default evaluation metric = rmse which we want to minimise
models <- 5 #5, 5
repeats <- 12 #10, 20
startTime = as.numeric(Sys.time())
yhat.test  <- rep(0,nrow(xgtest))
for (j in 1:repeats) {
  for (i in 1:models) {
    cat("\n", format(Sys.time(), "%a %b %d %X %Y"), ":", j,  "/", i, "\n")
    set.seed(j*1187 + i*83 + 30001)
    
    shuf = sample(1:n)
    
    xgtrain <- xgb.DMatrix(data = train[shuf[offset:n],], label= train_y[shuf[offset:n]])
    xgval <-  xgb.DMatrix(data = train[shuf[1:offset],], label= train_y[shuf[1:offset]])
    
    watchlist <- list(val=xgval, train=xgtrain)
    
    #bst1 <- xgb.train(params = param, data = xgtrain, nround=num_rounds, print.every.n = 100, watchlist=watchlist, early.stop.round = 50, maximize = FALSE)
    xgboost.mod <- xgb.train(data = xgtrain, nround = 3000, feval = evalgini,
                           print.every.n = 100,  early.stop.round = 60, maximize = TRUE,
                           watchlist=watchlist, 
                           nthread = 8, #8,
                           max.depth = logfile$depth[i],
                           objective = "reg:linear",
                           eta = logfile$shrinkage[i],
                           min.child.weight= logfile$min.child[i],
                           subsample= logfile$subsample[i],
                           colsample_bytree= logfile$colsample.bytree[i],
                           gamma = 0)
    #scale_pos_weight = 1.0, # from chippy
    yhat.test  <- yhat.test + predict(xgboost.mod, xgtest, ntreelimit = xgboost.mod$bestInd)
    validateNumber <- data.frame(label=train_y[shuf[1:offset]])
    validateNumber$pred <- predict(xgboost.mod, xgval, ntreelimit = xgboost.mod$bestInd)
    score.new <- NormalizedGini(validateNumber$label, validateNumber$pred)
    cat("\nPrev Score:", score.prev, " New Score:", score.new)
    elapsedTime <- as.numeric(Sys.time()) - startTime
    avgPerTree <- elapsedTime / ((j-1)*repeats + i)
    expectedCompletion <- (avgPerTree * (repeats*models - ((j-1)*repeats + i)))/60
    cat("\nAvg time per tree:", avgPerTree, "Elapsed time:", elapsedTime, "Expected completion in (min):", format(round(expectedCompletion, 1), nsmall=2), "\n\n")
    score.prev <- score.new
    
  }
}
yhat.test <-  yhat.test/(models*repeats)

write.csv(data.frame(Id=id.test, Hazard=yhat.test),"/Users/rbekbolatov/data/kaggle/liberty/subms/chippy_behar_gini.csv",row.names=F, quote=FALSE)

