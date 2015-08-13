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

orig_train <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/train.csv')
orig_test <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/test.csv')
n <- nrow(orig_train)
# extract id
train <- orig_train
test <- orig_test
id.test <- test$Id
test$Id <- NULL
train$Id <- NULL
y <- train$Hazard  #log(train$Hazard)*25
train$Hazard <- NULL

# save this 
parsed_train <- train
parsed_test <- test

################################################################################

# as sparse matrices, with OHE  ======  [1] ===========
train <- parsed_train
test <- parsed_test
train_sparse_matrix <- sparse.model.matrix(~., data = train)
test_sparse_matrix <- sparse.model.matrix(~., data = test)

xgtest_dmatrix_from_sparse_matrix <- xgb.DMatrix(data = test_sparse_matrix)

# as matrices, with mean Hz ======  [2] ===========
train <- parsed_train
test <- parsed_test
for (i in 1:ncol(train)) {
  if (class(train[,i])=="factor") {
    mm <- aggregate(y~train[,i], data=train, mean)
    levels(train[,i]) <- as.numeric(mm[,2])
    levels(test[,i]) <- as.numeric(mm[,2])
    train[,i] <- as.numeric(as.character(train[,i]))
    test[,i] <- as.numeric(as.character(test[,i]))
  }
}
train_matrix <- as.matrix(train)
test_matrix <- as.matrix(test)

xgtest_dmatrix_from_matrix <- xgb.DMatrix(data = test_matrix)

# more granularity ======  [3] ===========
# train <- parsed_train
# test <- parsed_test
# for (i in 1:ncol(train)) {
#   if (class(train[,i])=="factor") {
#     train[paste('Cmean', i, sep=":")] <- train[,i]
#     train[paste('Cmin', i, sep=":")] <- train[,i]
#     train[paste('Cmax', i, sep=":")] <- train[,i]
#     train[paste('Csd', i, sep=":")] <- train[,i]
#     test[paste('Cmean', i, sep=":")] <- test[,i]
#     test[paste('Cmin', i, sep=":")] <- test[,i]
#     test[paste('Cmax', i, sep=":")] <- test[,i]
#     test[paste('Csd', i, sep=":")] <- test[,i]
#     mmean <- aggregate(y~train[,i], data=train, mean)
#     mmin <- aggregate(y~train[,i], data=train, min)
#     mmax <- aggregate(y~train[,i], data=train, max)
#     msd <- aggregate(y~train[,i], data=train, sd)
#     levels(train[,c(paste('Cmean', i, sep=":"))]) <- as.numeric(mmean[,2])
#     levels(train[,c(paste('Cmin', i, sep=":"))]) <- as.numeric(mmin[,2])
#     levels(train[,c(paste('Cmax', i, sep=":"))]) <- as.numeric(mmax[,2])
#     levels(train[,c(paste('Csd', i, sep=":"))]) <- as.numeric(msd[,2])
#     levels(test[,c(paste('Cmean', i, sep=":"))]) <- as.numeric(mmean[,2])
#     levels(test[,c(paste('Cmin', i, sep=":"))]) <- as.numeric(mmin[,2])
#     levels(test[,c(paste('Cmax', i, sep=":"))]) <- as.numeric(mmax[,2])
#     levels(test[,c(paste('Csd', i, sep=":"))]) <- as.numeric(msd[,2])
#     
#     train[,c(paste('Cmean', i, sep=":"))] <- as.numeric(as.character(train[,c(paste('Cmean', i, sep=":"))]))
#     train[,c(paste('Cmin', i, sep=":"))] <- as.numeric(as.character(train[,c(paste('Cmin', i, sep=":"))]))
#     train[,c(paste('Cmax', i, sep=":"))] <- as.numeric(as.character(train[,c(paste('Cmax', i, sep=":"))]))
#     train[,c(paste('Csd', i, sep=":"))] <- as.numeric(as.character(train[,c(paste('Csd', i, sep=":"))]))
#     test[,c(paste('Cmean', i, sep=":"))] <- as.numeric(as.character(test[,c(paste('Cmean', i, sep=":"))]))
#     test[,c(paste('Cmin', i, sep=":"))] <- as.numeric(as.character(test[,c(paste('Cmin', i, sep=":"))]))
#     test[,c(paste('Cmax', i, sep=":"))] <- as.numeric(as.character(test[,c(paste('Cmax', i, sep=":"))]))
#     test[,c(paste('Csd', i, sep=":"))] <- as.numeric(as.character(test[,c(paste('Csd', i, sep=":"))]))
#     
#     levels(train[,i]) <- as.numeric(mm[,2])
#     levels(test[,i]) <- as.numeric(mm[,2])
#     train[,i] <- as.numeric(as.character(train[,i]))
#     test[,i] <- as.numeric(as.character(test[,i]))
#     
#   }
# }
# train_combFac_matrix <- as.matrix(train)
# test_combFac_matrix <- as.matrix(test)
# 
# xgtest_combFac_dmatrix_from_matrix <- xgb.DMatrix(data = test_combFac_matrix)

################################################################################
train <- train_sparse_matrix
test <- test_sparse_matrix
################################################################################



offset <- 10000
logfile <- data.frame(shrinkage=        c(0.03, 0.02), # c(0.005,  0.010,  0.015,  0.020,  0.025,  0.030 ),
                      depth =           c(7, 6), #rep(7, times=6)),  #c(3,    4,     5,     6,    7,     8  ),
                      min.child =        c(5,     5,     5 ,   5,    5,     5  ),
                      colsample.bytree = c(0.5,   1,  0.5,     0.5,   0.5,  0.5 ), #c(1), #
                      subsample =        c(1,     1,     1,     1,     1,    1   )) #c(1), # 
models <- 2
repeats <- 1 #10, 20
yhat.test  <- rep(0,n)
avgValScore <- 0
scores <- matrix(numeric(0), repeats, models)
startTime = as.numeric(Sys.time())
for (j in 1:repeats) {
  for (i in 2:models) {
    cat("\n", format(Sys.time(), "%a %b %d %X %Y"), ":", j,  "/", i, "\n")
    set.seed(j*1187 + 0*i*83 + 30002)
    ####   ONLY TRY SAME DATASET TO COMPARE
    shuf = sample(1:n)
    if (i == 1) {
      xgtrain <- xgb.DMatrix(data = train_matrix[shuf[offset:n],], label= y[shuf[offset:n]])
      xgval <-  xgb.DMatrix(data = train_matrix[shuf[1:offset],], label= y[shuf[1:offset]])
      xgtest <- test_matrix
    } else {
      xgtrain <- xgb.DMatrix(data = train[shuf[offset:n],], label= y[shuf[offset:n]])
      xgval <-  xgb.DMatrix(data = train[shuf[1:offset],], label= y[shuf[1:offset]])
      xgtest <- test
    }
    
    watchlist <- list(val=xgval, train=xgtrain)
    
    xgboost.mod <- xgb.train(data = xgtrain, feval = evalgini, nround = 2500, 
                           early.stop.round = 50, maximize = TRUE,
                           print.every.n = 150,
                           watchlist=watchlist, 
                           nthread = 8,
                           eta = logfile$shrinkage[i],
                           subsample = logfile$subsample[i],
                           max.depth = logfile$depth[i],
                           objective = "reg:linear",
                           min.child.weight= logfile$min.child[i],
                           colsample_bytree = logfile$colsample.bytree[i],
                           gamma = 0)

    yhat.test  <- yhat.test + predict(xgboost.mod, xgtest, ntreelimit = xgboost.mod$bestInd)
    validateNumber <- data.frame(label=y[shuf[1:offset]])
    validateNumber$pred <- predict(xgboost.mod, xgval, ntreelimit = xgboost.mod$bestInd)
    score.new <- NormalizedGini(validateNumber$label, validateNumber$pred)
    avgValScore <- avgValScore + score.new
    scores[j, i] <- score.new
    cat("\nPrev Score:", score.prev, " New Score:", score.new)
    elapsedTime <- as.numeric(Sys.time()) - startTime
    avgPerTree <- elapsedTime / ((j-1)*models + i)
    expectedCompletion <- (avgPerTree * (repeats*models - ((j-1)*models + i)))/60
    cat("\nAvg time per tree:", avgPerTree, "Elapsed time:", elapsedTime, "Expected completion in (min):", format(round(expectedCompletion, 1), nsmall=2), "\n\n")
    score.prev <- score.new
    
  }
  scores.compare <- scores[1:j,]
  boxplot(scores.compare, use.cols=T)
}
yhat.test <-  yhat.test/(models*repeats)
avgValScore <- avgValScore / (models*repeats)
cat("\n avg score:", avgValScore)


