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

#### DATA SAMPLE
#orig_train <- orig_train[orig_train$Hazard < 20, ]


n <- nrow(orig_train)
# extract id
train <- orig_train
test <- orig_test
id.test <- test$Id
test$Id <- NULL
train$Id <- NULL
y <- train$Hazard 
train$Hazard <- NULL

# save this 
parsed_train <- train
parsed_test <- test




  ############             ##################################################################
 ############     START   ##################################################################
############             ##################################################################




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

################################################################################
# train <- train_sparse_matrix
# test <- test_sparse_matrix
train <- train_matrix
test <- test_matrix
################################################################################


offset <- 15000
# train & tune --skipped--
logfile <- data.frame(shrinkage=c(0.04, 0.03, 0.03, 0.03, 0.02),
                      rounds = c(140, 160, 170, 140, 180),
                      depth = c(8, 7, 9, 10, 10),
                      gamma = c(0, 0, 0, 0, 0),
                      min.child = c(5, 5, 5, 5, 5),
                      colsample.bytree = c(0.7, 0.6, 0.65, 0.6, 0.85),
                      subsample = c(1, 0.9, 0.95, 1, 0.6))


models <- 5
repeats <- 5

yhat.test  <- rep(0,n)
avgValScore <- 0
score.prev <- 0
scores <- matrix(numeric(0), repeats, models)
startTime = as.numeric(Sys.time())

for (j in 1:repeats) {
  for (i in 1:models) {
    cat("\n", format(Sys.time(), "%a %b %d %X %Y"), ":", j,  "/", i, "\n")
    set.seed(j*1187 + i*83 + 30002)
    #set.seed(j*1187 + 0*i*83 + 30002)
    ####   ONLY TRY SAME DATASET TO COMPARE
    shuf = sample(1:n)
    
    set_tr <- train[shuf[offset:n],]
    set_tr_labels <- y[shuf[offset:n]]
    
    raw_set_cv <- train[shuf[1:offset],]
    raw_set_cv_labels <- y[shuf[1:offset]]
    
#     set_cv <- raw_set_cv[raw_set_cv_labels < sqrt(60),]
#     set_cv_labels <- raw_set_cv_labels[raw_set_cv_labels < sqrt(60)]
    set_cv <- raw_set_cv
    set_cv_labels <- raw_set_cv_labels
    
    xgtrain <- xgb.DMatrix(data = set_tr, label= set_tr_labels)
    xgval <-  xgb.DMatrix(data = set_cv, label= set_cv_labels)
    xgtest <- test_matrix

    watchlist <- list(val=xgval, train=xgtrain)
    
    xgboost.mod <- xgb.train(data = xgtrain, feval = evalgini, nround = 2500, 
                           early.stop.round = 250, maximize = TRUE,
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
    
    validateNumber <- data.frame(label=set_cv_labels)
    validateNumber$pred <- predict(xgboost.mod, xgval, ntreelimit = xgboost.mod$bestInd)
    score.new <- NormalizedGini(validateNumber$label, validateNumber$pred)
    avgValScore <- avgValScore + score.new
    scores[j, i] <- score.new
    cat("\nPrev Score:", score.prev, " New Score:", score.new)
    elapsedTime <- as.numeric(Sys.time()) - startTime
    avgPerTree <- elapsedTime / ((j-1)*models + i)
    expectedCompletion <- (avgPerTree * (repeats*models - ((j-1)*models + i)))/60
    cat("\nAvg time per tree:", avgPerTree, "Elapsed time (min):", format(round(elapsedTime/60, 1), nsmall=2), "Expected completion in (min):", format(round(expectedCompletion, 1), nsmall=2), "\n\n")
    score.prev <- score.new
    
  }
  scores.compare <- scores[1:j,]
  print('Avg score so far:', mean(scores[1:j,]))
  boxplot(scores.compare, use.cols=T)
}
yhat.test <-  yhat.test/(models*repeats)
avgValScore <- avgValScore / (models*repeats)
cat("\n avg score:", avgValScore)

write.csv(data.frame(Id=id.test, Hazard=yhat.test),"/Users/rbekbolatov/data/kaggle/liberty/subms/chippy_behar_uw1_3.csv",row.names=F, quote=FALSE)
