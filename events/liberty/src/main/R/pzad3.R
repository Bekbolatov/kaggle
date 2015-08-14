library(readr)
library(xgboost)
library(data.table)
library(Matrix)
library(caret)
library(e1071)
library(rpart)
library(treeClust)
library(rpart.plot)

source('/Users/rbekbolatov/repos/gh/bekbolatov/kaggle/events/liberty/src/main/R/ginifuncs.R')

orig_train <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/train.csv')
orig_test <- read.csv('/Users/rbekbolatov/data/kaggle/liberty/test.csv')

#### DATA
mm <- rpart(Hazard ~., data=orig_train[,-c(1)], control=rpart.control(cp=0.001, minbucket=100))
orig_train$leaf <- as.factor(rpart.predict.leaves(mm, orig_train[,-c(1, 2)], type="where"))
orig_test$leaf <- as.factor(rpart.predict.leaves(mm, orig_test[,-c(1)], type="where"))
rpart.plot(mm)

mm2 <- rpart(Hazard ~., data=orig_train[,-c(1)], control=rpart.control(cp=0.0005, minbucket=80))
orig_train$leaf2 <- as.factor(rpart.predict.leaves(mm2, orig_train[,-c(1, 2)], type="where"))
orig_test$leaf2 <- as.factor(rpart.predict.leaves(mm2, orig_test[,-c(1)], type="where"))
rpart.plot(mm2)

mm3 <- rpart(Hazard ~., data=orig_train[,-c(1)], control=rpart.control(cp=0.0001, minbucket=50, maxdepth = 6))
orig_train$leaf3 <- as.factor(rpart.predict.leaves(mm3, orig_train[,-c(1, 2)], type="where"))
orig_test$leaf3 <- as.factor(rpart.predict.leaves(mm3, orig_test[,-c(1)], type="where"))
rpart.plot(mm3)

mm4 <- rpart(Hazard ~., data=orig_train[,-c(1)], control=rpart.control(cp=0.00005, minbucket=20, maxdepth = 6))
orig_train$leaf4 <- as.factor(rpart.predict.leaves(mm4, orig_train[,-c(1, 2)], type="where"))
orig_test$leaf4 <- as.factor(rpart.predict.leaves(mm4, orig_test[,-c(1)], type="where"))
rpart.plot(mm4)


# extract id
n <- nrow(orig_train)
train <- orig_train
test <- orig_test
id.test <- test$Id
test$Id <- NULL
train$Id <- NULL
y <- train$Hazard ^ 0.75
y_classes_2 <- ifelse(train$Hazard > 10, 1, 0) 
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
train_sparse_matrix <- sparse.model.matrix(~ . -1, data = train)
test_sparse_matrix <- sparse.model.matrix(~ . -1, data = test)

xgtest_dmatrix_from_sparse_matrix <- xgb.DMatrix(data = test_sparse_matrix)


################################################################################
train <- train_sparse_matrix
test <- test_sparse_matrix
########################   TRY ADDING EXTRA FEATURE (=SVM score for label: hz > 10) ########################################################

offset <- 5000
logfile <- data.frame(shrinkage=c(0.04, 0.03, 0.03, 0.03, 0.02),
                      rounds = c(140, 160, 170, 140, 180),
                      depth = c(8, 7, 9, 10, 10),
                      gamma = c(0, 0, 0, 0, 0),
                      min.child = 10*c(5, 5, 5, 5, 5),    ################### WARNIN  
                      colsample.bytree = c(0.7, 0.6, 0.65, 0.6, 0.85),
                      subsample = c(1, 0.9, 0.95, 1, 0.6))


models <- 5
repeats <- 2

yhat.test  <- rep(0,n)
avgValScore <- 0
score.prev <- 0
scores <- matrix(numeric(0), repeats, models)  
startTime = as.numeric(Sys.time())

for (j in 1:repeats) {
  for (i in 1:models) {
    cat("\n", format(Sys.time(), "%a %b %d %X %Y"), ":", j,  "/", i, "\n")
    set.seed(j*3187 + i*83 + 30002)
    shuf = sample(1:n)
    
    set_tr <- train[shuf[offset:n],]
    set_tr_labels <- y[shuf[offset:n]]
    
    raw_set_cv <- train[shuf[1:offset],]
    raw_set_cv_labels <- y[shuf[1:offset]]
    
    set_cv <- raw_set_cv
    set_cv_labels <- raw_set_cv_labels
    
    xgtrain <- xgb.DMatrix(data = set_tr, label= set_tr_labels)
    xgval <-  xgb.DMatrix(data = set_cv, label= set_cv_labels)
    xgtest <- test
    
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
  cat('Avg score so far:', mean(scores[1:j,]))
  boxplot(scores.compare, use.cols=T)
}
yhat.test <-  yhat.test/(models*repeats)
avgValScore <- avgValScore / (models*repeats)
cat("\n avg score:", avgValScore)


write.csv(data.frame(Id=id.test, Hazard=yhat.test),"/Users/rbekbolatov/data/kaggle/liberty/subms/pzad2_1.csv",row.names=F, quote=FALSE)
scores.compare.seed1 <- scores.compare

