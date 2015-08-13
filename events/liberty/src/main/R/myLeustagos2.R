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

# Adding a couple of new features (totally made up, just to try)
# train$more1 <- train$T1_V1 - train$T1_V2 * 0.8
# test$more1 <- test$T1_V1 - test$T1_V2 * 0.8
# 
# train$more1_p <- train$T1_V1 + train$T1_V2 * 0.8
# test$more1_p <- test$T1_V1 + test$T1_V2 * 0.8
# 
# train$more2 <- train$T2_V2 - train$T2_V15 * 3.5
# test$more2 <- test$T2_V2 - test$T2_V15 * 3.5
# 
# train$more2_p <- train$T2_V2 + train$T2_V15 * 3.5
# test$more2_p <- test$T2_V2 + test$T2_V15 * 3.5
# 
# train$more3 <- train$T2_V4 - train$T2_V9
# test$more3 <- test$T2_V4 - test$T2_V9
# 
# train$more3_p <- train$T2_V4 + train$T2_V9
# test$more3_p <- test$T2_V4 + test$T2_V9
# 
# train$more4 <- train$T2_V1 - train$T2_V7*2.5
# test$more4 <- test$T2_V1 - test$T2_V7*2.5
# 
# train$more4_p <- train$T2_V1 + train$T2_V7*2.5
# test$more4_p <- test$T2_V1 + test$T2_V7*2.5
# 
# train$more5 <- train$T2_V4 / (train$T2_V9 + 1)
# test$more5 <- test$T2_V4 / (test$T2_V9 + 1)
# 
# train$more6 <- train$T2_V1 / (train$T2_V7 + 1)
# test$more6 <- test$T2_V1 / (test$T2_V7 + 1)
# 
# train$more7 <- train$T2_V2 / (train$T2_V15 + 1)
# test$more7 <- test$T2_V2 / (test$T2_V15 + 1)
# 
# train$more8 <- train$T1_V1 / (train$T1_V2 + 1)
# test$more8 <- test$T1_V1 / (test$T1_V2 + 1)


# extract id
id.test <- test$Id
test$Id <- NULL
train$Id <- NULL
n <- nrow(train)

y <- train$Hazard  #log(train$Hazard)*25
train$Hazard <- NULL

train_sparse_matrix <- sparse.model.matrix(~., data = train)
test_sparse_matrix <- sparse.model.matrix(~., data = test)

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

xgtest_dmatrix_from_sparse_matrix <- xgb.DMatrix(data = test_sparse_matrix)


offset <- 10000

logfile <- data.frame(shrinkage=        c(0.01, 0.01), # c(0.005,  0.010,  0.015,  0.020,  0.025,  0.030 ),
                      depth =           rep(7, times=6),  #c(3,    4,     5,     6,    7,     8  ),
                      min.child =        c(5,     5,     5 ,   5,    5,     5  ),
                      colsample.bytree = c(0.5,   0.5,  0.5,     0.5,   0.5,  0.5 ), #c(1), #
                      subsample =        c(1,     1,     1,     1,     1,    1   )) #c(1), # 

models <- 1
repeats <- 1 #10, 20
startTime = as.numeric(Sys.time())
yhat.test  <- rep(0,nrow(xgtest))
avgValScore <- 0
scores <- matrix(numeric(0), repeats, models)
for (j in 1:repeats) {
  for (i in 1:models) {
    cat("\n", format(Sys.time(), "%a %b %d %X %Y"), ":", j,  "/", i, "\n")
    set.seed(j*1187 + 0*i*83 + 30002)
    ####   ONLY TRY SAME DATASET TO COMPARE
    shuf = sample(1:n)
    if (i == 1) {
      xgtrain <- xgb.DMatrix(data = train[shuf[offset:n],], label= train_y[shuf[offset:n]])
      xgval <-  xgb.DMatrix(data = train[shuf[1:offset],], label= train_y[shuf[1:offset]])
    } else {
      xgtrain <- xgb.DMatrix(data = train[shuf[offset:n],c(1:10, 12:15, 19:32)], label= train_y[shuf[offset:n]])
      xgval <-  xgb.DMatrix(data = train[shuf[1:offset],c(1:10, 12:15, 19:32)], label= train_y[shuf[1:offset]])
      
    }
    
    watchlist <- list(val=xgval, train=xgtrain)
    
    #bst1 <- xgb.train(params = param, data = xgtrain, nround=num_rounds, print.every.n = 100, watchlist=watchlist, early.stop.round = 50, maximize = FALSE)
    xgboost.mod <- xgboost(data = xgtrain, # label = ifelse(d_train$dep_delayed_15min=='Y',1,0),
                           nthread = 8, nround = 1, max_depth = 20,
                           num_parallel_tree = 4500, subsample = 0.632,
                           watchlist=watchlist,  objective = "reg:linear",
                           colsample_bytree = 0.35)
    
#     xgb.train(data = xgtrain, feval = evalgini, nround = 500, 
#                            print.every.n = 150,
#                            watchlist=watchlist, 
#                            nthread = 8, #8,
#                            num_parallel_tree = 500, subsample = 0.632,
#                            max.depth = 20,
#                            objective = "reg:linear", #"rank:pairwise",
#                            min.child.weight= logfile$min.child[i],
#                            colsample_bytree= 0.35,
#                            gamma = 0)
#     #scale_pos_weight = 1.0, # from chippy
    yhat.test  <- yhat.test + predict(xgboost.mod, xgtest, ntreelimit = xgboost.mod$bestInd)
    validateNumber <- data.frame(label=train_y[shuf[1:offset]])
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
  scores.compare.lessfeats <- scores[1:j,]
  boxplot(scores.compare.lessfeats, use.cols=T)
}
yhat.test <-  yhat.test/(models*repeats)
avgValScore <- avgValScore / (models*repeats)
cat("\n avg score:", avgValScore)



#write.csv(data.frame(Id=id.test, Hazard=yhat.test),"/Users/rbekbolatov/data/kaggle/liberty/subms/leustagos_1_only_second_third.csv",row.names=F, quote=FALSE)

# boxplot(scores.compare.models, use.cols=T)
# mean(scores.compare.models)



mean(scores.plain)
mean(scores.plain.noextra)
mean(scores.plain.extra.more1)
mean(scores.plain.extra.more2)
mean(scores.plain.extra.more3)
mean(scores.plain.extra.more4)
mean(scores.plain.extra.more5)
mean(scores.plain.extra.more6)
mean(scores.plain.extra.more7)
mean(scores.plain.extra.more8)
mean(scores.plain.extra.more1.more)
mean(scores.plain.lower.colsample)

# > mean(scores.plain)
# [1] 0.3870998
# > mean(scores.plain.noextra) *
# [1] 0.3878905  
# > mean(scores.plain.extra.more1) **
# [1] 0.3883197
# > mean(scores.plain.extra.more2)
# [1] 0.3872126
# > mean(scores.plain.extra.more3)
# [1] 0.3876028
# > mean(scores.plain.extra.more4)
# [1] 0.3858249
# > mean(scores.plain.extra.more5)
# [1] 0.3865417
# > mean(scores.plain.extra.more6)
# [1] 0.3868264
# > mean(scores.plain.extra.more7)
# [1] 0.3875499

# > colMeans(scores.plain.extra.more1.more)
# [1] 0.3835696 0.3908961 0.3820005 0.3847195 0.3859035
# > mean(scores.plain.shrinkage.half)
# [1] 0.3874439


# boxplot(scores.plain.shrinkage.half, use.cols=T)
# boxplot(scores.plain.noextra, use.cols=T)
# boxplot(scores.plain.extra.more1.more, use.cols=T)
# 
# boxplot(scores.plain.lower.colsample, use.cols=T)
# 
# boxplot(cbind(scores.plain.noextra, scores.plain.lower.colsample[1:3,]), use.cols=T, col=c(rep("green", times = 5), rep("blue", 5)))





