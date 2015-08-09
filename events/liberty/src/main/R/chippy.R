library(readr)
library(xgboost)
library(data.table)
library(Matrix)
library(caret)

# The competition datafiles are in the directory ../input
# Read competition data files:
train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")

# keep copy of ID variables for test and train data
train_Id <- train$Id
test_Id <- test$Id

# response variable from training data
train_y <- train$Hazard

# predictor variables from training
train_x <- subset(train, select = -c(Id, Hazard))
train_x <- sparse.model.matrix(~., data = train_x)

# predictor variables from test
test_x <- subset(test, select = -c(Id))
test_x <- sparse.model.matrix(~., data = test_x)


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

# wrap up into a function to be called within xgboost.train
evalgini <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- NormalizedGini(as.numeric(labels),as.numeric(preds))
  return(list(metric = "Gini", value = err))
}


# Set xgboost parameters
param <- list("objective" = "reg:linear",
              "eta" = 0.002,
              "min_child_weight" = 5,
              "subsample" = .8,
              "colsample_bytree" = .8,
              "scale_pos_weight" = 1.0,
              "max_depth" = 8)

# Using 5000 rows for early stopping.
offset <- 5000
num_rounds <- 3000
set.seed(2015)

# Set xgboost test and training and validation datasets
xgtest <- xgb.DMatrix(data = test_x)
xgtrain <- xgb.DMatrix(data = train_x[offset:nrow(train_x),], label= train_y[offset:nrow(train_x)])
xgval <-  xgb.DMatrix(data = train_x[1:offset,], label= train_y[1:offset])

# setup watchlist to enable train and validation, validation must be first for early stopping
watchlist <- list(val=xgval, train=xgtrain)
# to train with watchlist, use xgb.train, which contains more advanced features



#----------------------------------------
# Fit and evaulate using GINI
# Best iteration: 2282 Best Score(GINI): 0.3675499
# Public Leaderboard Gini: 0.346841
#----------------------------------------

bst2a <- xgb.train(params = param, data = xgtrain, feval = evalgini, nround=num_rounds, print.every.n = 50, watchlist=watchlist, early.stop.round = 50, maximize = TRUE)
preds2a <- predict(bst2a,xgtest)
# Output submission
pred2a_df = data.frame(Id = test_Id, Hazard= preds2a)
write.table(pred2a_df, file = '../output/xgboost_2a.csv', row.names = F, col.names = T, sep = ",", quote = F)

#----------------------------------------
# Fit and evaulate using GINI
# Best iteration: 2409 Best Score(RMSE): 3.715605
# Public Leaderboard Gini: 0.380114
#----------------------------------------

bst2b <- xgb.train(params = param, data = xgtrain, nround=num_rounds, print.every.n = 50, watchlist=watchlist, early.stop.round = 50, maximize = FALSE)
preds2b <- predict(bst2b,xgtest)
# Output submission
pred2b_df = data.frame(Id = test_Id, Hazard= preds2b)
write.table(pred2b_df, file = '../output/xgboost_2b.csv', row.names = F, col.names = T, sep = ",", quote = F)




