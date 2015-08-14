require(xgboost)
library(caret)
library(randomForest)
library(readr)

# load raw data
train = read_csv('../input/train.csv')
test = read_csv('../input/test.csv')

# Create the response variable
y = train$Hazard

# Create the predictor data set and encode categorical variables using caret library.
mtrain = train[,-c(1,2)]
mtest = test[,-c(1)]
dummies <- dummyVars(~ ., data = mtrain)
mtrain = predict(dummies, newdata = mtrain)
mtest = predict(dummies, newdata = mtest)

cat("Training model - RF\n")
set.seed(8)
rf <- randomForest(mtrain, y, ntree=1000, imp=TRUE, sampsize=10000, do.trace=TRUE)
predict_rf <- predict(rf, mtest)

# Set necessary parameters and use parallel threads
param <- list("objective" = "reg:linear", "nthread" = 8, "verbose"=0)

cat("Training model - Xgboost\n")
# Fit the model
xgb.fit = xgboost(param=param, data = mtrain, label = y, nrounds=3000, eta = .01, max_depth = 7,
                  min_child_weight = 5, scale_pos_weight = 1.0, subsample=0.8)
predict_xgboost <- predict(xgb.fit, mtest)

# Predict Hazard for the test set
submission <- data.frame(Id=test$Id)
submission$Hazard <- (predict_rf+predict_xgboost)/2
write_csv(submission, "submission1.csv")
