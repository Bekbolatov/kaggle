


ratio_title
pawet_116
ratio_title.x
ratio_title.y


toremove <- c('remfeat', 'pawet_116', 'ratio_title.x', 'ratio_title.y')
train2 <- train[ , !(colnames(train) %in% toremove)]
test2 <- test[ , !(colnames(test) %in% toremove)]



toremove <- c('remfeat', 'pawet_116', 'ratio_title.x', 'ratio_title.y')
train2 <- train[ , !(colnames(train) %in% toremove)]
test2 <- test[ , !(colnames(test) %in% toremove)]




runname <- "forstack_Apr24_xgb01"
NROUNDS <- 5000

dtrain<-xgb.DMatrix(data=data.matrix(train),label=y,missing=NA)
dval<-xgb.DMatrix(data=data.matrix(test),label=y_t,missing=NA)
# for final model
dtrain_all <-xgb.DMatrix(data=data.matrix(train_all), label=orig_label, missing=NA)
dtest_all <-xgb.DMatrix(data=data.matrix(test_all), missing=NA)

param <- list(  objective = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                max_depth           = 7, 
                eta                 = 0.01,
                colsample_bytree    = 0.45,                
                subsample           = 0.9,
                min_child_weight    = 3    
)

set.seed(123)
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = NROUNDS, 
                    early.stop.round    = 50,
                    watchlist           = list(val=dval, train=dtrain),
                    print.every.n       = 20)

importance <- xgb.importance(colnames(train), model = clf)
head(importance)
remfeat <- importance$Feature[1]
cat('dropping ', remfeat, '\n')

train2 <- train[ , !(colnames(train) %in% c(remfeat))]
test2 <- test[ , !(colnames(test) %in% c(remfeat))]

dtrain<-xgb.DMatrix(data=data.matrix(train2),label=y,missing=NA)
dval<-xgb.DMatrix(data=data.matrix(test2),label=y_t,missing=NA)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = NROUNDS, 
                    early.stop.round    = 50,
                    watchlist           = list(val=dval, train=dtrain),
                    print.every.n       = 20)

importance2 <- xgb.importance(colnames(train2), model = clf)
head(importance2)
remfeat <- importance2$Feature[1]
cat('dropping ', remfeat, '\n')

train3 <- train2[ , !(colnames(train2) %in% c(remfeat))]
test3 <- test2[ , !(colnames(test2) %in% c(remfeat))]

dtrain<-xgb.DMatrix(data=data.matrix(train3),label=y,missing=NA)
dval<-xgb.DMatrix(data=data.matrix(test3),label=y_t,missing=NA)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = NROUNDS, 
                    early.stop.round    = 50,
                    watchlist           = list(val=dval, train=dtrain),
                    print.every.n       = 20)




