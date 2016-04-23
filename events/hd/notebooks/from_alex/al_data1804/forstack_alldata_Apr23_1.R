# model
library(readr)
library(xgboost)
cat("Reading data\n")
train <- read_csv('~/d_hd/train.csv')
y<-train$relevance
train <- read_csv('~/moredata/al_data1804/train_al_data1804.csv')
train2 <- read_csv('~/moredata/al_data1804/X_train54spel.csv')
train<-cbind(train,train2)
rm(train2)

test <- read_csv('~/moredata/al_data1804/test_al_data1804.csv')
test2 <- read_csv('~/moredata/al_data1804/X_test54spel.csv')
test<-cbind(test,test2)
rm(test2)

gc()

param <- list(  objective           = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                eta                 = 0.01, 
                #min_child_weight =2,    
                max_depth           = 7, 
                subsample           = 0.7,
                colsample_bytree    = 1.0                
)

xgtestF = xgb.DMatrix(as.matrix(test), missing = NA)
set.seed(123)
num_fold<-10
k<-sample(c(1:num_fold),nrow(train),replace=TRUE)
indxFile<-data.frame(k = k,TestIndex=c(1:nrow(train)))

indxK <- unique(indxFile[,"k"])
prt<-0
ic<-1
pr<-y
for (K in indxK) {
  currentIdx <- indxFile[indxFile[,"k"] == K,"TestIndex"]
  xtrain_B <-  train[-currentIdx,]
  xtrain_S <-  train[currentIdx,]
  set.seed(123)
  #modeling
  xgtrain = xgb.DMatrix(as.matrix(xtrain_B), label = y[-currentIdx], missing = NA)
  xgtest = xgb.DMatrix(as.matrix(xtrain_S), missing = NA)
  x.mod.t  <- xgb.train(params = param, data = xgtrain , nrounds =1517)
  pr[currentIdx]<- predict(x.mod.t,xgtest)
  prt<-prt+ predict(x.mod.t,xgtestF)
}
results = data.frame("alldata_Apr23_1" = pr)
results1 = data.frame("alldata_Apr23_1" = prt/num_fold)

name_tr<-paste('~/moredata/al_data1804/train_Apr23_1_',num_fold,'.csv',sep="")

write.table(results, file = name_tr, row.names = F, col.names = T, sep = ",", quote = F)

name_tst<-paste('~/moredata/al_data1804/test_Apr23_1_',num_fold,'.csv',sep="")

write.table(results1, file = name_tst , row.names = F, col.names = T, sep = ",", quote = F)



