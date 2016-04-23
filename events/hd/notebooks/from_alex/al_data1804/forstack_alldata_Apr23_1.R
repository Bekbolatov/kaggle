library(readr)
library(xgboost)
############################################################
set.seed(123)
############################################################
orig_label <- read_csv('/home/ec2-user/data/hd/unpacked/train.csv')$relevance
train_all <- cbind(read_csv('~/moredata/al_data1804/train_al_data1804.csv'), read_csv('~/moredata/al_data1804/X_train54spel.csv'))
test_all <- cbind(read_csv('~/moredata/al_data1804/test_al_data1804.csv'), read_csv('~/moredata/al_data1804/X_test54spel.csv'))
############################################################
dtest_all <-xgb.DMatrix(data=data.matrix(test_all), missing=NA)
############################################################
############################################################
param <- list(  objective           = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                eta                 = 0.01, 
                #min_child_weight =2,    
                max_depth           = 7, 
                subsample           = 0.7,
                colsample_bytree    = 1.0                
)
############################################################
NROUNDS <- 5000

num_fold<-10
k<-sample(c(1:num_fold),nrow(train_all),replace=TRUE)
indxFile<-data.frame(k = k,TestIndex=c(1:nrow(train_all)))

indxK <- unique(indxFile[,"k"])
prt <- 0
ic <- 1
pr <- orig_label
fono <- 1
for (K in indxK) {
  cat("Next fold: ", fono, "\n")
  fono <- fono + 1
  currentIdx <- indxFile[indxFile[,"k"] == K,"TestIndex"]
  xtrain_B <-  train_all[-currentIdx,]
  xtrain_S <-  train_all[currentIdx,]
  #modeling
  xgtrain = xgb.DMatrix(as.matrix(xtrain_B), label = orig_label[-currentIdx], missing = NA)
  xgtest = xgb.DMatrix(as.matrix(xtrain_S), missing = NA)
  x.mod.t  <- train_all(params = param, data = xgtrain , nrounds = NROUNDS)
  pr[currentIdx]<- predict(x.mod.t, xgtest)
  prt<-prt+ predict(x.mod.t, dtest_all)
}
results = data.frame("alldata_Apr23_1" = pr)
results1 = data.frame("alldata_Apr23_1" = prt/num_fold)

name_tr<-paste('~/moredata/al_data1804/train_Apr23_1_',num_fold,'.csv',sep="")

write.table(results, file = name_tr, row.names = F, col.names = T, sep = ",", quote = F)

name_tst<-paste('~/moredata/al_data1804/test_Apr23_1_',num_fold,'.csv',sep="")

write.table(results1, file = name_tst , row.names = F, col.names = T, sep = ",", quote = F)



