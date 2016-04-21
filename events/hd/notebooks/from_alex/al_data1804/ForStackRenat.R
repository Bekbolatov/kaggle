# model
library(readr)
library(xgboost)

cat("Reading data\n")
train <- read_csv('/home/ec2-user/data/hd/unpacked/train.csv')
y<-train$relevance

# Renat (all features, except bad 6)
train <- read_csv('/home/ec2-user/moredata/al_data1804/renat_train.csv')
test <- read_csv('/home/ec2-user/moredata/al_data1804/renat_test.csv')

gc()

param <- list(  objective           = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                eta                 = 0.01, 
                min_child_weight = 3,    
                max_depth           = 8, 
                subsample           = 0.9,
                colsample_bytree    = 0.6                
)
NROUNDS = 1876
FEATSNAME = "Renat"

#  FROM HERE SAME
xgtestF = xgb.DMatrix(as.matrix(test), missing = NA)
set.seed(123)
num_fold<-10
k<-sample(c(1:num_fold),nrow(train),replace=TRUE)
indxFile<-data.frame(k = k,TestIndex=c(1:nrow(train)))

indxK <- unique(indxFile[,"k"])
prt<-0
ic<-1
pr<-y
fono <-1
for (K in indxK) {
  cat("Next fold: ", fono, "\n")
  fono <- fono + 1
  currentIdx <- indxFile[indxFile[,"k"] == K,"TestIndex"]
  xtrain_B <-  train[-currentIdx,]
  xtrain_S <-  train[currentIdx,]
  set.seed(123)
  #modeling
  xgtrain = xgb.DMatrix(as.matrix(xtrain_B), label = y[-currentIdx], missing = NA)
  xgtest = xgb.DMatrix(as.matrix(xtrain_S), missing = NA)
  x.mod.t  <- xgb.train(params = param, data = xgtrain , nrounds = NROUNDS, watchlist= list(train=xgtrain), print.every.n = 50)
  pr[currentIdx]<- predict(x.mod.t,xgtest)
  prt<-prt+ predict(x.mod.t,xgtestF)
}


results = data.frame('Apr21_Renat' = pr)
results1 = data.frame('Apr21_Renat' = prt/num_fold)

name_tr<-paste('/home/ec2-user/moredata/al_data1804/train_Apr21_', FEATSNAME, '_',num_fold,'.csv',sep="")

write.table(results, file = name_tr, row.names = F, col.names = T, sep = ",", quote = F)

name_tst<-paste('/home/ec2-user/moredata/al_data1804/test_Apr21_', FEATSNAME, '_',num_fold,'.csv',sep="")

write.table(results1, file = name_tst , row.names = F, col.names = T, sep = ",", quote = F)



