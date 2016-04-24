library(readr)
library(xgboost)
runname <- "forstack_Apr23_xgb02"
XGB_SEED <- 247
NROUNDS <- 1761
############################################################
set.seed(123)
############################################################
orig_label<-read_csv('/home/ec2-user/data/hd/unpacked/train.csv')$relevance
train_all<-cbind(read_csv('~/moredata/al_data1804/train_al_data1804.csv'), read_csv('~/moredata/al_data1804/X_train54spel.csv'))
test_all<-cbind(read_csv('~/moredata/al_data1804/test_al_data1804.csv'), read_csv('~/moredata/al_data1804/X_test54spel.csv'))
############################################################
# train split [for validation]
htr<-read.csv('/home/ec2-user/moredata/al_data1804/train_rows.csv')[,1]+1
train<-train_all[htr,]
y<-orig_label[htr]
# test split [for validation]
h<-read.csv('/home/ec2-user/moredata/al_data1804/val_rows.csv')[,1]+1
test<-train_all[h,]
y_t<-orig_label[h]
# for local validation
dtrain<-xgb.DMatrix(data=data.matrix(train),label=y,missing=NA)
dval<-xgb.DMatrix(data=data.matrix(test),label=y_t,missing=NA)
# for final model
dtrain_all <-xgb.DMatrix(data=data.matrix(train_all), label=orig_label, missing=NA)
dtest_all <-xgb.DMatrix(data=data.matrix(test_all), missing=NA)
############################################################
############################################################
param <- list(  objective = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                max_depth           = 7, 
                eta                 = 0.01,
                colsample_bytree    = 0.45,                
                subsample           = 0.9,
                min_child_weight    = 3    
)
############################################################
############################################################

############################################################
############################################################
# local validation
cat(paste(Sys.time()), "Starting local validation", "\n")

NROUNDS <- 10000

set.seed(123)
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = NROUNDS, 
                    early.stop.round    = 50,
                    watchlist           = list(val=dval, train=dtrain),
                    print.every.n       = 20)

cat(paste(Sys.time()), "Finished local validation", "\n")
############################################################
############################################################
# final model
cat(paste(Sys.time()), "Starting generating submission", "\n")

NROUNDS <- clf$bestInd

set.seed(123)
clf <- xgb.train(   params              = param, 
                    data                = dtrain_all, 
                    nrounds             = NROUNDS,
                    watchlist           = list(val=dval),
                    print.every.n       = 50)

test_relevance <- predict(clf, dtest_all)
test_relevance <- ifelse(test_relevance>3,3,test_relevance)
test_relevance <- ifelse(test_relevance<1,1,test_relevance)

submission <- read_csv("~/moredata/al_data1804/sample_submission.csv") 
submission[,2]<-test_relevance 

submission_filename <-paste('~/moredata/al_data1804/xgb_submission_', runname, '.csv',sep="")
write_csv(submission, submission_filename)

cat(paste(Sys.time()), "Finished generating submission", "\n")
############################################################
############################################################
# generate for stacking
cat(paste(Sys.time()), "Starting stacking", "\n")

num_fold<-10
set.seed(123)
k<-sample(c(1:num_fold),nrow(train_all),replace=TRUE)
indxFile<-data.frame(k = k,TestIndex=c(1:nrow(train_all)))
indxK <- unique(indxFile[,"k"])

prt <- 0
pr <- orig_label
fono <- 1

for (K in indxK) {
  cat(paste(Sys.time()), "Starting fold #: ", fono, "\n")
  fono <- fono + 1
  currentIdx <- indxFile[indxFile[,"k"] == K,"TestIndex"]
  xtrain_B <-  train_all[-currentIdx,]
  xtrain_S <-  train_all[currentIdx,]
  #modeling
  xgtrain = xgb.DMatrix(as.matrix(xtrain_B), label = orig_label[-currentIdx], missing = NA)
  xgtest = xgb.DMatrix(as.matrix(xtrain_S), missing = NA)
  
  set.seed(XGB_SEED)
  x.mod.t  <- xgb.train(params=param, data=xgtrain, nrounds=NROUNDS, watchlist=list(val=dval), print.every.n=200)
  
  pr[currentIdx]<- predict(x.mod.t, xgtest)
  prt<-prt+ predict(x.mod.t, dtest_all)
}

results <- data.frame("alldata" = pr)
colnames(results) <- list(paste('alldata_', runname, sep=""))
write.table(results, 
            file = paste('~/moredata/al_data1804/train_', runname, '_',num_fold,'.csv',sep=""), 
            row.names = F, col.names = T, sep = ",", quote = F)

results <- data.frame("alldata" = prt/num_fold)
colnames(results) <- list(paste('alldata_', runname, sep=""))
write.table(results, 
            file = paste('~/moredata/al_data1804/test_', runname, '_', num_fold, '.csv', sep="") , 
            row.names = F, col.names = T, sep = ",", quote = F)

cat(paste(Sys.time()), "Finished stacking", "\n")
############################################################
############################################################


  




