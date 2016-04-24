library(readr)
library(xgboost)
options( java.parameters = "-Xmx35g" )
library(extraTrees)
runname <- "forstack_Apr24_et"
XGB_SEED <- 247
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
############################################################
rmse <- function(dact, dpred) { sqrt(mean((dact - dpred)^2)) }
############################################################
runname <- "forstack_Apr24_et01"
XGB_SEED <- 111
############################################################

ntree <-200
numRandomCuts <- 4
NCOL <- 3183
mtry <- NCOL/3
nodesize <- 4

rmse <- function(dact, dpred) { sqrt(mean((dact - dpred)^2)) }
cat(paste(Sys.time()), "Starting stacking", "\n")

num_fold<-10
set.seed(123)
k<-sample(c(1:num_fold),nrow(train_all),replace=TRUE)
indxFile<-data.frame(k = k,TestIndex=c(1:nrow(train_all)))
indxK <- unique(indxFile[,"k"])

prt <- 0
pr <- orig_label
fono <- 1
lasttime <- Sys.time()
for (K in indxK) {
  cat(paste(Sys.time()), "Starting fold #: ", fono, "\n")
  fono <- fono + 1
  currentIdx <- indxFile[indxFile[,"k"] == K,"TestIndex"]
  xtrain_B <-  train_all[-currentIdx,]
  xtrain_S <-  train_all[currentIdx,]
  #modeling
  set.seed(XGB_SEED)
  x.mod.t <- extraTrees(xtrain_B, orig_label[-currentIdx], mtry=mtry, ntree=ntree, numRandomCuts=numRandomCuts, numThreads=32)
  yhat_t <- predict(x.mod.t, xtrain_S)
  err_t <- rmse(yhat_t, orig_label[currentIdx])
  cat("rmse: ", err_t, '\n')
  
  pr[currentIdx] <- yhat_t
  prt<-prt+ predict(x.mod.t, test_all)
  
  thistime <- Sys.time() - lasttime
  units(thistime) <- 'mins'
  lasttime <- Sys.time()
  cat('{ "elapsed":', thistime,', "fold":', fono,'},\n')
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
