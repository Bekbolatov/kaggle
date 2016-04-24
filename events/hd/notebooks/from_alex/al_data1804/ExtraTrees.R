#====
#install.packages('rJava')
#install.packages('extraTrees')
#====
#detach("package:extraTrees", unload=TRUE)
#detach("package:rJava", unload=TRUE)
#options( java.parameters = "-Xmx35g" )
#library(extraTrees)
#====

library(readr)
library(xgboost)
options( java.parameters = "-Xmx35g" )
library(extraTrees)
runname <- "forstack_Apr24_et"
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
############################################################
rmse <- function(dact, dpred)
{
  sqrt(mean((dact - dpred)^2))
}
############################################################


et <- extraTrees(train, y, ntree=1, numRandomCuts=1, numThreads=32)
yhat <- predict(et, test)
rmse(yhat, y_t)





rmse <- function(dact, dpred)
{
  sqrt(mean((dact - dpred)^2))
}
NCOL <- 3183
lasttime <- Sys.time()
for (ntree in c(200)) {
  for (numRandomCuts in c(4)) {
    for (mtry in NCOL/c(3.5, 3, 2.5)) {
      for (nodesize in c(4)) {
        et <- extraTrees(train, y, mtry=mtry, ntree=ntree, numRandomCuts=numRandomCuts, numThreads=32)
        yhat <- predict(et, test)
        res <- rmse(yhat, y_t)
        thistime <- Sys.time() - lasttime
        units(thistime) <- 'mins'
        lasttime <- Sys.time()
        cat('{ "elapsed":', thistime,', "ntree":', ntree,', "numRandomCuts":',numRandomCuts, ', "mtry":',mtry, ', "nodesize":', nodesize,  ', "res":', res,'},\n')
      }
    }
  }
}






rmse <- function(dact, dpred)
{
  sqrt(mean((dact - dpred)^2))
}
rmse(yhat, y_t)




littlesample = c(1, 2, 3, 4, 5, 6, 7, 8, 9)
smalltrain = train[littlesample,]
smally = y[littlesample]
smallet <- extraTrees(smalltrain, smally)
smallyhat <- predict(smallet, test)



sqrt( sum( (df$model - df$measure)^2 , na.rm = TRUE ) / nrow(df) )
