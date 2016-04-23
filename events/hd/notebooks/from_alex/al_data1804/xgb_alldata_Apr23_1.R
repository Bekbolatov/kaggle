library(readr)
library(xgboost)
############################################################
orig_label<-read_csv('/home/ec2-user/data/hd/unpacked/train.csv')$relevance
train_all<-cbind(read_csv('~/moredata/al_data1804/train_al_data1804.csv'), read_csv('~/moredata/al_data1804/X_train54spel.csv'))
test_all<-cbind(read_csv('~/moredata/al_data1804/test_al_data1804.csv'), read_csv('~/moredata/al_data1804/X_test54spel.csv'))
############################################################
htr<-read.csv('/home/ec2-user/moredata/al_data1804/train_rows.csv')[,1]+1
train<-train_all[htr,]
y<-orig_label[htr]

h<-read.csv('/home/ec2-user/moredata/al_data1804/val_rows.csv')[,1]+1
test<-train_all[h,]
y_t<-orig_label[h]

dtrain<-xgb.DMatrix(data=data.matrix(train),label=y,missing=NA)
dval<-xgb.DMatrix(data=data.matrix(test),label=y_t,missing=NA)
watchlist<-list(val=dval,train=dtrain)
############################################################
# gblinear "gbtree",
param <- list(  objective           = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                eta                 = 0.05, #0.01, 
                #min_child_weight    =3,    
                max_depth           = 7, 
                subsample           = 0.7,
                colsample_bytree    = 1.0                
)

set.seed(123)
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 3500, 
                    early.stop.round    = 50,
                    watchlist           = watchlist,
                    print.every.n       = 20
                    
)

############################################################

######### model
param <- list(  objective           = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                eta                 = 0.01, 
                #min_child_weight =2,    
                max_depth           = 7, 
                subsample           = 0.7,
                colsample_bytree    = 1.0                
)
dtrain_all <-xgb.DMatrix(data=data.matrix(train_all), label=orig_label, missing=NA)
dtest_all <-xgb.DMatrix(data=data.matrix(test_all), missing=NA)

set.seed(123)
clf <- xgb.train(   params              = param, 
                    data                = dtrain_all, 
                    nrounds             = 2122 
)

test_relevance <- predict(clf, dtest_all)
test_relevance <- ifelse(test_relevance>3,3,test_relevance)
test_relevance <- ifelse(test_relevance<1,1,test_relevance)


submission <- read_csv("/home/ec2-user/data/hd/unpacked/sample_submission.csv") 
submission[,2]<-test_relevance 
write_csv(submission,"~/moredata/al_data1804/xgb_alldata_Apr23_1.csv")

































> toRemove
[1] "lv_desc" "dl_desc" "ren_129" "ren_131" "ren_135" "ren_133" "ren_220"
