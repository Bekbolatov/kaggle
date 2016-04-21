library(readr)
library(xgboost)
######### 
cat("Reading data\n")
train <- read_csv('C:\\HDP\\Data\\train.csv')
y0<-train$relevance
rm(train)
train <- read_csv('C:\\HDP\\Data\\train_al_data1804.csv')
train2 <- read_csv('C:\\HDP\\Data\\X_train54spel.csv')
train<-cbind(train,train2)
rm(train2)
######
set.seed(123)
ind_v<-read.csv('C:\\HDP\\Data\\valid_ind.csv')
ind_tr<-read.csv('C:\\HDP\\Data\\train_ind.csv')
h<-ind_v[,2]+1
test<-train[h,]
y_t<-y0[h]
htr<-ind_tr[,2]+1
train<-train[htr,]
y<-y0[htr]

dtrain<-xgb.DMatrix(data=data.matrix(train),label=y,missing=NA)
dval<-xgb.DMatrix(data=data.matrix(test),label=y_t,missing=NA)
watchlist<-list(val=dval,train=dtrain)
# gblinear "gbtree",
param <- list(  objective           = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                eta                 = 0.01, 
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
                    watchlist           = watchlist

)


## new split
#[1500]  val-rmse:0.450350       train-rmse:0.300599 8 0.7 1 eu_al_stas -dubl
#[1871]  val-rmse:0.450423       train-rmse:0.283739 8 0.7 0.9 eu_al_stas -dubl
#[1870]  val-rmse:0.450511       train-rmse:0.290999 8 0.7 1 0.009 eu_al_stas -dubl
#[1711]  val-rmse:0.450173       train-rmse:0.290780 8 0.7 1 eu_al_stas -dubl+uid
#[1524]  val-rmse:0.448014       train-rmse:0.297824 alldata_spel_stas
#[1498]  val-rmse:0.447927       train-rmse:0.299359 alldata_spel_stas_mik
#[1522]  val-rmse:0.447541       train-rmse:0.296994 alldata_stas_mik_tfidf(25)
#[1569]  val-rmse:0.447458       train-rmse:0.292387 alldata_stas_mik_tfidf(50te)
#[1337]  val-rmse:0.447930       train-rmse:0.299212 alldata_stas_mik_tfidf(200te)
#[1352]  val-rmse:0.447452       train-rmse:0.305437 alldata_stas_mik_tfidf(50te)
#[1514]  val-rmse:0.447447       train-rmse:0.295829
#[1406]  val-rmse:0.448202       train-rmse:0.304207 alldata_spel_stas cosdist
#[1417]  val-rmse:0.448018       train-rmse:0.303504 alldata_spel_stas stack
#[1910]  val-rmse:0.447274       train-rmse:0.274886 alldata_spel_stas x5
#[1554]  val-rmse:0.446951       train-rmse:0.292064 alldata
#[1487]  val-rmse:0.444176       train-rmse:0.288388 alldata2203_Zach
#[1452]  val-rmse:0.443922*      train-rmse:0.290717 alldata2203_ZachDst
#[1369]  val-rmse:0.443702       train-rmse:0.295075 data_eu3
#[735]   val-rmse:0.441324       train-rmse:0.330978 data_eu3+R
#[824]   val-rmse:0.442777       train-rmse:0.324524 data_eu3+R -4
#[1424]  val-rmse:0.443068       train-rmse:0.290548 data_eu3+R -6
#[1459]  val-rmse:0.443724       train-rmse:0.287218 data_eu3+P
#[1681]  val-rmse:0.443128*      train-rmse:0.274206 alldataRenPaw
#[1500]  val-rmse:0.443004*      train-rmse:0.283924 alldatRenPaw-dubl-6
#[1308]  val-rmse:0.442848*      train-rmse:0.293244 eu3alrenpaw-6-dubl
#[1301]  val-rmse:0.442757*      train-rmse:0.292598 eu3alrenpaw-6-dublw2v
#[1308]  val-rmse:0.442580       train-rmse:0.292465 eu3alrenpaw-6-dublw2vN
#[1302]  val-rmse:0.442464       train-rmse:0.292813 eu3(cut[1:1360])alrenpaw-6-dublw2vN
#[1470]  val-rmse:0.442294       train-rmse:0.284334 eu3(cut[1:1360,2285:2319])alrenpaw-6-dublw2vN
#[1127]  val-rmse:0.439856       train-rmse:0.299004 ezarpNew 8
#[1406]  val-rmse:0.439768*      train-rmse:0.325914 ezarpNew 7
#[2141]  val-rmse:0.439938       train-rmse:0.339399 ezarpNew 6
### 7 0.7
#[1517]  val-rmse:0.439175       train-rmse:0.320359 ezaNewrpNew 




# model
library(readr)
library(xgboost)
cat("Reading data\n")
train <- read_csv('C:\\HDP\\Data\\train.csv')
y<-train$relevance
train <- read_csv('C:\\HDP\\Data\\train_al_data1804.csv')
train2 <- read_csv('C:\\HDP\\Data\\X_train54spel.csv')
train<-cbind(train,train2)
rm(train2)

test <- read_csv('C:\\HDP\\Data\\test_al_data1804.csv')
test2 <- read_csv('C:\\HDP\\Data\\X_test54spel.csv')
test<-cbind(test,test2)
rm(test2)



param <- list(  objective           = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                eta                 = 0.01, 
                #min_child_weight =2,    
                max_depth           = 7, 
                subsample           = 0.7,
                colsample_bytree    = 1.0                
)

set.seed(123)

dtrain<-xgb.DMatrix(data=data.matrix(train),label=y,missing=NA)
dtest<-xgb.DMatrix(data=data.matrix(test),missing=NA)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 2122 
)

test_relevance <- predict(clf,dtest)
test_relevance <- ifelse(test_relevance>3,3,test_relevance)
test_relevance <- ifelse(test_relevance<1,1,test_relevance)


submission <- read_csv("C:\\HDP\\Sub\\sample_submission.csv") 
submission[,2]<-test_relevance 
write_csv(submission,"C:\\HDP\\Sub\\xgb_alldata1804.csv")
#0.44364 0.44311


> 1128/0.715
[1] 1577.622

































> toRemove
[1] "lv_desc" "dl_desc" "ren_129" "ren_131" "ren_135" "ren_133" "ren_220"
