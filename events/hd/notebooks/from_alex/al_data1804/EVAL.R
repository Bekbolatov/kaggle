library(readr)
library(xgboost)
######### 
cat("Reading data\n")
train <- read_csv('/home/ec2-user/data/hd/unpacked/train.csv')
y0<-train$relevance
rm(train)

train <- read_csv('/home/ec2-user/moredata/al_data1804/train_al_data1804.csv')

######
set.seed(123)
ind_v<-read.csv('/home/ec2-user/moredata/al_data1804/val_rows.csv')
ind_tr<-read.csv('/home/ec2-user/moredata/al_data1804/train_rows.csv')
h<-ind_v[,1]+1
test<-train[h,]
y_t<-y0[h]
htr<-ind_tr[,1]+1
train<-train[htr,]
y<-y0[htr]

dtrain<-xgb.DMatrix(data=data.matrix(train),label=y,missing=NA)
dval<-xgb.DMatrix(data=data.matrix(test),label=y_t,missing=NA)
watchlist<-list(val=dval,train=dtrain)
# gblinear "gbtree",
param <- list(  objective           = "reg:linear",   
                booster = "gbtree",
                eval_metric = "rmse",
                eta                 = 0.005, 
                #min_child_weight    =3,    
                max_depth           = 7, 
                subsample           = 0.9,
                colsample_bytree    = 0.8                
)
# My best model so far is xgboost with 5000 rounds, eta 0.005, maxdepth 7, subsample 0.9 and colsample by tree 0.8. 
# This model scored 0.45446 or so. I was not successful with any ensembling to improve th
set.seed(123)
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 8500, 
                    early.stop.round    = 50,
                    watchlist           = watchlist,
                    print.every.n       = 20
                    
)
