## R version 3.1.0 (2014-04-10) -- "Spring Dance"
## Copyright (C) 2014 The R Foundation for Statistical Computing
## Platform: x86_64-w64-mingw32/x64 (64-bit)


##Loading Libraries

library(gbm)

##set the seed

set.seed(123)

## Read the train & test file 

train=read.csv("train.csv")
test=read.csv("test.csv")

gbm1= gbm(Hazard~.-Id, 
                    data=train,
                    distribution = "gaussian",
                    n.trees = 9,
                    interaction.depth = 9,
                    n.minobsinnode = 1,
                    shrinkage = 0.2,
                    bag.fraction = 0.9)

pred=predict(gbm1,test[,-1],n.trees=9,type="response")

Submission1=data.frame("Id"=test$Id,"Hazard"=pred)
write.csv(Submission1,"Submission1.csv",row.names=FALSE,quote=FALSE)

