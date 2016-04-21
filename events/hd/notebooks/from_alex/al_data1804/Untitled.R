





#renat_train <- train[:, grepl("ren_", names(train))]
#renat_test <- test[:, grepl("ren_", names(test))]



train <- read_csv('/home/ec2-user/moredata/al_data1804/train_al_data1804.csv')
test <- read_csv('/home/ec2-user/moredata/al_data1804/test_al_data1804.csv')

product_id_train <- train[,1]
product_id_test <- test[,1]

renat_train <- cbind(product_id_train, train[, grepl("ren_", names(train))])
renat_test <- cbind(product_id_test, test[, grepl("ren_", names(test))])

write.table(renat_train, file = 'renat_train.csv' , row.names = F, col.names = T, sep = ",", quote = F)
write.table(renat_test, file = 'renat_test.csv' , row.names = F, col.names = T, sep = ",", quote = F)
