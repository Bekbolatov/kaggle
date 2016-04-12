#############################
# Fixed validation split 
#############################

The fixed validation split is implemented by Zach, based on forum post: 
https://www.kaggle.com/remap1/home-depot-product-search-relevance/get-train-validation-indices


To obtain the split

1) load the train set from one of Zach's feature sets 
2) the column "set" on the train data contain's 2 values, "train" and "test"
3) Use "test" records as the validation set which all members in "Slippery Appraisals" are using 

