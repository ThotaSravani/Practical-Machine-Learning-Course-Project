# Data Preprocessing

library(caret)
library(corrplot)
library(Rtsne)
library(xgboost)
library(rpart)
library(rpart.plot)
library(randomForest)
library(stats)
library(knitr)
library(ggplot2)
knitr::opts_chunk$set(cache=TRUE)

# Load The Data

## URL of training and testing data

trainUrl ="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# File Names
trainFile= "E:/Projects/Practical-Machine-Learning-Course-Project/Practical-Machine-Learning-Course-Project/data/pml-training.csv"
testFile = "E:/Projects/Practical-Machine-Learning-Course-Project/Practical-Machine-Learning-Course-Project/data/pml-testing.csv"

# If directory does not exist, create new
if (!file.exists("./data")) {
  dir.create("./data")
}

# If files does not exist, download the files
if (!file.exists(trainFile)) 
  {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) 
  {
  download.file(testUrl, destfile=testFile, method="curl")
}
# Read the Data

## After downloading the data from the data source,We can read the two csv files into two data frames
train = read.csv("E:/Projects/Practical-Machine-Learning-Course-Project/Practical-Machine-Learning-Course-Project/pml-training.csv")
test = read.csv("E:/Projects/Practical-Machine-Learning-Course-Project/Practical-Machine-Learning-Course-Project/pml-testing.csv")
dim(train)
dim(test)
names(train)

# Clean the Data


#First, extract target outcome (the activity quality) from training data, so now the training data contains only the predictors (the activity monitors).

# Target outcome (label)
outcome.org = train[, "classe"]
outcome = outcome.org 
levels(outcome)

# convert character levels to numeric
num.class = length(levels(outcome))
levels(outcome) = 1:num.class
head(outcome)


# Remove outcome from train
train$classe = NULL

#The assignment rubric asks to use data from accelerometers on the belt, forearm, arm, and dumbell, so the features are extracted based on these keywords.

# Filter columns on: belt, forearm, arm, dumbell
filter = grepl("belt|arm|dumbell", names(train))
train = train[, filter]
test = test[, filter]

# Instead of less-accurate imputation of missing data, remove all columns with NA values.

# Remove columns with NA, use test data as referal for NA
cols.without.na = colSums(is.na(test)) == 0
train = train[, cols.without.na]
test = test[, cols.without.na]

# Preprocessing 

# check for zero variance
zero.var = nearZeroVar(train, saveMetrics=TRUE)
zero.var

# Plot of relationship between features and outcome

featurePlot(train, outcome.org, "strip")

# Plot of correlation matrix


corrplot.mixed(cor(train), lower="circle", upper="color", 

                              tl.pos="lt", diag="n", order="hclust", hclust.method="complete")
# tSNE plot

# A tSNE (t-Distributed Stochastic Neighbor Embedding) visualization is 2D plot of multidimensional features, that is multidimensional reduction into 2D plane. In the tSNE plot below there is no clear separation of clustering of the 5 levels of outcome (A, B, C, D, E). So it hardly gets conclusion for manually building any regression equation from the irregularity.

# t-Distributed Stochastic Neighbor Embedding

tsne = Rtsne(as.matrix(train), check_duplicates=FALSE, pca=TRUE, 
             perplexity=30, theta=0.5, dims=2)

embedding = as.data.frame(tsne$Y)
embedding$Class = outcome.org
s = ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
  geom_point(size=1.25) +
  guides(colour=guide_legend(override.aes=list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("2D Embedding of 'Classe' Outcome") +
  theme_light(base_size=20) +
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank())
print(s)

# Build machine learning model

# XGBoost data

## XGBoost supports only numeric matrix data. Converting all training, testing and outcome data to matrix.

# convert data to matrix
train.matrix = as.matrix(train)
mode(train.matrix) = "numeric"
test.matrix = as.matrix(test)
mode(test.matrix) = "numeric"
# convert outcome from factor to numeric matrix 
#   xgboost takes multi-labels in [0, numOfClass)
y = as.matrix(as.integer(outcome)-1)

# XGBoost parameters


# xgboost parameters
param <- list("objective" = "multi:softprob",    # multiclass classification 
              "num_class" = num.class,    # number of classes 
              "eval_metric" = "merror",    # evaluation metric 
              "nthread" = 8,   # number of threads to be used 
              "max_depth" = 16,    # maximum depth of tree 
              "eta" = 0.3,    # step size shrinkage 
              "gamma" = 0,    # minimum loss reduction 
              "subsample" = 1,    # part of data instances to grow tree 
              "colsample_bytree" = 1,  # subsample ratio of columns when constructing each tree 
              "min_child_weight" = 12  # minimum sum of instance weight needed in a child 
)


# 4-fold cross validation

# set random seed, for reproducibility 
set.seed(1234)
# k-fold cross validation, with timing
nround.cv = 200
system.time( bst.cv <- xgb.cv(param=param, data=train.matrix, label=y, 
                              nfold=4, nrounds=nround.cv, prediction=TRUE, verbose=FALSE) )

tail(bst.cv$dt) 

# Index of minimum merror
min.merror.idx = which.min(bst.cv$dt[, test.merror.mean]) 
min.merror.idx 

# Minimum merror
bst.cv$dt[min.merror.idx,]


# Confusion matrix

# Get CV's prediction decoding
pred.cv = matrix(bst.cv$pred, nrow=length(bst.cv$pred)/num.class, ncol=num.class)
pred.cv = max.col(pred.cv, "last")
# Confusion matrix
confusionMatrix(factor(y+1), factor(pred.cv))

# Model training
# Real model fit training, with full data
system.time( bst <- xgboost(param=param, data=train.matrix, label=y, 
                            nrounds=min.merror.idx, verbose=0) )
# Predicting the testing data

# xgboost predict test data using the trained model
pred <- predict(bst, test.matrix)  
head(pred, 10)  

# Post-processing


# Decode prediction
pred = matrix(pred, nrow=num.class, ncol=length(pred)/num.class)
pred = t(pred)
pred = max.col(pred, "last")
pred.char = toupper(letters[pred])

# Feature Importance

# Get the trained model
model = xgb.dump(bst, with.stats=TRUE)
# Get the feature real names
names = dimnames(train.matrix)[[2]]
# Compute feature importance matrix
importance_matrix = xgb.importance(names, model=bst)

# plot
t = xgb.plot.importance(importance_matrix)
print(t) 

# Creating Submission 

pml.testing <- read.csv("E:/Projects/Practical-Machine-Learning-Course-Project/Practical-Machine-Learning-Course-Project/pml-testing.csv")
answers <- as.character(predict(bst, test.matrix))
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)

