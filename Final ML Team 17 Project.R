########################### Team 17 Final Project #############################

rm(list=ls())

## Load in our Libraries
library(plyr)
library(rjson)
library(magrittr)
library(lubridate)
library(stringi)
library(doSNOW)
library(tm) # text mining
library(NLP)
library(SnowballC)
library(rpart)
library(tidyverse)
library(caret)

setwd('C:\\Users\\zosok\\MyWork\\Spring 2023\\ML2')
getwd()

set.seed(1)

## Load in Original Train Dataset - 1.13GB
data = read.csv('goodreads_train.csv')
dim(data)

data$rating <- as.factor(data$rating)
print(summary(data$rating))

subset0 = filter(data, rating == 0)[1:500,]
subset1 = filter(data, rating == 1)[1:500,]
subset2 = filter(data, rating == 2)[1:500,]
subset3 = filter(data, rating == 3)[1:500,]
subset4 = filter(data, rating == 4)[1:500,]
subset5 = filter(data, rating == 5)[1:500,]

datasubset = rbind(subset0, subset1, subset2, subset3, subset4, subset5)


## Subset of Train Dataset
dim(datasubset)

summary(datasubset$rating)

## Format the review texts
corpus <- Corpus(VectorSource(datasubset$review_text))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))# find stopwords then remove them
# print(stopwords('english'))
corpus <- tm_map(corpus, stemDocument) # performing stemming on the text corpus

## Transform the terms into a matrix and remove sparse terms
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, sparse = 0.7) # A higher "sparse" parameter value means 
                                            # that less common words will be removed

dtmsparse <- as.data.frame(as.matrix(dtm))

## Split train and test
trainrows <- sample(1:nrow(dtmsparse), nrow(dtmsparse) * 0.8)
testrows <- (-trainrows)

train <- dtmsparse[trainrows,] ## 11 Variables
test <- dtmsparse[testrows,]

train$rating <- datasubset[trainrows,]$rating

## Build our model with rpart
## "rpart" stands for Recursive Partitioning and Regression Trees
## You can specify splitting criteria, maximum tree depth, minimum number of observations in a node
## You can also prune the model if it is overfitting
## rpart is better than tree when handling missing values
model <- rpart(rating ~., data = train, method = 'class')

mypred <- predict(model, newdata = test, type = 'class')

obs <- datasubset[testrows,]$rating

confusionMatrix(obs, mypred)
table(mypred, obs)

## 11 x features
## 30.5% accurate
## Not many effective words being brought into the train data
plot(model)
text(model,pretty=1)

################################################################################
## Let's try raising the sparsity to 0.999
set.seed(1)

dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, sparse = 0.999)

dtmsparse <- as.data.frame(as.matrix(dtm))

## Split train and test
trainrows <- sample(1:nrow(dtmsparse), nrow(dtmsparse) * 0.8)
testrows <- (-trainrows)

train <- dtmsparse[trainrows,]
test <- dtmsparse[testrows,]

train$rating <- datasubset[trainrows,]$rating

## Build our model with rpart
model <- rpart(rating ~., data = train, method = 'class')

mypred <- predict(model, newdata = test, type = 'class')

obs <- datasubset[testrows,]$rating

confusionMatrix(obs, mypred)
table(mypred, obs)

## 4791 x features
## 32.17% accurate
## Too many features, lots of null values
## No 1 predictions - likely due to splitting criteria
plot(model)
text(model,pretty=1)

################################################################################
## Let's split the difference at 0.9
set.seed(1)

dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, sparse = 0.9)

dtmsparse <- as.data.frame(as.matrix(dtm))

## Split train and test
trainrows <- sample(1:nrow(dtmsparse), nrow(dtmsparse) * 0.8)
testrows <- (-trainrows)

train <- dtmsparse[trainrows,]
test <- dtmsparse[testrows,]

train$rating <- datasubset[trainrows,]$rating

## Build our model with rpart
model <- rpart(rating ~., data = train, method = 'class')

mypred <- predict(model, newdata = test, type = 'class')

obs <- datasubset[testrows,]$rating

confusionMatrix(obs, mypred)
table(mypred, obs)

## 128 x features
## 33.17% accurate
## Still not predicting any class 1

################################################################################
## Let's remove certain words from the dataset
set.seed(1)

custom_stopwords = list('book', 'read', 'peopl', 'will', 'year', 'say', 'also', 
                        'get', 'mani', 'way', 'bit', 'just', 'noth', 'seri', 
                        'someth', 'one', 'stori', 'let', 'review', 'come', 
                        'now', 'time', 'tri', 'back', 'realli', 'ive', 'even', 
                        'charact', 'two', 'got', 'person', 'next', 'make', 
                        'everyth', 'spoiler', 'work', 'write', 'happen', 
                        'thing', 'seem', 'author', 'alway', 'look', 'arc', 
                        'complet', 'point', 'that', 'pretti', 'novel', 'anoth', 
                        'everi', 'provid', 'togeth') ## 53 words

train <- dtmsparse[trainrows, -which(colnames(dtmsparse) %in% custom_stopwords)]
test <- dtmsparse[testrows, -which(colnames(dtmsparse) %in% custom_stopwords)]

train$rating <- datasubset[trainrows,]$rating

## Build our model with rpart
model <- rpart(rating ~., data = train, method = 'class')

mypred <- predict(model, newdata = test, type = 'class')

obs <- datasubset[testrows,]$rating

confusionMatrix(obs, mypred)
table(mypred, obs)
## 75 features
## 35.33% Accuracy
plot(model)
text(model,pretty=1)

## Next steps:
      ## best model on kaggle - 67% accuracy - transformers
      ## prune to avoid overfitting
      ## Tweak our sparsity and stopwords
      ## Weigh certain words more heavily
      ## Search for word phrases
      ## Need more negative words in dataset
      ## XGBoost



