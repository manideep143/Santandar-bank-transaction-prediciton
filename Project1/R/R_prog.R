
#Load the path
setwd("C:/manideep/edwisor/Project-1")
getwd()

#Load the data
test_data = read.csv("test.csv", header = T)
train_data = read.csv("train.csv", header = T)

#Required packages
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

lapply(x, require, character.only = TRUE)
library(glmnet)

#---------------------------------------------DATA PRE PROCESSING---------------------------------------------------
#sampling
test_sample = test_data[sample(nrow(test_data), 20000, replace = F), ]
train_sample = train_data[sample(nrow(train_data), 20000, replace = F), ]


# Missing value Analysis for test and train data

missing_val = data.frame(apply(train_sample,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
row.names(missing_val) = NULL
names(missing_val)[1] = "percent"
missing_val$percent = (missing_val$percent/nrow(train_sample)) * 100
missing_val = missing_val[order(-missing_val$percent),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]

missing_val1 = data.frame(apply(test_sample,2,function(x){sum(is.na(x))}))
missing_val1$Columns = row.names(missing_val1)
row.names(missing_val1) = NULL
names(missing_val1)[1] = "percent"
missing_val1$percent = (missing_val1$percent/nrow(test_sample)) * 100
missing_val1 = missing_val1[order(-missing_val1$percent),]
row.names(missing_val1) = NULL
missing_val1 = missing_val1[,c(2,1)]


#Outlier Analysis for both train and test data
cnames = colnames(train_sample[,3:202])
train_sample1 = train_sample


for(i in cnames){
print(i)
val = train_sample[,i][train_sample[,i] %in% boxplot.stats(train_sample[,i])$out]
print(length(val))
train_sample = train_sample[which(!train_sample[,i] %in% val),]
}

#Correlation matrix
tr = cor(train_sample[,unlist(lapply(train_sample, is.numeric))])

#Standardization
for(i in cnames){
        print(i)
        train_sample[,i] = (train_sample[,i] - mean(train_sample[,i]))/
                                     sd(train_sample[,i])
   }
  


#---------------------------------------------------- Algorithms----------------------------------------------------
#Divide data into train and test.

division = function(){
set.seed(123)
train.num = createDataPartition(train_sample[,"target"], p = .80, list = FALSE)
train = train_sample[ train.num,]
test  = train_sample[-train.num,]
}

library(dplyr)

#scaling
X_train <- scale(train[,-(1:2)]) %>% data.frame
X_test <- scale(test[,-1]) %>% data.frame
target <- train$target

library(speedglm)

#Logistic regression
logit_model = speedglm(target ~ ., data = X_train, family = binomial())

summary(logit_model)

logit_Predictions = predict(logit_model, newdata = X_test , type = "response")


logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)

X_test$target = logit_Predictions

#Evaluation
ConfMatrix_RF = table(X_test$target, logit_Predictions)
ConfMatrix_RF

#0 3359    0
#1    0  150

#Metrics
library(pROC)
roc_obj_lr <- roc(X_test$target, logit_Predictions)  #1
auc(roc_obj_lr)

Precision_lr = 150/(150 + 0) #1 tp/(tp + fp)
recall_lr =  150/(150 + 0)   #1 tp/(tp + fn)

# Before running next algorithm run the lines 1 to 80.

# Decision Tree C50
train$target<-as.factor(train$target)
str(train$target)

scale_fun = function(){
target <- train$target
Y_train <- scale(train[,-(1:2)]) %>% data.frame
Y_test <- scale(test[,-(1:2)]) %>% data.frame
Y_test$target = test[,2]
}




C50_model = C5.0(target ~., Y_train, trials = 10, rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules.txt")



#Lets predict for test cases
C50_Predictions = predict(C50_model, Y_test[,-201], type = "class")
write.csv(C50_Predictions, "C50_target.csv", row.names = T)

##Evaluate the performance of classification model
ConfMatrix_C50 = table(Y_test$target, C50_Predictions)
confusionMatrix(ConfMatrix_C50)

#C50_Predictions
#0    1
#0 3145   41
#1  297   27

C50_Predictions = as.numeric(as.character(C50_Predictions))

roc_obj_c50 <- roc(test$target, C50_Predictions)
auc(roc_obj_c50) #0.5352

Precision_c50 = 3145/(3145 + 27) #0.99
recall_c50 =  3145/(3145+41) #0.987

#Random Forest
division()
scale_fun()
RF_model = randomForest(target ~ ., Y_train, importance = TRUE, ntree = 500)
RF_Predictions = predict(RF_model, Y_test[,-201])
write.csv(RF_Predictions, "RF_target.csv", row.names = T)

##Evaluate the performance of classification model
ConfMatrix_RF = table(Y_test$target, RF_Predictions)
confusionMatrix(ConfMatrix_RF)

#RF_Predictions
#0    1
#0 3186   0
#1  324    0

RF_Predictions = as.numeric(as.character(RF_Predictions))

roc_obj_RF <- roc(Y_test$target, RF_Predictions)
auc(roc_obj_RF) #0.5

Precision_RF = 3186/(3186 + 0) #1
recall_RF =  3186/(3186+0) #1

library(e1071)

#Develop model
division()
scale_fun()
NB_model = naiveBayes(target ~ ., data = Y_train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, Y_test[,1:200], type = 'class')
write.csv(NB_Predictions, "NB_target.csv", row.names = T)

#Look at confusion matrix
Conf_matrix = table(observed = Y_test[,201], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)

#predicted
#observed    0    1
#0 3137   49
#1  227  97



NB_Predictions = as.numeric(as.character(NB_Predictions))

roc_obj_nb <- roc(Y_test$target, NB_Predictions)
auc(roc_obj_nb) #0.642

Precision_nb = 3137/(3137 + 97) #0.970
recall_nb =  3137/(3137+49) #0.984

#-------------------------------------Prediction on new data----------------------------------------
#New data
test_data$ID_code = NULL
pred = predict(logit_model, newdata = test_data,type = 'response')
pred = ifelse(pred > 0.5, 1, 0)
test_data$target = pred

write.csv(pred, "final_output_r.csv", row.names = T)

