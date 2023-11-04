library(readr)
install.packages("ggplot2")
library(ggplot2)
install.packages("plotly")
library(plotly)
install.packages("dplyr")
library(dplyr)
install.packages("tidyverse")
library(tidyverse)
library(ggcorrplot) # finding the correlation with variables 
install.packages("caTools")
library(caTools)# splitting data into training set test set 
install.packages("caret")
library(caret)
install.packages("cvms")
library(cvms)
install.packages("e1071")
library(e1071)
install.packages("class")
library(class)
install.packages("utils")
library(utils)
library(tidyr)
install.packages("ggcorrplot")
library(ggcorrplot)


cancer<-read.csv("C:/Users/Admin/Downloads/breast.cancer.dataset.csv")
head(cancer)

str(cancer)
canc_num <- cancer %>%
  as.data.frame() %>%
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value")

print(typeof(canc_num))

ggplot(canc_num, aes(value)) +
  geom_density() +
  facet_wrap(~variable)

r <- cor(cancer[, 3:32])
round(r, 2)
ggcorrplot(r)

cancer$diagnosis<-factor(cancer$diagnosis,levels=c("M","B"),labels=c(0,1))
cancer$diagnosis <- as.character(cancer$diagnosis)
cancer$diagnosis <- as.numeric(cancer$diagnosis)
str(cancer)
cancer <- cancer %>% relocate(diagnosis,.after= fractal_dimension_worst)#relocating the diagnosis to the end for no confusion on the target variable

install.packages("naniar")
library(naniar)
data.cancer<-cancer[,1:32]#sampling the data for analysis
vis_miss(data.cancer)#shows the presence of missing values
sum(is.na(data.cancer))#0 shows there r no missing values
sapply(data.cancer,function(x)sum(is.na(x)))

#splitting of the data for the model
split<-sample.split(cancer$diagnosis,SplitRatio = 0.7)
train<-cancer[split,]
test<-cancer[!split,]

#feature scaling for predictive modelling
train[,2:5]<-scale(train[,2:5])
test[,2:5]<-scale(test[,2:5])

train[,14:15]<-scale(train[,14:15])
test[,14:15]<-scale(test[,14:15])

train[,22:25]<-scale(train[,22:25])
test[,22:25]<-scale(test[,22:25])

view(train)
view(test)
str(train)
str(test)

train <- train[, -which(names(train) == "X")]
test<-test[,-which(names(test)=="X")]


columns_with_na <- colSums(is.na(train))
print(columns_with_na)

library(e1071)

regressor_svm <- svm(formula = diagnosis ~ ., 
                     data=train,
                     type = 'C-classification',
                     kernel = 'linear')


#SVM IMPLEMENTATION
regressor_svm <- svm(formula = diagnosis ~ ., 
                     data=train,
                     type = 'C-classification',
                     kernel = 'linear')

y_pred1 = predict(regressor_svm, newdata = test[-32])
true_labs<-as.numeric(test$diagnosis)
cm = table(test[ , 32], y_pred1)
cm

confmat<-confusion_matrix(y_pred1,true_labs)
plot_confusion_matrix(confmat)




#KNN IMPLEMENTATION
y_predknn = knn(train = train[, 2:31],
                test = test[, 2:31],
                cl = train[, 32],
                k = 5,
                prob = TRUE)

true_lab<-as.numeric(test$diagnosis)

cmknn = table(test[, 32], y_predknn)
cmknn

confmaps<-confusion_matrix(y_predknn,true_lab)
plot_confusion_matrix(confmaps)

install.packages("gbm")
library(gbm)

set.seed(1)
gbm_model <- train(diagnosis ~ ., train, method="gbm", verbose=FALSE)

