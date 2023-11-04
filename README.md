# Breast-Cancer-Analysis
R Language

# PROBLEM STATEMENT
Breast cancer is the most common cancer among women, and the second leading cause of cancer death in women. Early detection and treatment are essential for improving survival rates.

# HYPOTHESES
Hypotheses are typically formulated before conducting data analysis and modeling to set the direction and expectations for the investigation. In the code, weare  working on a classification problem for breast cancer detection. Here are some example hypotheses you might consider for this project:

# Null Hypothesis (H0):

a.H0: There is no significant relationship between the features in the breast cancer dataset and the diagnosis of breast cancer (Malignant or Benign).

# Alternative Hypotheses (Ha):

a)Ha: Certain features in the dataset are significantly associated with the diagnosis of breast cancer, making them useful for classification.
b)Ha: Machine learning models, such as Support Vector Machine (SVM) and K-Nearest Neighbors (KNN), can effectively classify breast cancer cases based on the provided features.
c)Ha: Feature scaling and preprocessing techniques can improve the accuracy of breast cancer diagnosis predictions.
d)Ha: The SVM model can outperform the KNN model in terms of classification accuracy for breast cancer detection.
e)Ha: The KNN model can outperform the SVM model in terms of classification accuracy for breast cancer detection.

# DATA SCIENCE LIFE CYCLE
## 1.Discovery:
In this step, you gather the data required for your analysis. Data can come from various sources, including databases, APIs, web scraping, or manual data entry. This is a crucial step in gaining insights into the dataset's characteristics before proceeding with the data science process.
In this phase, we start by loading the "adult" dataset to discover and understand the data's structure and content.


Input: 
cancer<-read.csv("C:/Users/Admin/Downloads/breast.cancer.dataset.csv")
head(cancer)
str(cancer) 
str(cancer)

## 2.Exploratory Data Analysis
The purpose of this EDA is to use summary statistics and visualizations to better understand data, find clues about the tendencies of the data, its quality and to formulate assumptions and the hypothesis of our analysis. For data preprocessing to be successful, it is essential to have an overall picture of your data
In this phase with the help of the dataset and structure we explore the relation between the variables by using the visualization 
1.Visualization of Numeric Variables:
The below part of the code generates density plots for the numeric variables in the dataset. Data visualization is a crucial aspect of data exploration to identify patterns and distributions in the data.
Input :
canc_num<-cancer %>%
  as.data.frame() %>%
  select_if(is.numeric) %>%
  gather(key = "variable", value = "value")
print(typeof(canc_num))
ggplot(canc_num, aes(value)) +
  geom_density() +  facet_wrap(~variable)




## 2.Correlation Analysis:
	Calculating and visualizing the correlation between variables helps in understanding the relationships between different features in the dataset.
Input: r <- cor(cancer[, 3:32])
round(r, 2)
ggcorrplot(r)

Output:



## 3.Data Preparation:
This phase involves data cleaning and transformation to make the dataset suitable for analysis and modelling.

Data Cleaning: 

a)Handling Missing Values:
These below lines of code check for missing values in the dataset and visualize the missing data using the naniar package. Dealing with missing data is an important data cleaning step.

Input:
	install.packages("naniar")
library(naniar)
data.cancer<-cancer[,1:32]
vis_miss(data.cancer)
sum(is.na(data.cancer))
sapply(data.cancer,function(x)sum(is.na(x)))

b)Converting data to numeric values:
In this the "diagnosis" column in the dataset is converted to a factor variable and then into a numeric variable, where "M" and "B" are mapped to 0 and 1, respectively. This makes it suitable for modeling.

Input : 
cancer$diagnosis<-factor(cancer$diagnosis,levels=c("M","B"),labels=c(0,1))
cancer$diagnosis <- as.character(cancer$diagnosis)
cancer$diagnosis <- as.numeric(cancer$diagnosis)
str(cancer)
cancer <- cancer %>% relocate(diagnosis,.after= fractal_dimension_worst)# relocating the diagnosis to the end for no confusion
	

## Data Preprocessing:

a)Data Splitting:
Here, the data is split into training and test sets, which is a common preprocessing step in supervised machine learning.

Input:
split <- sample.split(cancer$diagnosis, SplitRatio = 0.7)
train <- cancer[split, ]
test <- cancer[!split, ]



b)Feature Scaling:
Scaling or standardizing features is a common preprocessing step to ensure that all variables have the same scale. In this below code, columns 2 to 5, 14 to 15, and 22 to 25 are being scaled.

Input: 
train[, 2:5] <- scale(train[, 2:5])
test[, 2:5] <- scale(test[, 2:5])
	
train[, 14:15] <- scale(train[, 14:15])
test[, 14:15] <- scale(test[, 14:15])

train[, 22:25] <- scale(train[, 22:25])
test[, 22:25] <- scale(test[, 22:25])

str(train)
str(test)




c)Removing Unnecessary Columns:
In this code, the "X" columns are removed from both the training and test datasets, indicating that these columns are not needed for modeling.

Input:

train <- train[, -which(names(train) == "X")]
test <- test[, -which(names(test) == "X")]
columns_with_na <- colSums(is.na(train))
print(columns_with_na)





## 4.Model Planning and Building:

The part of the code that is used for model building is where you actually train the machine learning models using your training data.

a)SVM IMPLEMENTATION:
The actual model building occurs when you execute the svm function with your training data. The regressor_svm variable stores the trained Support Vector Machine (SVM) model.

Input:
regressor_svm <- svm(formula = diagnosis ~ ., 
                     data = train,
                     type = 'C-classification',
                     kernel = 'linear')

b)KNN IMPLEMENTATION:
In this section, the k-Nearest Neighbors (KNN) model is built when you call the knn function with the training data. The y_predknn variable stores the predicted labels based on the KNN model.

Input:
y_predknn = knn(train = train[, 2:31],
                test = test[, 2:31],
                cl = train[, 32],
                k = 5,
                prob = TRUE)

# 5.Modelling Evaluating:

a)SVM Model Evaluation:
After making predictions using the trained SVM model, you use the table function to create a confusion matrix (cm) that summarizes the model's performance. This confusion matrix helps evaluate the model's accuracy, precision, recall, and other classification metrics.

Input:
y_pred1 = predict(regressor_svm, newdata = test[-32])
true_labs <- as.numeric(test$diagnosis)
cm = table(test[, 32], y_pred1)

Output:
y_pred1
      0   1
  0 59   5
  1   1 106

b)KNN Model Evaluation:
Similar to the SVM model, you evaluate the KNN model by creating a confusion matrix (cmknn) to assess its performance in terms of correctly and incorrectly classified instances.

Input: 
y_predknn = knn(train = train[, 2:31],
                test = test[, 2:31],
                cl = train[, 32],
                k = 5,
                prob = TRUE)
true_lab <- as.numeric(test$diagnosis)
cmknn = table(test[, 32], y_predknn)

confmat<-confusion_matrix(y_pred1,true_labs)
plot_confusion_matrix(confmat)


# CONCLUSION

In this data science project, we embarked on a comprehensive journey to analyze breast cancer data with the ultimate goal of improving early diagnosis and decision-making. Our project followed a structured data science lifecycle, including data collection, data understanding, data preparation, model planning and building, and model evaluation.

                                   
