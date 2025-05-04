# Project Machine Learning

# Background of the Assignment
By using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible to collect a large amount of data about personal activities at a relatively low cost. These types of devices are part of the quantified self movement—a group of enthusiasts who regularly measure themselves to improve their health, to discover their behavioral patterns, or simply because they are technology fans. One common thing people do is measure how much of a certain activity they perform, but they rarely measure how well they perform it. In this project, your goal is to use data from accelerometers on the belt, forearm, arm, and dumbbell of six participants. They were asked to perform barbell lifts correctly and incorrectly in five different ways. More information is available on the website here:
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har
(see the Weight Lifting Exercise Dataset section).

# Data
The training data for this project is available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The testing data is available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project comes from the following source:
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

# Load libraries
Before performing any data manipulation or modeling, Before performing any data manipulation or modeling, we load the necessary libraries for data cleaning, model training, and evaluation.
```{r}
library(caret)
library(randomForest)
library(dplyr)
```

# 1. Load Data
We start by reading the training and testing datasets, making sure that invalid entries such as "NA", empty strings, and "#DIV/0!" are treated as missing values.
```{r}
train_data <- read.csv("/cloud/project/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
test_data <- read.csv("/cloud/project/pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))
```

# Explore Data
Before cleaning, we explore the structure of the data and check for missing values to understand which columns might need to be removed or handled.
```{r}
# Explore Data
str(train_data)

# check for missing values
missing_summary <- sapply(train_data, function(x) sum(is.na(x)))
missing_summary[missing_summary > 0]
```

# 2. Cleaning Data
## How the model was built?
To build the model, we first clean the data by removing columns that are irrelevant or contain too many missing values, ensuring the test and training data have matching features, and remove near-zero variance predictors that provide little to no useful information. We also convert the target variable classe into a factor.
```{r}
# Remove unnecessary columns (ID, timestamps, etc.)
train_data_clean <- train_data[, -c(1:7)]

# Remove columns with many missing values
train_data_clean <- train_data_clean[, colSums(is.na(train_data_clean)) == 0]

# Match columns in the test set
test_data_clean <- test_data[, names(train_data_clean)[-ncol(train_data_clean)]]

# Remove near-zero variance columns
nzv <- nearZeroVar(train_data_clean, saveMetrics = TRUE)
nzv_cols <- rownames(nzv)[nzv$nzv == TRUE]
train_data_clean <- train_data_clean[, !names(train_data_clean) %in% nzv_cols]

# Ensure 'classe' is a factor
train_data_clean$classe <- as.factor(train_data_clean$classe)

# Check the final structure
str(train_data_clean)
```

# 3. Split Data into Training and Validation Sets
To evaluate the model properly, we split the cleaned dataset into two parts: 70% for training and 30% for validation. This ensures we can assess model performance before applying it to the real test data.
```{r}
# Split training into training and validation
set.seed(123)
inTrain <- createDataPartition(train_data_clean$classe, p = 0.7, list = FALSE)
trainSet <- train_data_clean[inTrain, ]
validSet <- train_data_clean[-inTrain, ]
```

# 4. Train Random Forest Model
Now, we train a Random Forest model using cross-validation. 

## Why Random Forest was chosen?
Random Forest is chosen because it is a powerful ensemble learning algorithm that combines multiple decision trees to improve accuracy and stability. It performs well with high-dimensional data, is robust to noise and overfitting, and requires minimal preprocessing.

## How cross-validation was used?
To estimate the model’s generalization ability and reduce overfitting, we use 5-fold cross-validation. This means the training set is divided into 5 parts, and the model is trained on 4 parts and validated on the remaining part in a rotating manner.
```{r}
# Train Random Forest model
set.seed(123)
fitControl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
model_rf <- train(classe ~ ., data = train_data_clean, method = "rf", trControl = fitControl, ntree = 100)

# Display model information
model_rf
```

# 5. Evaluate Model
After training, we predict the classes for the validation set and use a confusion matrix to evaluate the accuracy and error rates of the model.
```{r}
# Evaluate model
pred_valid <- predict(model_rf, validSet)
confusionMatrix(pred_valid, validSet$classe)
cm <- confusionMatrix(pred_valid, validSet$classe)
cm$overall['Accuracy']
```

# 6. Predict on Test Data
Once the model is validated, we use it to make predictions on the final test dataset, which does not contain the target variable.
```{r}
# Make predictions on the test data
final_predictions <- predict(model_rf, test_data_clean)
```

# 7. Print Predictions
We print the predictions to visually confirm the output before saving.
```{r}
# Print final predictions
print(final_predictions)
```

# 8. Save Predictions to File
Finally, we save the predicted results into a text file without row names or quotation marks, ready for submission or further analysis.
```{r}
# Save final predictions
write.table(final_predictions, file = "final_predictions.txt", row.names = FALSE, quote = FALSE)
```

# Conclusion
The Random Forest model achieved 100% accuracy in predicting the classe variable on the validation set. The model showed perfect classification, with a 95% confidence interval of (99.94%, 100%). 

This method was chosen for its robustness, resistance to overfitting, and suitability for high-dimensional data. Cross-validation confirmed that the model generalizes well to unseen data, and the combination of preprocessing steps, such as cleaning, near-zero variance removal, and careful data splitting, ensured a high-quality model.

While the model performs excellently on this dataset, it is important to consider the possibility of overfitting if similar data is used for future predictions. Nonetheless, the performance on the validation set confirms the potential strength of the Random Forest model in this context.
