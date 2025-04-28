# Project-Machine-Learning

The goal of your project is to predict how people perform physical exercises. This is represented by the variable classe in the training dataset. You may use the other variables to make your predictions. You are required to write a report explaining how you built your model, how you applied cross-validation, your thoughts on the expected out-of-sample error, and the reasons behind the choices you made. You will also use your predictive model to make predictions on 20 different test cases.

# Load libraries
Before performing any data manipulation or modeling, we need to load the necessary libraries for data processing and model building.
```{r}
library(caret)
library(randomForest)
library(dplyr)
```

## 1. Load Data
We start by reading the training and testing datasets, making sure that invalid entries such as "NA", empty strings, and "#DIV/0!" are treated as missing values.
```{r}
train_data <- read.csv("/cloud/project/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
test_data <- read.csv("/cloud/project/pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))
```

## Explore Data
Before cleaning, we explore the structure of the data and check for missing values to understand which columns might need to be removed or handled.
```{r}
# Explore Data
str(train_data)

# check for missing values
missing_summary <- sapply(train_data, function(x) sum(is.na(x)))
missing_summary[missing_summary > 0]
```

## 2. Cleaning Data
To prepare the data for training, we remove irrelevant columns (like IDs and timestamps), eliminate columns with many missing values, ensure feature alignment between training and test sets, and remove near-zero variance predictors. We also convert the target variable classe into a factor.
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
Now, we train a Random Forest model using cross-validation. This technique helps improve model generalization and reduces the risk of overfitting by training the model on different subsets of the data.
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
The Random Forest model achieved high accuracy in predicting the classe variable. This method was chosen for its robustness, resistance to overfitting, and suitability for high-dimensional data. Cross-validation showed that the model generalizes well to unseen data.
