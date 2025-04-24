# Project-Machine-Learning

The goal of your project is to predict how people perform physical exercises. This is represented by the variable classe in the training dataset. You may use the other variables to make your predictions. You are required to write a report explaining how you built your model, how you applied cross-validation, your thoughts on the expected out-of-sample error, and the reasons behind the choices you made. You will also use your predictive model to make predictions on 20 different test cases.

# Load libraries
```{r}
library(caret)
library(randomForest)
library(dplyr)
```

## 1. Load Data
```{r}
train_data <- read.csv("/cloud/project/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
test_data <- read.csv("/cloud/project/pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))
```

## Explore Data
```{r}
str(train_data)
# check for missing values
missing_summary <- sapply(train_data, function(x) sum(is.na(x)))
missing_summary[missing_summary > 0]
```

## 2. Cleaning Data
```{r}
# Hapus kolom yang tidak berguna (ID, timestamps, dll)
train_data_clean <- train_data[, -c(1:7)]

# Hapus kolom yang memiliki banyak NA
train_data_clean <- train_data_clean[, colSums(is.na(train_data_clean)) == 0]

# Samakan kolom pada testing
test_data_clean <- test_data[, names(train_data_clean)[-ncol(train_data_clean)]]

# Hapus kolom near-zero variance
nzv <- nearZeroVar(train_data_clean, saveMetrics = TRUE)
nzv_cols <- rownames(nzv)[nzv$nzv == TRUE]
train_data_clean <- train_data_clean[, !names(train_data_clean) %in% nzv_cols]

# Pastikan 'classe' adalah faktor
train_data_clean$classe <- as.factor(train_data_clean$classe)

# Struktur akhir data pelatihan setelah cleaning
str(train_data_clean)
```

# 3. Split training menjadi training dan validation
```{r}
set.seed(123)
inTrain <- createDataPartition(train_data_clean$classe, p = 0.7, list = FALSE)
trainSet <- train_data_clean[inTrain, ]
validSet <- train_data_clean[-inTrain, ]
```

# 4. Training model Random Forest
```{r}
set.seed(123)
fitControl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
model_rf <- train(classe ~ ., data = train_data_clean, method = "rf", trControl = fitControl, ntree = 100)

# Menampilkan informasi tentang model
model_rf
```

# 5. Evaluasi model
```{r}
pred_valid <- predict(model_rf, validSet)
confusionMatrix(pred_valid, validSet$classe)
```

# 6. Prediksi pada data testing
```{r}
final_predictions <- predict(model_rf, test_data_clean)
```

# 7. Print hasil prediksi
```{r}
print(final_predictions)
```

# 8. Simpan hasil ke file
```{r}
write.table(final_predictions, file = "final_predictions.txt", row.names = FALSE, quote = FALSE)
```

# Conclusion
The Random Forest model achieved high accuracy in predicting the classe variable. This method was chosen for its robustness, resistance to overfitting, and suitability for high-dimensional data. Cross-validation showed that the model generalizes well to unseen data.
