# Machine Learning Project Plan Implementation

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Results](#results)

## Introduction
This project involves implementing a machine learning project plan using the "adult" dataset, which contains Census information from 1994. The project includes steps for data loading, preprocessing, model training, evaluation, and hyperparameter tuning.

## Data Preprocessing
Data preprocessing is a crucial step to ensure the quality of the data before training the model. The following steps are performed:

1. **Handling Missing Values:**
    - Missing values in numerical columns (`age` and `hours-per-week`) are filled with the median of the respective columns.
    - Missing values in categorical columns (`workclass`, `occupation`, `native-country`) are filled with the mode of the respective columns.

2. **Handling Outliers:**
    - Outliers in numerical columns are detected using the Z-score method.
    - Outliers are then handled by winsorizing the data, which limits extreme values to reduce their impact.

3. **Balancing Classes:**
    - For categorical variables such as `race` and `sex_selfID`, class imbalance is addressed by upsampling the minority classes to match the size of the majority class. This helps in creating a balanced dataset, promoting fair AI.

4. **One-Hot Encoding:**
    - Categorical variables are converted into numerical format using one-hot encoding. This step creates binary columns for each category, allowing the machine learning model to process categorical data effectively.

## Model Training and Evaluation
The processed data is split into training and testing sets to evaluate the model's performance on unseen data.

1. **Data Splitting:**
    - The dataset is split into training (80%) and testing (20%) sets.

2. **Training the Initial Model:**
    - A Random Forest classifier is used as the initial model. Random Forest is an ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

3. **Evaluating the Initial Model:**
    - The model is evaluated using the testing set, and the accuracy is computed to understand the initial performance of the model.

4. **Feature Importance:**
    - The importance of each feature is calculated using the trained Random Forest model. This helps in understanding which features contribute the most to the prediction.

## Hyperparameter Tuning
To improve the model's performance, hyperparameter tuning is performed using GridSearchCV.

1. **GridSearchCV:**
    - A grid search is performed over a set of hyperparameters (`n_estimators` and `max_depth`) to find the best combination that maximizes the model's performance.
    - GridSearchCV performs cross-validation to evaluate the model for each combination of hyperparameters.

## Model Evaluation
The final model is evaluated using various metrics to ensure its performance.

1. **Final Model Accuracy:**
    - The accuracy of the final model (after hyperparameter tuning) is calculated using the testing set.

2. **ROC Curve:**
    - The Receiver Operating Characteristic (ROC) curve is plotted to visualize the performance of the model. The ROC curve shows the trade-off between the true positive rate and false positive rate.
    - The Area Under the ROC Curve (AUC) is calculated to quantify the overall performance of the model.

## Results
The results of the project include:

1. **Initial Model Accuracy:**
    - The accuracy of the initial Random Forest model before hyperparameter tuning.

2. **Final Model Accuracy:**
    - The accuracy of the final model after hyperparameter tuning.

3. **Feature Importance:**
    - A list of the top 5 features that contribute the most to the prediction.

4. **ROC Curve:**
    - A plot of the ROC curve with the AUC value.

The final model and its evaluation metrics provide insights into the model's performance and the importance of different features in predicting the target variable.

