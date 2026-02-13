# 2025AA05169  | RAJAMANI S 
13-Feb-2026 | ML Assignment 2

## a. Problem Statement

As a part of this assignment, I am trying to predict the current survival status of the patients who were diagnosed with breast cancer. This is a research problem which tries to analyze the survival status of the patients, based on the patient test data given in the dataset.

## b. Dataset Description

This dataset "Breast_Cancer.csv" is downloaded from Kaggle. [Link: https://www.kaggle.com/datasets/reihanenamdari/breast-cancer/data] The dataset has a total of 15 features, out of which 10 features are categorical and required one-hot encoding. I have removed few features from the dataset (like patient name, etc.) which were not relevant to the problem. The output label is also one-hot encoded as the XGBoost model failed to parse the non-encoded categorical data.

## c. Models Used

I have used the following models for the comparison and analysis:
* Simple Logistic Regression
* Decision Tree Model
* K-Nearest Neighbors Model
* Naive Bayes Classifier
* Random Forest
* XGBoost Model

I have given the analysis of the metrics obtained after the training and evaluation of each of these models

### Metrics Comparison

| Model | Accuracy | Precision | Recall | F1 Score | AUC Score | MCC Score |
|-------|----------|-----------|--------|----------|-----------|-----------|
| Simple Logistic Regression | 0.8940 | 0.8859 | 0.8940 | 0.8796 | 0.8652 | 0.5228 |
| Decision Trees | 0.8212 | 0.8228 | 0.8212 | 0.8220 | 0.6582 | 0.3137 |
| K-Nearest Neighbors | 0.8411 | 0.7970 | 0.8411 | 0.8052 | 0.7219 | 0.1788 |
| Naive Bayes Classifier | 0.8063 | 0.8123 | 0.8063 | 0.8092 | 0.7695 | 0.2728 |
| Random Forest | 0.8990 | 0.8956 | 0.8990 | 0.8830 | 0.8587 | 0.5456 |
| XG Boost Model | 0.8891 | 0.8783 | 0.8891 | 0.8787 | 0.8389 | 0.5124 |
 


### Observation

| Model | Observation |
|-------|-------------|
| Simple Logistic Regression | Good accuracy, precision and recall. This means that the model is well balanced with the prediction. The True/False positives/Negatives are balanced. AUC score indicates that the classes are well separated. MCC indicates that the correlation is moderate. Overall, based on this exercise, this is one of the best models.
| Decision Trees | Model looks balanced based on Accuracy, Precision and Recall. However, the correlation observed from MCC looks weak. Decision trees might not have identified complex relationships in this dataset and may be a bit overfitting for unseen data.
| K-Nearest Neighbors | Good Accuracy, however, lesser Precision and Recall - this means that the model is not well balanced in the prediction. Weak MCC scrore indicates that the feature relationships are not determined well. This may become overfitting for unseen data. 
| Naive Bayes Classifier | Balanded Accuracy, Precision and Recall. However, weak MCC indicates poor correlation. This is slightly better than KNN but once again, this could be overfitting for unseen data.
| Random Forest | Good and balanced Accuracy, Precision and Recall. Good correlation observed in MCC. AUC indicates, good defined classification. Overall, this is one of the best models evaluated with this dataset. 
| XG Boost Model | Good and balanced Accuracy, Precision and Recall. Good correlation observed in MCC. Strong classification observed in AUC. This model is slightly behind Random Forest. Overall, this is one of the best models evaluated with this dataset. 

## d. About the utility

Side Bar Panel:
* This application has a side bar panel which has an option to download the dataset
* Only the dataset thus downloaded MUST BE uploaded using the upload option
* After uploading the file, user can select the models from the drop down, which can be trained
* Post model selection, user can select the TRAIN:TEST split - 0.15 means 15% test data and 85% test data
* After selecting the test split parameter, user can click on the Train model button

Main Section:
* Main section has three tabs
   * Data Exploration
   * Model Traing Metrics
   * Model Comparison Metrics
* Once user uploads the dataset, the "Data Exploration" section in the main panel will display the details about the uploaded dataset
* Once user clicks on "Train Model" button, second tab "Model Training Metrics" will display the following metrics
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - AUC Score
   - MCC Score
* Once user trains all the models, the third tab "Model Comparison Metrics" will display the comparison of all the above metrics with respect to the trained models
* The third tab will also highlight the best model with respect to few metrics

