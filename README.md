# -CODSOFT-
Movie Genre Classification Using Machine Learning A text-based classification system that predicts the genre of a movie based on its plot summary using traditional NLP techniques and machine learning models. Built using Python, Scikit-learn, and NLTK.


# Dataset Documentation for Credit Card Fraud Detection Project

## Dataset Overview
This project utilizes a dataset containing credit card transactions, which is used to detect fraudulent activities. The dataset includes various features related to the transactions, such as transaction amount, time, and user information.

## Source
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/dalpozz/creditcard-fraud) and is publicly available for research purposes.

## Dataset Structure
The dataset consists of the following columns:

- **Time**: The number of seconds elapsed between this transaction and the first transaction in the dataset.
- **V1, V2, ..., V28**: These are the principal components obtained with PCA (Principal Component Analysis) to protect the anonymity of the users.
- **Amount**: The transaction amount.
- **Class**: This is the target variable, where 1 indicates a fraudulent transaction and 0 indicates a legitimate transaction.

## Preprocessing Steps
Before using the dataset for model training, the following preprocessing steps are recommended:

1. **Data Cleaning**: Check for any inconsistencies or errors in the dataset.
2. **Handling Missing Values**: Identify and handle any missing values appropriately.
3. **Normalization**: Scale the 'Amount' feature to ensure that it is on a similar scale as the other features.
4. **Encoding**: If there are any categorical variables, encode them into numerical format.

## Usage
To load and preprocess the dataset, refer to the `data_preprocessing.py` file in the `src` directory. This file contains functions that will help in cleaning and preparing the dataset for analysis and model training.
