# Medical Expense Prediction Model

Problem statement: Predict the future medical expenses of patients based on certain features.Factors affecting the medical expenses of the patients-age,gender,bodymassindex,region,smoking behaviour,medical health expenses.

# Importing Libraries

#for mathemaical operations
import numpy as np

#for dataframe manipulations
import pandas as pd

#for data visualizations
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#for Model building
from sklearn.model_selection import train_test_split

#setting parameters for visualization
plt.rcParams['figure.figsize'] = (16, 5)
plt.style.use('fivethirtyeight')


# Analysis

1.Descriptive statistics
2.Univariate Analysis
3.Bivariate  Analysis
4.Multivariate analysis


# Feature engineering

Feature engineering was done on features of dataset. These Feature engineering techniques were applied to make the dataset such that Machine Learning algorithms can have a maximum accuracy in predicting the medical expenses.


# Data Sets

After successfull Feature engineering, a training data set and a testing data set were made.
Expenses column was separated from original dataset as it is to be predited.

# Predictive modelling

Different Predictive models were applied on the training data set to get maximum accuracy.

1.Linear Regression
2.Random Forest Regression
3.Gradient Boosting Regression.


# Conclusion

Based on the predictive modeling, Gradient Boosting Model algorithm has the best score compared to the others, with RMSE Score : 4093.2715100014866,R2 Score : 0.8947095107441357.Gradient Boosting Model is fit based on the train & test accuracy.



