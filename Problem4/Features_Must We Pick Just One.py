# The aim of this exercise is to understand how to use Multi-Linear Regression. Here we will compare simple
# Linear Regression models consisting of different columns with a Multi-linear Regression model comprising of
# all columns.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper import fit_and_plot_linear, fit_and_plot_multi


# Read the file "Advertising.csv"
df = pd.read_csv("Advertising.csv")

# Take a quick look at the dataframe
df.head()


# Define an empty Pandas dataframe to store the R-squared value associated with each 
# predictor for both the train and test split
df_results = pd.DataFrame(columns=['Predictor', 'R2 Train', 'R2 Test'])


# For each predictor in the dataframe, call the function "fit_and_plot_linear()"
# from the helper file with the predictor as a parameter to the function

# This function will split the data into train and test split, fit a linear model
# on the train data and compute the R-squared value on both the train and test data

#-----------------one way---------------------------
# Iterate over each predictor in the dataframe
for predictor in df.columns[:-1]:
    # Call the function "fit_and_plot_linear()" with the predictor as a parameter
    r2_train, r2_test = fit_and_plot_linear(df[[predictor]])
    
    # Append the results to the dataframe
    df_results = df_results.append({'Predictor': predictor, 'R2 Train': r2_train, 'R2 Test': r2_test}, ignore_index=True)



# Call the function "fit_and_plot_multi()" from the helper to fit a multilinear model
# on the train data and compute the R-squared value on both the train and test data

r2_train, r2_test = fit_and_plot_multi()

# Append the result into df_results
df_results.loc[len(df_results)] = ['Multi-linear Regression', r2_train, r2_test]
#-----------------one way---------------------------


#-----------------another way---------------------------
r2_train_tv, r2_test_tv = fit_and_plot_linear(df[["TV"]])

r2_train_radio, r2_test_radio = fit_and_plot_linear(df[["Radio"]])

r2_train_paper, r2_test_paper = fit_and_plot_linear(df[["Newspaper"]])

r2_train_ml, r2_test_ml = fit_and_plot_multi()


# Store the R-squared values for all models
# in the dataframe intialized above

# **Your code here**
df_results.loc[len(df_results)] = ['TV', r2_train_tv, r2_test_tv]
df_results.loc[len(df_results)] = ['Radio', r2_train_radio, r2_test_radio]
df_results.loc[len(df_results)] = ['Newspaper', r2_train_paper, r2_test_paper]
df_results.loc[len(df_results)] = ['Multi-linear Regression', r2_train_ml, r2_test_ml]

#-----------------another way---------------------------


# Take a quick look at the dataframe
df_results.head()








