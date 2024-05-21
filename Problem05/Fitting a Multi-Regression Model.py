# The aim of this exercise is to understand how to use multi regression. Here we will observe the difference in MSE for each
# model as the predictors change.

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Read the file "Advertising.csv"
df = pd.read_csv("Advertising.csv")

# Take a quick look at the data to list all the predictors
df.head()


# Initialize a list to store the MSE values
mse_list = []


# Create a list of lists of all unique predictor combinations
# For example, if you have 2 predictors,  A and B, you would 
# end up with [['A'],['B'],['A','B']]
cols = [list(df.columns[:-1][i:j]) for i in range(len(df.columns) - 1) for j in range(i + 1, len(df.columns))]
cols.append(['TV', 'Newspaper'])

# Loop over all the predictor combinations 
for i in cols:

    # Set each of the predictors from the previous list as x
    x = df[i]
    
    # Set the "Sales" column as the reponse variable
    y = df["Sales"]
   
    # Split the data into train-test sets with 80% training data and 20% testing data. 
    # Set random_state as 0
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=0)


    # Initialize a Linear Regression model
    lreg = LinearRegression()

    # Fit the linear model on the train data
    lreg.fit(x_train,y_train)

    # Predict the response variable for the test set using the trained model
    y_pred = lreg.predict(x_test)
    
    # Compute the MSE for the test data
    MSE = mean_squared_error(y_test, y_pred)
    
    # Append the computed MSE to the initialized list
    mse_list.append(MSE)


# Helper code to display the MSE for each predictor combination
t = PrettyTable(['Predictors', 'MSE'])

for i in range(len(mse_list)):
    t.add_row([cols[i],round(mse_list[i],3)])

print(t)

# The result will look like as follows:
#+------------------------------+--------+
#|          Predictors          |  MSE   |
#+------------------------------+--------+
#|            ['TV']            | 10.186 |
#|       ['TV', 'Radio']        | 4.391  |
#| ['TV', 'Radio', 'Newspaper'] | 4.402  |
#|          ['Radio']           | 24.237 |
#|    ['Radio', 'Newspaper']    | 24.783 |
#|        ['Newspaper']         | 32.137 |
#|     ['TV', 'Newspaper']      | 8.688  |
#+------------------------------+--------+


