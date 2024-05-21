# The aim of this exercise is to train 5 models - a Linear Regression model, two
# Polynomial Regression models and two kNN regression models- and decide which a
# among them is the best?


# Import necessary libraries
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# Load the data from the csv file
df = pd.read_csv("dataset.csv")

# Take a quick look at the data
df.head()


# Use the values of column x as the predictor variable
x = df[['x']].values

# Use the values of column y as the response variable
y = df['y'].values


# Split the data into train and test splits
# Set aside 25% for testing with a random state of 1
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=1)


# Create an empty list to store test MSEs
test_error = []


# Linear Regression
# Initialize a simple Linear Regression model
model = LinearRegression()

# Fit the model on the train data
model.fit(x_train, y_train)

# Predict using the trained model on the test data
y_pred = model.predict(x_test)

# Compute the MSE of the test predictions
mse = mean_squared_error(y_test, y_pred)

# Append the error to the list initialized above
test_error.append(mse)


# Polynomial Regression

# Initialize a list of degree values to create polynomial features
degree_list = [2,5]

# Run a for loop through the degrees of the polynomial
for d in degree_list:
    
    # Compute the polynomial features for the train data, for the current degree
    X_train = PolynomialFeatures(degree=d).fit_transform(x_train)
    
    # Compute the polynomial features for the test data, for the current degree
    X_test =  PolynomialFeatures(degree=d).fit_transform(x_test)
    
    # Initialize a linear regression model
    lreg = LinearRegression()
    
    # Fit the linear model on the transformed train data
    lreg.fit(X_train,y_train)
    
    # Predict using the trained model on the test data
    y_pred = lreg.predict(X_test)
    
    # Compute the MSE of the test predictions
    mse = mean_squared_error(y_test, y_pred)

    # Append the error to the list initialized above
    test_error.append(mse)


# kNN Regression
# Initialize a list of k values to specify the number of neighbors
knn_list = [1,20]

# Loop through the k values from the list defined above
for i in knn_list:
    
    # Initialize a kNN model with the current k value
    model =  KNeighborsRegressor(n_neighbors=int(i))
    
    # Fit the model on the train data
    model.fit(x_train,y_train)
    
    # Predict using the trained model on the test data
    y_pred = model.predict(x_test)
    
    # Compute the MSE of the test predictions
    mse = mean_squared_error(y_test, y_pred)

    # Append the error to the list initialized above
    test_error.append(mse)

























