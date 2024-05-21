# We've seen how adding multiple predictors can improve our models. In this
# exercise we'll see how categorical/qualitative variables must be one-hot
# encoded before they can be used to fit a model.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Load the credit data.
df = pd.read_csv('credit.csv')
df.head()


# The response variable will be 'Balance.'

# create a new DataFrame x that is identical to df, except it does not include the 'Balance' column
x = df.drop('Balance', axis=1)

y = df['Balance']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# Trying to fit on all features in their current representation throws an error.
try:
    test_model = LinearRegression().fit(x_train, y_train)
except Exception as e:
    print('Error!:', e)
    

# Inspect the data types of the DataFrame's columns.
df.dtypes


# Fit a linear model using only the numeric features in the dataframe.
# Identify numeric features
numeric_features = x.select_dtypes(include=['int', 'float']).columns
model1 = LinearRegression().fit(x_train[numeric_features], y_train)


# Report train and test R2 scores.
train_score = model1.score(x_train[numeric_features], y_train)
test_score = model1.score(x_test[numeric_features], y_test)
print('Train R2:', train_score)
print('Test R2:', test_score)


# Look at unique values of Ethnicity feature.
print('In the train data, Ethnicity takes on the values:', list(x_train['Ethnicity'].unique()))


# Create x train and test design matrices creating dummy variables for the categorical.
# hint: use pd.get_dummies() with the drop_first hyperparameter for this
x_train_design = pd.get_dummies(x_train, drop_first=True)
x_test_design = pd.get_dummies(x_test, drop_first=True)
x_train_design.head()


# Confirm that all data types are now numeric.
x_train_design.dtypes


# Fit model2 on design matrix
model2 = LinearRegression().fit(x_train_design, y_train)


# Report train and test R2 scores
train_score = model2.score(x_train_design, y_train)
test_score = model2.score(x_test_design, y_test)
print('Train R2:', train_score)
print('Test R2:', test_score)



# Note that the intercept is not a part of .coef_ but is instead stored in .intercept_.
coefs = pd.DataFrame(model2.coef_, index=x_train_design.columns, columns=['beta_value'])
coefs


# Visualize crude measure of feature importance; below code plots feature importance graph 
sns.barplot(data=coefs.T, orient='h').set(title='Model Coefficients');


# Specify best categorical feature
best_cat_feature = 'Student_Yes'

# Define the model.
features = ['Income', best_cat_feature]
model3 = LinearRegression()
model3.fit(x_train_design[features], y_train)

# Collect betas from fitted model.
beta0 = model3.intercept_
beta1 = model3.coef_[features.index('Income')]
beta2 = model3.coef_[features.index(best_cat_feature)]

# Display betas in a DataFrame.
coefs = pd.DataFrame([beta0, beta1, beta2], index=['Intercept']+features, columns=['beta_value'])
coefs


# Visualize crude measure of feature importance.
sns.barplot(data=coefs.T, orient='h').set(title='Model Coefficients');


# Create space of x values to predict on.
x_space = np.linspace(x['Income'].min(), x['Income'].max(), 1000)

# Generate 2 sets of predictions based on best categorical feature value.
# When categorical feature is true/present (1)
# here since the categorical feature is present (1), you need only write beta2
y_hat_yes = beta0 + beta1 * x_space + beta2 

# When categorical feature is false/absent (0)
# here since the categorical feature is absent (0), beta2 * 0 = 0; hence omitting beta2 as well.
y_hat_no = beta0 + beta1 * x_space 

# Plot the 2 prediction lines for students and non-students.
ax = sns.scatterplot(data=pd.concat([x_train_design, y_train], axis=1), x='Income', y='Balance', hue=best_cat_feature, alpha=0.8)
ax.plot(x_space, y_hat_no)
ax.plot(x_space, y_hat_yes);















