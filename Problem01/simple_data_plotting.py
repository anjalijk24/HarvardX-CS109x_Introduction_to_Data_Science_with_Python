
# The aim of this exercise is to plot TV Ads vs Sales based on the Advertisement dataset

import pandas as pd
import matplotlib.pyplot as plt


# Reading the Advertisement dataset

# "Advertising.csv" containts the data set used in this exercise
data_filename = 'Advertising.csv'

# Read the file "Advertising.csv" file using the pandas library
df = pd.read_csv(data_filename)


# Get a quick look of the data
#df.shape
#(200, 4)

# Create a new dataframe by selecting the first 7 rows of
# the current dataframe
df_new = df.iloc[:7, [0, 3]]

# Use a scatter plot for plotting a graph of TV vs Sales
plt.scatter(df_new['TV'], df_new['Sales'])

#or
#plt.scatter(df_new.iloc[:, 0], df_new.iloc[:, 1])

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel("TV budget")
plt.ylabel("Sales")

# Add plot title 
plt.title("TV vs Sales")


# Use a scatter plot for plotting all points
plt.scatter(df['TV'], df['Sales'])

# Add axis labels for clarity (x : TV budget, y : Sales)
plt.xlabel("TV budget")
plt.ylabel("Sales")

# Add plot title 
plt.title("TV vs Sales")

