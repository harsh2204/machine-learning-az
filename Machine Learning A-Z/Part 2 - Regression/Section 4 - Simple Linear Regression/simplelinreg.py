# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values #Remove the last column | Matrix of independent variables
y = dataset.iloc[:, 1].values # Vector of dependant variables

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting the data to a simple linear regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting
y_pred = regressor.predict(X_test)

#Plotting the predictions and test sets

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='black')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary in dollars')
plt.show()

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='black')
plt.title('Salary vs. Experience (Tet Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary in dollars')
plt.show()
