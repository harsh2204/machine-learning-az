# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Make a linear regression model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Make a polynomial regression model
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)

X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualizing the results
plt.scatter(X, y, color='green')

plt.plot(X, lin_reg.predict(X), color='red')

plt.title('Results of the Linear regression model')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Poly
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='green')

plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')

plt.title('Results of the Polynomial regression model')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Predicting the results
# Linear
lin_reg.predict([[6.5]])
# Poly
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
