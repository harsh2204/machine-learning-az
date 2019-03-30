# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:, 3])
colT = ColumnTransformer([("State", OneHotEncoder(categories =[[0,1,2]]),[0]),
                          ("other", "passthrough",[0,1,2])])
 
X= colT.fit_transform(X)


# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)

# Feature Scaling
'''
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)
# We apply feature scaling for dummy variables as well here.
'''