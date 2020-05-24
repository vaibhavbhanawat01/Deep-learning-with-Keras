# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:03:38 2020

@author: vaibhav_bhanawat
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as mat

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_country = LabelEncoder()
label_gender = LabelEncoder()
X[:, 1] = label_country.fit_transform(X[:, 1])
X[:, 2] = label_gender.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
columnTran = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], 
                               remainder = 'passthrough')
X = np.array(columnTran.fit_transform(X))
X = X[:, 1:]

# Train and test data split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, 
                                                    random_state = 0)

# feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Building ANN Model
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialze the ANN model
classifier = Sequential()

# adding input and first hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', 
                     input_dim = 11))
# adding second hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))

# adding output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

# compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# fitting the training set to ANN
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)

# predicating the test set data
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# making the confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)









