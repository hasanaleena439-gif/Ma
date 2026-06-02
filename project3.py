#house price prediction using linear regression
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split    
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# loading the dataset
housing_dataset = sklearn('housing.csv')
house_price_data = pd.DataFrame(housing_dataset.data, columns=housing_dataset.feature_names)
house_price_data['price'] = housing_dataset.target
print(house_price_data.head())
print(house_price_data.shape)
# separating the data and labels
X = house_price_data.drop(columns='price')
Y = house_price_data['price']
print(X)
print(Y)
# splitting the data into training and testing dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
# model training
model = LinearRegression()
model.fit(X_train, Y_train)
# model evaluation
Y_pred = model.predict(X_test)
r2_score = r2_score(Y_test, Y_pred)
print("R2 Score:", r2_score)
