# %%
from numpy.random.mtrand import RandomState
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plt

df = pd.read_csv("Fish.csv")

# Data Parameters:
  # Length 1 = Vertical length in centimeters
  # Legnth 2 = Diagonal length in centimeters
  # Length 3 = Cross length in centimeters
  # Height = Height in centimeters
  # Width = Diagonal width in centimeters

# We will split the dataset into the continuous input variables vs continuous output variables
  # x will represent our inputs
  # y will represent out outputs
x = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = df[['Weight']]

# Train test split will split 80% for training and 20% of the data for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Linear Regression #
ols = LinearRegression()
ols.fit(x_train, y_train)
y_prediction = ols.predict(x_test)
ytrain_prediction = ols.predict(x_train)

# Training Deta #
print("Training Data(Linear Regression):")
print('Linear Regression Accuracy:', ols.score(x_train, y_train))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, ytrain_prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, ytrain_prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, ytrain_prediction)))

# Test Data #
print("\nTest Data(Linear Regression):")
print('Coefficients/Weights:', ols.coef_)
print('Intercept/Bias:', ols.intercept_)
print('Linear Regression Accuracy:', ols.score(x_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))

# Gradient Descent #
grad_des = SGDRegressor(max_iter = 1000000, alpha = .000001, learning_rate = 'constant', eta0 = 1e-7)
grad_des.fit(x_train, np.ravel(y_train))
yds_prediction = grad_des.predict(x_test)
ydstrain_prediction = grad_des.predict(x_train)

# Training Data#
print("\nTraining Data(Gradient Descent):")
print('Gradient Descent Accuracy:', r2_score(y_train, ydstrain_prediction))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, ydstrain_prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, ydstrain_prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, yds_prediction)))

# Test Data #
print("\nTest Data(Gradient Descent):")
print('Coefficients/Weights:', grad_des.coef_)
print('Intercept/Bias:', grad_des.intercept_)
print('Gradient Descent Accuracy:', r2_score(y_test, yds_prediction))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, yds_prediction))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, yds_prediction))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, yds_prediction)))
print('Iterations:', grad_des.n_iter_)

# %%
