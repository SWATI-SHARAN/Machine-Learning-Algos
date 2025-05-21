import numpy as np  #numpy is used for creating numerical data arrays.
from sklearn.linear_model import LinearRegression
# LinearRegression is a class from scikit-learn that implements linear regression (including multiple linear regression).

np.random.seed(42) # seed is a starting point for a random number generator (RNG). It ensures that the sequence of random numbers you get is repeated
X = 3 * np.random.rand(100, 2) # 100 samples, 2 features
Y = 4 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100) # Linear relation with noise
#4: the intercept

#3 * X[:, 0]: contribution of feature 1

#X[:, 0]:
#This gets all the rows (:) of the first column (column index 0) from the matrix X.
#Represents the first feature (X1) of the dataset.

#5 * X[:, 1]: contribution of feature 2

#np.random.randn(100): adds random Gaussian noise to make the data more realistic

# np.random.rand() picks new random values every time.

model = LinearRegression() #Creates an instance of the Linear Regression model.
model.fit(X, Y)  #Tains (fits) the model on the data.The model learns the best coefficients and intercept to predict Y from X.

coefficients = model.coef_  #learned coefficients
intercept = model.intercept_ #learned intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)