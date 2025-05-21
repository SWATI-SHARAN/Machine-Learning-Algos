import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = 3 * np.random.rand(100, 2) # 100 samples, 2 features
Y = 4 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100) # Linear relation with noise
model = LinearRegression()
model.fit(X, Y)

coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)