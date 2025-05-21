import numpy as np
from sklearn.model_selection import MinMaxScaler

np.random.seed(42)
data = np.random.rand(100, 5) * 100 +500 # Example data

scalar = MinMaxScaler()
scaled_data = scalar.fit_transform(data)

print("Original Data:\n", data[:5])
print("Scaled Data:\n", scaled_data[:5])
print("Min-Max Scaler Parameters:\n", scalar.get_params())