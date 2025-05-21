import numpy as np
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42) #Fixes the random state so that the generated numbers are the same each time you run the script.
data = np.random.rand(100, 5) * 100 +500 # Example data
#Generates a 100×5 matrix of random numbers between 0 and 1.
# *100 +500 Scales each value from [0, 1] → [0, 100], then shifts it to [500, 600].

scalar = MinMaxScaler()
#Creates a scaler object that will normalize data to a specific range—by default, [0, 1].
scaled_data = scalar.fit_transform(data)
#First, fit() computes the min and max of each column.
#Then transform() applies the Min-Max scaling formula.
#Result: scaled_data contains values from 0 to 1 for each column.

print("Original Data:\n", data[:5]) #Displays the first 5 samples (rows) from the original dataset before scaling
print("\n Scaled Data:\n", scaled_data[:5])