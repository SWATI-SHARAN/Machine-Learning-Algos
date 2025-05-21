from sklearn.model_selection import train_test_split  #This function is used to split your dataset into a training set and a testing set
import numpy as np

np.random.seed(42)  # For reproducibility
X = np.random.rand(100,1)
Y = 2 * X.squeeze() + 1.1 +np.random.randn(100)
#X.squeeze() turns the shape from (100, 1) to (100,) → a flat array.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#X_train, Y_train: for training the model
#X_test, Y_test: for testing/evaluation
#test_size=0.2: 20% of the data (20 samples) goes into the test set 80% (80 samples) goes into the train set
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
