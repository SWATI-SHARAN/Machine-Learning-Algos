from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris # Loads the classic Iris dataset, which is built into scikit-learn.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score  # measure the accuracy of your model — how many test samples were classified correctly.

iris = load_iris()
X, y = iris.data, iris.target
#iris.data holds the input features and iris.target holds the class labels (0, 1, or 2).
#X = 2D array of features (150 samples × 4 features)
#y = 1D array of target labels (150 elements, values: 0, 1, or 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state =42)
clf.fit(X_train, y_train) #Trains the decision tree on the training set.
y_pred = clf.predict(X_test) #Uses the trained model to predict class labels for the test set
accuracy = accuracy_score(y_test, y_pred) #Compares the predicted labels (y_pred) to the actual labels (y_test) to compute accuracy.
print(f"Accuracy: {accuracy}")