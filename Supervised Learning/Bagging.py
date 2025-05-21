#Bootstrap Aggregating
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


base_classifier = DecisionTreeClassifier(random_state=42)
bagging_classifier = BaggingClassifier(estimator=base_classifier, n_estimators=10, random_state=42)
#creates an ensemble of classifiers.
#estimator=base_classifier means each ensemble member is a Decision Tree.
#n_estimators=10 means the ensemble contains 10 Decision Trees.
#Each tree is trained on a bootstrap sample (random sample with replacement) from the training data.

bagging_classifier.fit(X_train, y_train)

y_pred_bagging = bagging_classifier.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print(f'Bagging Accuracy: {accuracy_bagging:.2f}')