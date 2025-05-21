#Ensures our model peforms well on new unseen data too
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target

svm_classifier = SVC(kernel='linear', C=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(svm_classifier, X, y, cv=kfold)
print("Accuracy scores for each fold:", accuracy_scores)
print("Mean accuracy:", accuracy_scores.mean())
