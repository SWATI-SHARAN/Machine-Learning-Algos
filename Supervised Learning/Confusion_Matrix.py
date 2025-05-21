import seaborn as sns     #seaborn and matplotlib.pyplot are used for plotting the confusion matrix heatmap.
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits #load_digits() is a built-in dataset in sklearn.datasets, containing images of digits (0–9).
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

X, y =load_digits(return_X_y=True)
#X: Shape (1797, 64) — each digit image is 8x8 pixels (flattened to 64 features).
#y: Target digits (0–9) for each image.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
#Creates a heatmap of the confusion matrix:
#annot=True: Show the numbers.
#fmt='d': Format as integer.
#cmap='Blues': Color scheme.
#xticklabels, yticklabels: Show digits 0–9 on axes.

plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title(label='Confusion Matrix', fontsize=17)
# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
plt.show()