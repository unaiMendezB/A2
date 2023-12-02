import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the datasets
train_data = pd.read_csv('A2-ring-merged-normalized.txt', sep='\t', header=None, names=['A', 'B', 'Result'])
test_data = pd.read_csv('A2-ring-test-normalized.txt', sep='\t', header=None, names=['A', 'B', 'Result'])

# Split the datasets into features and labels
X_train = train_data[['A', 'B']]
y_train = train_data['Result']
X_test = test_data[['A', 'B']]
y_test = test_data['Result']

# Define the parameter range for grid search
param_grid = [{'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear']}]

# Create a GridSearchCV object and fit it to the training data
svc = svm.SVC()
grid = GridSearchCV(svc, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid.best_params_)

# Predict the labels for the test set
y_pred = grid.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Compute the classification error
E = 100 * (cm[0][1] + cm[1][0]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

# Model Precision and Recall
print(classification_report(y_test, y_pred))

# Print the classification error and confusion matrix
print("Classification error: ", E)
print("Confusion matrix: \n", cm)

'''
# Predict the responses for test dataset
y_pred = grid.predict(X_test)

# Print the best parameters
print("Best parameters found: ", grid.best_params_)

# Model Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Model Precision and Recall
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

'''