import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df_train = pd.read_csv('bank-additional-train.txt', sep=",")
df_test = pd.read_csv('bank-additional-test.txt', sep=",")

# Split the data into features and target variable
X_train = df_train.drop('y_yes', axis=1)
y_train = df_train['y_yes']
X_test = df_test.drop('y_yes', axis=1)
y_test = df_test['y_yes']

# Define the parameter range for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}

# Create a GridSearchCV object and fit it to the training data
svc = svm.SVC()
grid = GridSearchCV(svc, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

# Predict the responses for test dataset
y_pred = grid.predict(X_test)

# Print the best parameters found
print("Best Parameters:", grid.best_params_)

# Print the kernel used
print("Kernel Used:", grid.best_estimator_.kernel)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", accuracy_score(y_test, y_pred))

# Model Precision and Recall
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Compute the classification error on the Test and Validation sets
classification_error = 1 - accuracy_score(y_test, y_pred)
print("Classification Error:", classification_error)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(svc, X_train, y_train, cv=5)
print("5-fold Cross-validation Scores:", cv_scores)
print("Mean 5-fold Cross-validation Score:", cv_scores.mean())
