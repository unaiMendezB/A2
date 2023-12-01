import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load your training and test datasets
df_train = pd.read_csv('bank-additional-train.csv')
df_test = pd.read_csv('bank-additional-test.csv')

X_train = df_train.drop('y_yes', axis=1)
y_train = df_train['y_yes']

X_test = df_test.drop('y_yes', axis=1)
y_test = df_test['y_yes']

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define the parameter range for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf']}

# Perform grid search with 5-fold cross-validation
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid.best_params_)

# Predict the labels of the test set
y_pred = grid.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)

# Compute the classification error
error = 100 * (cm[0][1] + cm[1][0]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
print("Classification Error (%): ", error)
