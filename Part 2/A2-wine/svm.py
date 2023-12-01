import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the datasets
train_data = pd.read_csv('wine-train.txt', sep='\t', header=None,
                         names=['Fixed_Acidity', 'Volatile_Acidity', 'Citric_Acid', 'Residual_Sugar', 'Chlorides',
                                'Free_Sulfur_Dioxide', 'Total_Sulfur_Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol',
                                'Quality'])
test_data = pd.read_csv('wine-test.txt', sep='\t', header=None,
                        names=['Fixed_Acidity', 'Volatile_Acidity', 'Citric_Acid', 'Residual_Sugar', 'Chlorides',
                               'Free_Sulfur_Dioxide', 'Total_Sulfur_Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol',
                               'Quality'])

# Split the datasets into features and labels
X_train = train_data[
    ['Fixed_Acidity', 'Volatile_Acidity', 'Citric_Acid', 'Residual_Sugar', 'Chlorides',
     'Free_Sulfur_Dioxide', 'Total_Sulfur_Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']]
y_train = train_data['Quality']
X_test = test_data[
    ['Fixed_Acidity', 'Volatile_Acidity', 'Citric_Acid', 'Residual_Sugar', 'Chlorides',
     'Free_Sulfur_Dioxide', 'Total_Sulfur_Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']]
y_test = test_data['Quality']

# Define the parameter range for grid search
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'poly', 'sigmoid']}

# Create a GridSearchCV object and fit it to the training data
svc = svm.SVC()
grid = GridSearchCV(svc, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

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
