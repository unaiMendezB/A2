import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('bank-additional-train.txt', sep='\t')
test_data = pd.read_csv('bank-additional-test.txt', sep='\t')

# Split the datasets into features and labels
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# Create a Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Apply cross-validation
scores = cross_val_score(model, X_train, y_train, cv=20)
print(f'Cross-validation scores: {scores}')

# Predict the test set results
y_pred = model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, (y_pred > 0.99))
print(f'Confusion matrix: \n{cm}')

# Compute the classification error
E = 100 * (cm[0][1] + cm[1][0]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
print("Classification error: ", E)

# Calculate the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print(f'ROC AUC: {roc_auc}')

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Perform PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Plot the PCA
plt.figure()
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Plot')
plt.show()

# Print the parameters of the Multiple Linear Regression model
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')
