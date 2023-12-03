import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('A2-ring-merged-normalized.txt', sep='\t', header=None, names=['A', 'B', 'Result'])
test_data = pd.read_csv('A2-ring-test-normalized.txt', sep='\t', header=None, names=['A', 'B', 'Result'])

# Split the datasets into features and labels
X_train = train_data[['A', 'B']]
y_train = train_data['Result']
X_test = test_data[['A', 'B']]
y_test = test_data['Result']

# Define the parameter space for grid search
param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# Initialize the MLP classifier
mlp = MLPClassifier(max_iter=100)

# Perform grid search with cross-validation
clf = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy')
clf.fit(X_train,y_train)

# Print the best parameters
print("Best parameters found: ", clf.best_params_)

# Predict the test set results
y_pred = clf.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n", cm)

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('ROC curve (area = %0.2f)' % roc_auc)

# Plot ROC curve
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
principalComponents = pca.fit_transform(X_train)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# Plot PCA
plt.figure(figsize = (8,8))
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], c = y_train)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.title('2 Component PCA', fontsize = 20)
plt.show()
