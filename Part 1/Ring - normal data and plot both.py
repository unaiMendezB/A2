import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

# Read the data from the file
data = pd.read_csv('A2-ring/A2-ring-merged.txt', sep='\t', header=None)

# Assign columns to variables
x = data[0]
y = data[1]
colors = data[2]

# Map 0 to 'green' and 1 to 'red'
color_map = {0: 'green', 1: 'red'}
colors = colors.map(color_map)

# Create two copies of the data
data_normalized = data.copy()
data_standardized = data.copy()

# Apply normalization
scaler = MinMaxScaler()
data_normalized[[0, 1]] = scaler.fit_transform(data_normalized[[0, 1]])


# Save normalized data to a CSV file
data_normalized.to_csv('A2-ring/A2-ring-merged-normalized.txt', sep='\t', header=False, index=False)

# Apply standardization
scaler = StandardScaler()
data_standardized[[0, 1]] = scaler.fit_transform(data_standardized[[0, 1]])

# Save standardized data to a CSV file
data_standardized.to_csv('A2-ring/A2-ring-merged-standardized.txt', sep='\t', header=False, index=False)

# Plot normalized data
plt.figure()
plt.scatter(data_normalized[0], data_normalized[1], c=colors)
plt.title('Normalized Data')
plt.savefig(os.path.join(os.path.dirname('A2-ring/A2-ring-merged.txt'), 'Normalized_Data.png'))
plt.show()

# Plot standardized data
plt.figure()
plt.scatter(data_standardized[0], data_standardized[1], c=colors)
plt.title('Standardized Data')
plt.savefig(os.path.join(os.path.dirname('A2-ring/A2-ring-merged.txt'), 'Standardized_Data.png'))
plt.show()
