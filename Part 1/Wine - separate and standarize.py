import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('A2-wine/winequality-white.csv', sep=';')

# Check if there's any missing data
if df.isnull().values.any():
    # Replace null values with 0
    df.fillna(0, inplace=True)

# Standardize the dataset
scaler = preprocessing.StandardScaler().fit(df)
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Split the dataset into training and test sets
train, test = train_test_split(df_scaled, test_size=0.2, random_state=42)

# Save the training and test sets to files
np.savetxt('A2-wine/wine-train.txt', train.values, fmt='%f')
np.savetxt('A2-wine/wine-test.txt', test.values, fmt='%f')
