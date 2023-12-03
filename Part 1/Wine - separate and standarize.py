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

# Separate the last column
last_col = df.iloc[:, -1]
df = df.iloc[:, :-1]

# Apply the condition to the last column
last_col = last_col.apply(lambda x: 0 if x <= 6 else 1)

# Standardize the dataset
scaler = preprocessing.StandardScaler().fit(df)
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Add the last column back
df_scaled = pd.concat([df_scaled, last_col], axis=1)

# Split the dataset into training and test sets
train, test = train_test_split(df_scaled, test_size=0.2, random_state=42)

# Save them in two .txt files, using tabulation as a marker
train.to_csv('A2-wine/wine-train.txt', sep='\t', index=False)
test.to_csv('A2-wine/wine-test.txt', sep='\t', index=False)
