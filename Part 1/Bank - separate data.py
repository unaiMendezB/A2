import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('A2-bank/bank-additional-full.csv', sep=';')

# Replace 'unknown' with NaN
df.replace('unknown', np.nan, inplace=True)

# Separate the last column
last_col = df.iloc[:, -1]
df = df.iloc[:, :-1]

# Convert categorical variables to dummy variables
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Standardize the numerical columns in the dataset
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
scaler = preprocessing.StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Replace True with 0 and False with 1
df.replace({True: 0, False: 1}, inplace=True)

# Apply the yes/no formula to the last column
last_col = last_col.map({'yes': 0, 'no': 1})

# Add the last column back
df = pd.concat([df, last_col], axis=1)

# Split the dataset into train and test
train_bank, test_bank = train_test_split(df, test_size=0.2, random_state=42)

# Save them in two .txt files, using tabulation as a marker
train_bank.to_csv('A2-bank/bank-additional-train.txt', sep='\t', index=False)
test_bank.to_csv('A2-bank/bank-additional-test.txt', sep='\t', index=False)
