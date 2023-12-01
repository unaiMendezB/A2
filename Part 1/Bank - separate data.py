import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('A2-bank/bank-additional-full.csv', sep=';')

# Convert categorical variables to numerical representations
categorical_cols = df.columns[df.dtypes==object].tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split the dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Initialize a scaler
scaler = StandardScaler()

# Fit on the training set only
scaler.fit(train_df)

# Apply the transform to both the training set and the test set
train_df = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

# Save the training and test sets to csv files
train_df.to_csv('A2-bank/bank-additional-train.csv', index=False)
test_df.to_csv('A2-bank/bank-additional-test.csv', index=False)
