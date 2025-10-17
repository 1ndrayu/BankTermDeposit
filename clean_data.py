import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('d:/code/WindSurfCode/BankTermDeposit/bank-full.csv', sep=';')

print("=== DATA CLEANING AND PREPROCESSING ===")

# 1. Check for outliers in numerical columns
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

print("\nNumerical columns statistics:")
print(df[numerical_cols].describe())

# Check for outliers using IQR method
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n{col} outliers count: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  Lower bound: {lower_bound}, Upper bound: {upper_bound}")

# 2. Handle outliers - for now, we'll keep them as they might be valid in banking data
# But let's check if pdays has -1 values which might indicate missing data
print(f"\nUnique values in pdays: {df['pdays'].unique()}")
print(f"Count of -1 in pdays: {(df['pdays'] == -1).sum()}")

# 3. Encode categorical variables
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
label_encoders = {}

print("\nEncoding categorical variables...")
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  {col}: {df[col].unique()} -> {list(range(len(df[col].unique())))}")

# 4. Encode target variable
target_encoder = LabelEncoder()
df['y_encoded'] = target_encoder.fit_transform(df['y'])

# 5. Scale numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 6. Create final feature matrix
feature_cols = numerical_cols + [col + '_encoded' for col in categorical_cols]
X = df[feature_cols]
y = df['y_encoded']

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Feature columns: {feature_cols}")

# Save cleaned data
df.to_csv('d:/code/WindSurfCode/BankTermDeposit/bank_cleaned.csv', index=False)
print("\nCleaned dataset saved to bank_cleaned.csv")

# Show correlation with target
print("\nFeature correlation with target (y_encoded):")
correlations = X.corrwith(y).sort_values(ascending=False)
print(correlations)
