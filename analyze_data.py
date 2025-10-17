import pandas as pd
import numpy as np

# Read the dataset
df = pd.read_csv('d:/code/WindSurfCode/BankTermDeposit/bank-full.csv', sep=';')

# Print basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nTarget variable distribution:")
print(df['y'].value_counts())
print("\nMissing values:")
print(df.isnull().sum())
print("\nColumn types:")
print(df.dtypes)
