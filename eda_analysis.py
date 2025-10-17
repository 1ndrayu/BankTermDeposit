import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Read cleaned dataset
df = pd.read_csv('d:/code/WindSurfCode/BankTermDeposit/bank_cleaned.csv')

print("=== EXPLORATORY DATA ANALYSIS ===")

# 1. Basic statistics
print("\n1. BASIC STATISTICS")
print("Dataset Shape:", df.shape)
print("\nTarget Distribution:")
print(df['y'].value_counts(normalize=True))

# 2. Numerical features analysis
numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

print("\n2. NUMERICAL FEATURES ANALYSIS")
print("\nNumerical features summary:")
print(df[numerical_cols].describe())

# Histograms for numerical features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Distribution of Numerical Features')

for i, col in enumerate(numerical_cols):
    row, col_pos = divmod(i, 3)
    df[col].hist(ax=axes[row, col_pos], bins=30)
    axes[row, col_pos].set_title(f'{col} Distribution')
    axes[row, col_pos].set_xlabel(col)

plt.tight_layout()
plt.savefig('d:/code/WindSurfCode/BankTermDeposit/numerical_distributions.png', dpi=300, bbox_inches='tight')
print("Saved numerical distributions plot")

# 3. Categorical features analysis
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

print("\n3. CATEGORICAL FEATURES ANALYSIS")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Target Distribution by Categorical Features')

for i, col in enumerate(categorical_cols):
    row, col_pos = divmod(i, 3)

    # Calculate subscription rate by category
    subscription_rate = df.groupby(col)['y'].apply(lambda x: (x == 'yes').mean())

    subscription_rate.plot(kind='bar', ax=axes[row, col_pos])
    axes[row, col_pos].set_title(f'Subscription Rate by {col}')
    axes[row, col_pos].set_xlabel(col)
    axes[row, col_pos].set_ylabel('Subscription Rate')
    axes[row, col_pos].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('d:/code/WindSurfCode/BankTermDeposit/categorical_analysis.png', dpi=300, bbox_inches='tight')
print("Saved categorical analysis plot")

# 4. Correlation analysis
print("\n4. CORRELATION ANALYSIS")
# Correlation matrix for numerical features
plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('d:/code/WindSurfCode/BankTermDeposit/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Saved correlation matrix plot")

# 5. Feature importance based on correlation with target
print("\n5. FEATURE IMPORTANCE")
feature_cols = numerical_cols + [col + '_encoded' for col in categorical_cols]
correlations = df[feature_cols].corrwith(df['y_encoded']).abs().sort_values(ascending=False)

print("\nTop 10 most correlated features with target:")
print(correlations.head(10))

# 6. Outlier analysis summary
print("\n6. OUTLIER ANALYSIS")
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")

print("\nEDA completed. Check the generated PNG files for visualizations.")
