import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and test data
model = joblib.load('d:/code/WindSurfCode/BankTermDeposit/final_model.pkl')
df = pd.read_csv('d:/code/WindSurfCode/BankTermDeposit/bank_cleaned.csv')

# Prepare features and target
feature_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
                'job_encoded', 'marital_encoded', 'education_encoded', 'default_encoded',
                'housing_encoded', 'loan_encoded', 'contact_encoded', 'month_encoded', 'poutcome_encoded']

X = df[feature_cols]
y = df['y_encoded']

# Split the data (70:30) - same as before
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("=== FEATURE ANALYSIS FOR TRUE POSITIVES AND TRUE NEGATIVES ===")

# Get predictions and probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Get confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"True Negatives: {tn}, False Positives: {fp}")
print(f"False Negatives: {fn}, True Positives: {tp}")

# Create dataframe with actual, predicted, and probabilities
results_df = X_test.copy()
results_df['actual'] = y_test
results_df['predicted'] = y_pred
results_df['probability'] = y_pred_prob
results_df['correct'] = (y_test == y_pred)

# Analyze features for true positives (correctly predicted 'yes')
tp_data = results_df[(results_df['actual'] == 1) & (results_df['predicted'] == 1)]
tn_data = results_df[(results_df['actual'] == 0) & (results_df['predicted'] == 0)]
fp_data = results_df[(results_df['actual'] == 0) & (results_df['predicted'] == 1)]
fn_data = results_df[(results_df['actual'] == 1) & (results_df['predicted'] == 0)]

print("\n=== FEATURE ANALYSIS ===")

# Calculate mean values for each group
feature_analysis = {}
for col in feature_cols:
    feature_analysis[col] = {
        'TP_mean': tp_data[col].mean(),
        'TN_mean': tn_data[col].mean(),
        'FP_mean': fp_data[col].mean(),
        'FN_mean': fn_data[col].mean(),
        'TP_std': tp_data[col].std(),
        'TN_std': tn_data[col].std(),
        'overall_mean': results_df[col].mean()
    }

# Create comparison dataframe
comparison_df = pd.DataFrame(feature_analysis).T
comparison_df['TP_vs_overall_diff'] = comparison_df['TP_mean'] - comparison_df['overall_mean']
comparison_df['TN_vs_overall_diff'] = comparison_df['TN_mean'] - comparison_df['overall_mean']
comparison_df['TP_vs_TN_diff'] = comparison_df['TP_mean'] - comparison_df['TN_mean']

# Sort by absolute difference between TP and TN means
comparison_df['abs_diff'] = abs(comparison_df['TP_vs_TN_diff'])
comparison_df = comparison_df.sort_values('abs_diff', ascending=False)

print("\nTop 10 features that differ most between True Positives and True Negatives:")
print(comparison_df[['TP_mean', 'TN_mean', 'TP_vs_TN_diff', 'abs_diff']].head(10))

# Visualize the most important features
top_features = comparison_df.head(5).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Feature Distributions: True Positives vs True Negatives')

for i, feature in enumerate(top_features):
    if i >= 6:  # Only show first 6
        break

    row = i // 3
    col = i % 3

    # Create histograms
    axes[row, col].hist(tp_data[feature], alpha=0.7, label='True Positive', bins=20, density=True)
    axes[row, col].hist(tn_data[feature], alpha=0.7, label='True Negative', bins=20, density=True)
    axes[row, col].set_title(f'{feature}\nTP mean: {tp_data[feature].mean():.3f}, TN mean: {tn_data[feature].mean():.3f}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Density')
    axes[row, col].legend()

plt.tight_layout()
plt.savefig('d:/code/WindSurfCode/BankTermDeposit/feature_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved feature analysis plot")

# Calculate correlation between features and prediction confidence for correct predictions
tp_prob_corr = {}
tn_prob_corr = {}

for feature in feature_cols:
    # For true positives - correlation between feature value and prediction probability
    if len(tp_data) > 1:
        tp_corr = np.corrcoef(tp_data[feature], tp_data['probability'])[0, 1]
        tp_prob_corr[feature] = tp_corr

    # For true negatives - correlation between feature value and prediction probability
    if len(tn_data) > 1:
        tn_corr = np.corrcoef(tn_data[feature], tn_data['probability'])[0, 1]
        tn_prob_corr[feature] = tn_corr

# Create correlation analysis dataframe
corr_df = pd.DataFrame({
    'TP_Prob_Corr': tp_prob_corr,
    'TN_Prob_Corr': tn_prob_corr
}).fillna(0)

corr_df['TP_abs_corr'] = abs(corr_df['TP_Prob_Corr'])
corr_df['TN_abs_corr'] = abs(corr_df['TN_Prob_Corr'])
corr_df = corr_df.sort_values('TP_abs_corr', ascending=False)

print("\nTop features correlated with prediction confidence for True Positives:")
print(corr_df[['TP_Prob_Corr', 'TP_abs_corr']].head(5))

# Save feature analysis results
feature_importance = {
    'top_differentiating_features': comparison_df.head(10)[['TP_mean', 'TN_mean', 'TP_vs_TN_diff']].to_dict(),
    'tp_probability_correlations': corr_df['TP_Prob_Corr'].to_dict(),
    'tn_probability_correlations': corr_df['TN_Prob_Corr'].to_dict(),
    'sample_sizes': {
        'true_positives': len(tp_data),
        'true_negatives': len(tn_data),
        'false_positives': len(fp_data),
        'false_negatives': len(fn_data)
    }
}

import json
with open('d:/code/WindSurfCode/BankTermDeposit/feature_analysis.json', 'w') as f:
    json.dump(feature_importance, f, indent=2)

print("\nFeature analysis saved to feature_analysis.json")
print("\nFeature analysis completed!")
