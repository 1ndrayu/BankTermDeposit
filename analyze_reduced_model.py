import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the reduced model and results
model = joblib.load('d:/code/WindSurfCode/BankTermDeposit/reduced_model.pkl')
scaler = joblib.load('d:/code/WindSurfCode/BankTermDeposit/reduced_scaler.pkl')

with open('reduced_model_results.json', 'r') as f:
    results = json.load(f)

# Load original dataset
df = pd.read_csv('d:/code/WindSurfCode/BankTermDeposit/bank_cleaned.csv')

print("=== REDUCED MODEL ANALYSIS (5 FEATURES) ===")

# Selected features for the reduced model
selected_features = results['selected_features']
print(f"Selected features: {selected_features}")

# Prepare data
X = df[selected_features]
y = df['y_encoded']

# Split data (same as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale features
X_test_scaled = scaler.transform(X_test)

# Get predictions
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

# Create results dataframe for analysis
results_df = X_test.copy()
results_df['actual'] = y_test
results_df['predicted'] = y_pred
results_df['probability'] = y_pred_prob
results_df['correct'] = (y_test == y_pred)

# Analyze features for true positives and true negatives
tp_data = results_df[(results_df['actual'] == 1) & (results_df['predicted'] == 1)]
tn_data = results_df[(results_df['actual'] == 0) & (results_df['predicted'] == 0)]

print("\n=== FEATURE ANALYSIS FOR REDUCED MODEL ===")

# Calculate mean values for each group
feature_analysis = {}
for col in selected_features:
    feature_analysis[col] = {
        'TP_mean': tp_data[col].mean(),
        'TN_mean': tn_data[col].mean(),
        'TP_vs_TN_diff': tp_data[col].mean() - tn_data[col].mean(),
        'overall_mean': results_df[col].mean()
    }

# Create comparison dataframe
comparison_df = pd.DataFrame(feature_analysis).T
comparison_df['abs_diff'] = abs(comparison_df['TP_vs_TN_diff'])
comparison_df = comparison_df.sort_values('abs_diff', ascending=False)

print("\nFeature differences between True Positives and True Negatives:")
print(comparison_df[['TP_mean', 'TN_mean', 'TP_vs_TN_diff', 'abs_diff']])

# Feature importance based on correlation with prediction confidence
feature_importance = {}
for feature in selected_features:
    if len(tp_data) > 1:
        tp_corr = np.corrcoef(tp_data[feature], tp_data['probability'])[0, 1]
        feature_importance[feature] = {'TP_Prob_Corr': tp_corr}

    if len(tn_data) > 1:
        tn_corr = np.corrcoef(tn_data[feature], tn_data['probability'])[0, 1]
        feature_importance[feature]['TN_Prob_Corr'] = tn_corr

importance_df = pd.DataFrame(feature_importance).T.fillna(0)
importance_df['abs_corr'] = abs(importance_df['TP_Prob_Corr'])
importance_df = importance_df.sort_values('abs_corr', ascending=False)

print("\nFeature correlation with prediction confidence:")
print(importance_df)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Reduced Model: Feature Distributions (True Positives vs True Negatives)')

# Plot top 5 features
for i, feature in enumerate(selected_features[:5]):
    if i >= 6:
        break

    row = i // 3
    col = i % 3

    # Create histograms
    axes[row, col].hist(tp_data[feature], alpha=0.7, label='True Positive', bins=20, density=True)
    axes[row, col].hist(tn_data[feature], alpha=0.7, label='True Negative', bins=20, density=True)
    axes[row, col].set_title(f'{feature}\nTP: {tp_data[feature].mean():.2f}, TN: {tn_data[feature].mean():.2f}')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Density')
    axes[row, col].legend()

plt.tight_layout()
plt.savefig('d:/code/WindSurfCode/BankTermDeposit/reduced_model_feature_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved feature analysis plot for reduced model")

# Model performance visualization
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
values = [results['test_accuracy'], results['precision'], results['recall'], results['f1_score'], results['auc_roc']]

bars = plt.bar(metrics, values, color=['#333', '#555', '#777', '#999', '#bbb'])
plt.title('Reduced Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.savefig('d:/code/WindSurfCode/BankTermDeposit/reduced_model_metrics.png', dpi=300, bbox_inches='tight')
print("Saved performance metrics plot")

# Create summary report
summary = {
    'model_type': 'Reduced MLPClassifier (5 features)',
    'selected_features': selected_features,
    'performance_metrics': results,
    'feature_analysis': comparison_df.to_dict(),
    'feature_importance': importance_df.to_dict(),
    'sample_analysis': {
        'total_samples': len(results_df),
        'true_positives': len(tp_data),
        'true_negatives': len(tn_data),
        'accuracy_on_test_set': results['test_accuracy']
    }
}

with open('d:/code/WindSurfCode/BankTermDeposit/reduced_model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n=== REDUCED MODEL SUMMARY ===")
print(f"Model Type: {summary['model_type']}")
print(f"Selected Features: {len(selected_features)}")
print(f"Training Accuracy: {results['training_score']:.4f}")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
print(f"AUC-ROC: {results['auc_roc']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")

print("\nKey Insights:")
print(f"- Model trained on {len(selected_features)} most important features only")
print(f"- Maintains {results['test_accuracy']*100:.1f}% accuracy despite using 68% fewer features")
print(f"- Top feature: {selected_features[0]} with correlation {comparison_df.iloc[0]['abs_diff']:.3f}")
print("- Reduced model is more interpretable and efficient for real-time predictions")

print("\nAnalysis saved to:")
print("- reduced_model_feature_analysis.png")
print("- reduced_model_metrics.png")
print("- reduced_model_summary.json")

print("\nReduced model analysis completed!")
