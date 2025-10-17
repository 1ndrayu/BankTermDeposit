import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import joblib

# Read cleaned dataset
df = pd.read_csv('d:/code/WindSurfCode/BankTermDeposit/bank_cleaned.csv')

# Select only the 5 features used in the web application
selected_features = ['duration', 'pdays', 'job_encoded', 'contact_encoded', 'previous']

print("=== REDUCED MODEL TRAINING (5 FEATURES ONLY) ===")
print(f"Selected features: {selected_features}")

X = df[selected_features]
y = df['y_encoded']

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {np.bincount(y)}")

# Split the data (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train MLP model with scaled data
model = MLPClassifier(
    hidden_layer_sizes=(32, 16, 8),  # Smaller architecture for fewer features
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

print("\n=== TRAINING REDUCED MODEL ===")
print("Model parameters:")
print(f"Hidden layers: {model.hidden_layer_sizes}")
print(f"Activation: {model.activation}")
print(f"Max iterations: {model.max_iter}")

# Train the model
model.fit(X_train_scaled, y_train)

# Training history
print(f"\nTraining completed in {model.n_iter_} iterations")
print(f"Training score: {model.score(X_train_scaled, y_train):.4f}")
print(f"Best validation score: {model.best_validation_score_:.4f}")

# Evaluate on test set
print("\n=== MODEL EVALUATION ===")
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Calculate additional metrics
tn, fp, fn, tp = conf_matrix.ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\nAdditional Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.4f}")

# Save model, scaler, and results
joblib.dump(model, 'd:/code/WindSurfCode/BankTermDeposit/reduced_model.pkl')
joblib.dump(scaler, 'd:/code/WindSurfCode/BankTermDeposit/reduced_scaler.pkl')
print("\nModel saved to reduced_model.pkl")
print("Scaler saved to reduced_scaler.pkl")

# Save evaluation results for reduced model
results = {
    'test_accuracy': (tp + tn) / (tp + tn + fp + fn),
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'auc_roc': roc_auc_score(y_test, y_pred_prob),
    'confusion_matrix': conf_matrix.tolist(),
    'training_iterations': model.n_iter_,
    'training_score': model.score(X_train_scaled, y_train),
    'best_validation_score': model.best_validation_score_,
    'selected_features': selected_features
}

with open('d:/code/WindSurfCode/BankTermDeposit/reduced_model_results.json', 'w') as f:
    json.dump(results, f)

print("Reduced model results saved to reduced_model_results.json")
print("\nReduced model training completed!")

# Compare with original model performance
print("\n=== PERFORMANCE COMPARISON ===")
with open('model_results.json', 'r') as f:
    original_results = json.load(f)

print("Original Model (16 features):")
print(f"  Accuracy: {original_results['test_accuracy']:.4f}")
print(f"  Precision: {original_results['precision']:.4f}")
print(f"  Recall: {original_results['recall']:.4f}")
print(f"  F1-Score: {original_results['f1_score']:.4f}")
print(f"  AUC-ROC: {original_results['auc_roc']:.4f}")

print("\nReduced Model (5 features):")
print(f"  Accuracy: {results['test_accuracy']:.4f}")
print(f"  Precision: {results['precision']:.4f}")
print(f"  Recall: {results['recall']:.4f}")
print(f"  F1-Score: {results['f1_score']:.4f}")
print(f"  AUC-ROC: {results['auc_roc']:.4f}")

accuracy_diff = results['test_accuracy'] - original_results['test_accuracy']
print(f"\nAccuracy difference: {accuracy_diff:.4f} ({'better' if accuracy_diff > 0 else 'worse'})")
