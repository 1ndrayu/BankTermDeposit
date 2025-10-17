import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

# Read cleaned dataset
df = pd.read_csv('d:/code/WindSurfCode/BankTermDeposit/bank_cleaned.csv')

# Prepare features and target
feature_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous',
                'job_encoded', 'marital_encoded', 'education_encoded', 'default_encoded',
                'housing_encoded', 'loan_encoded', 'contact_encoded', 'month_encoded', 'poutcome_encoded']

X = df[feature_cols]
y = df['y_encoded']

print("=== NEURAL NETWORK MODEL BUILDING (MLPClassifier) ===")
print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {np.bincount(y)}")

# Split the data (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Create and train MLP model
model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
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

print("\n=== TRAINING MODEL ===")
print("Model parameters:")
print(f"Hidden layers: {model.hidden_layer_sizes}")
print(f"Activation: {model.activation}")
print(f"Max iterations: {model.max_iter}")

# Train the model
model.fit(X_train, y_train)

# Training history (MLPClassifier doesn't provide detailed history like TensorFlow)
print(f"\nTraining completed in {model.n_iter_} iterations")
print(f"Training score: {model.score(X_train, y_train):.4f}")
print(f"Best validation score: {model.best_validation_score_:.4f}")

# Evaluate on test set
print("\n=== MODEL EVALUATION ===")
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

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

# Save model and results
import joblib
joblib.dump(model, 'd:/code/WindSurfCode/BankTermDeposit/final_model.pkl')
print("\nModel saved to final_model.pkl")

# Save evaluation results
results = {
    'test_accuracy': (tp + tn) / (tp + tn + fp + fn),
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'auc_roc': roc_auc_score(y_test, y_pred_prob),
    'confusion_matrix': conf_matrix.tolist(),
    'training_iterations': model.n_iter_,
    'training_score': model.score(X_train, y_train),
    'best_validation_score': model.best_validation_score_
}

with open('d:/code/WindSurfCode/BankTermDeposit/model_results.json', 'w') as f:
    json.dump(results, f)

print("Model results saved to model_results.json")
print("\nNeural network model training completed!")
