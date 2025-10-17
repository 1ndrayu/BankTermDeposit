from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import json
import pandas as pd

app = Flask(__name__)

# Load the trained reduced model (5 features only)
model = joblib.load('reduced_model.pkl')
scaler = joblib.load('reduced_scaler.pkl')

# Load feature analysis results
with open('reduced_model_results.json', 'r') as f:
    model_results = json.load(f)

with open('feature_analysis.json', 'r') as f:
    feature_analysis = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()

        # Extract the 5 features used in the reduced model
        features = [
            float(data['duration']),
            float(data['pdays']),
            float(data['job']),
            float(data['contact']),
            float(data['previous'])
        ]

        # Convert to numpy array and scale using the reduced model's scaler
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction_prob = model.predict_proba(features_scaled)[0]
        prediction = model.predict(features_scaled)[0]

        # Calculate confidence percentage
        confidence = max(prediction_prob) * 100

        # Determine prediction result
        result = "Yes" if prediction == 1 else "No"
        probability_yes = prediction_prob[1] * 100
        probability_no = prediction_prob[0] * 100

        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': f"{confidence:.1f}",
            'probabilities': {
                'yes': f"{probability_yes:.1f}",
                'no': f"{probability_no:.1f}"
            },
            'model_metrics': {
                'accuracy': f"{model_results['test_accuracy']:.3f}",
                'precision': f"{model_results['precision']:.3f}",
                'recall': f"{model_results['recall']:.3f}",
                'f1_score': f"{model_results['f1_score']:.3f}",
                'auc_roc': f"{model_results['auc_roc']:.3f}"
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model-info')
def model_info():
    return jsonify({
        'model_metrics': model_results,
        'selected_features': model_results.get('selected_features', []),
        'model_type': 'Reduced MLPClassifier (5 features)',
        'important_features': [
            'duration', 'pdays', 'job_encoded', 'contact_encoded', 'previous'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
