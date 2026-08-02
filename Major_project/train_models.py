# train_models.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Ensure deployment directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

np.random.seed(42)
N = 1000  # Samples per dataset

print("Generating synthetic physiological data distributions...")

# --- 1. DIABETES DATASET ---
diabetes_data = pd.DataFrame({
    'age': np.random.randint(18, 80, N),
    'bmi': np.random.uniform(15, 45, N),
    'glucose': np.random.uniform(70, 250, N),
    'blood_pressure': np.random.uniform(60, 140, N)
})
# Ground truth logic: High BMI and high glucose drive higher probability
diabetes_prob = 1 / (1 + np.exp(-(-5 + 0.1 * diabetes_data['bmi'] + 0.03 * diabetes_data['glucose'])))
diabetes_data['target'] = (diabetes_prob > np.random.uniform(0, 1, N)).astype(int)
diabetes_data.to_csv('data/diabetes.csv', index=False)

# --- 2. HEART DISEASE DATASET ---
heart_data = pd.DataFrame({
    'age': np.random.randint(25, 80, N),
    'cholesterol': np.random.uniform(150, 400, N),
    'blood_pressure': np.random.uniform(90, 180, N),
    'smoking': np.random.choice([0, 1], N, p=[0.6, 0.4])
})
heart_prob = 1 / (1 + np.exp(-(-6 + 0.02 * heart_data['cholesterol'] + 0.02 * heart_data['blood_pressure'] + 1.5 * heart_data['smoking'])))
heart_data['target'] = (heart_prob > np.random.uniform(0, 1, N)).astype(int)
heart_data.to_csv('data/heart.csv', index=False)

# --- 3. LIVER DISEASE DATASET ---
liver_data = pd.DataFrame({
    'age': np.random.randint(18, 80, N),
    'alcohol': np.random.choice([0, 1], N, p=[0.7, 0.3]),
    'bilirubin': np.random.uniform(0.3, 5.0, N),
    'alkphos': np.random.uniform(50, 300, N)
})
liver_prob = 1 / (1 + np.exp(-(-4 + 2.0 * liver_data['alcohol'] + 0.5 * liver_data['bilirubin'] + 0.005 * liver_data['alkphos'])))
liver_data['target'] = (liver_prob > np.random.uniform(0, 1, N)).astype(int)
liver_data.to_csv('data/liver.csv', index=False)

# --- 4. KIDNEY FAILURE DATASET ---
kidney_data = pd.DataFrame({
    'age': np.random.randint(18, 80, N),
    'blood_pressure': np.random.uniform(60, 140, N),
    'glucose': np.random.uniform(70, 250, N),
    'creatinine': np.random.uniform(0.5, 6.0, N)
})
kidney_prob = 1 / (1 + np.exp(-(-5 + 0.01 * kidney_data['blood_pressure'] + 0.8 * kidney_data['creatinine'])))
kidney_data['target'] = (kidney_prob > np.random.uniform(0, 1, N)).astype(int)
kidney_data.to_csv('data/kidney.csv', index=False)

# --- TRAINING PIPELINE ---
datasets = {
    'diabetes': ('data/diabetes.csv', ['age', 'bmi', 'glucose', 'blood_pressure']),
    'heart': ('data/heart.csv', ['age', 'cholesterol', 'blood_pressure', 'smoking']),
    'liver': ('data/liver.csv', ['age', 'alcohol', 'bilirubin', 'alkphos']),
    'kidney': ('data/kidney.csv', ['age', 'blood_pressure', 'glucose', 'creatinine'])
}

for name, (path, features) in datasets.items():
    df = pd.read_csv(path)
    X = df[features]
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save optimized binary file
    joblib.dump(model, f'models/{name}_model.pkl')
    print(f"Successfully trained and exported models/{name}_model.pkl")

print("\nModel Training phase completed successfully!")