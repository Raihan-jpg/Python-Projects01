# backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import shap
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests from frontend UI

# Load predictive model models globally
import os

# This automatically finds the absolute path of your multi_disease_system root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Go up one more level to reach Major_project where the models folder actually lives
MAJOR_PROJECT_DIR = os.path.dirname(BASE_DIR)

models = {
    'diabetes': joblib.load(os.path.join(MAJOR_PROJECT_DIR, 'models', 'diabetes_model.pkl')),
    'heart': joblib.load(os.path.join(MAJOR_PROJECT_DIR, 'models', 'heart_model.pkl')),
    'liver': joblib.load(os.path.join(MAJOR_PROJECT_DIR, 'models', 'liver_model.pkl')),
    'kidney': joblib.load(os.path.join(MAJOR_PROJECT_DIR, 'models', 'kidney_model.pkl'))
}# Define model feature maps explicitly
feature_keys = {
    'diabetes': ['age', 'bmi', 'glucose', 'blood_pressure'],
    'heart': ['age', 'cholesterol', 'blood_pressure', 'smoking'],
    'liver': ['age', 'alcohol', 'bilirubin', 'alkphos'],
    'kidney': ['age', 'blood_pressure', 'glucose', 'creatinine']
}

SPECIALIST_MATRIX = {
    'diabetes': {'doctor': 'Endocrinologist', 'desc': 'Specialist in metabolic systems and regulatory hormone therapy.'},
    'heart': {'doctor': 'Cardiologist', 'desc': 'Specialist focusing entirely on structural anomalies and vascular health.'},
    'liver': {'doctor': 'Hepatologist', 'desc': 'Specialist targeting progressive hepatic and gallbladder degradation.'},
    'kidney': {'doctor': 'Nephrologist', 'desc': 'Specialist handling systemic renal filtration and chronic failure mitigation.'}
}

def generate_shap_explanations(model, feature_names, input_data):
    """Calculates feature impact arrays using localized linear tree models."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    # Handle random forest output dimensions seamlessly
    if isinstance(shap_values, list):
        # Multi-output model check
        vals = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        vals = shap_values[0, :, 1]
    else:
        vals = shap_values[0]
        
    explanation_map = {}
    for feature, val in zip(feature_names, vals):
        explanation_map[feature] = round(float(val), 4)
    return explanation_map

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Standardize types cleanly from payload
    age = float(data.get('age', 30))
    bmi = float(data.get('bmi', 22))
    glucose = float(data.get('glucose', 90))
    blood_pressure = float(data.get('blood_pressure', 120))
    cholesterol = float(data.get('cholesterol', 180))
    smoking = int(data.get('smoking', 0))
    alcohol = int(data.get('alcohol', 0))
    bilirubin = float(data.get('bilirubin', 0.8))
    alkphos = float(data.get('alkphos', 80))
    creatinine = float(data.get('creatinine', 0.9))

    # Map raw entries to structured sub-arrays
    inputs = {
        'diabetes': pd.DataFrame([[age, bmi, glucose, blood_pressure]], columns=feature_keys['diabetes']),
        'heart': pd.DataFrame([[age, cholesterol, blood_pressure, smoking]], columns=feature_keys['heart']),
        'liver': pd.DataFrame([[age, alcohol, bilirubin, alkphos]], columns=feature_keys['liver']),
        'kidney': pd.DataFrame([[age, blood_pressure, glucose, creatinine]], columns=feature_keys['kidney'])
    }

    results = {}
    recommended_doctors = []

    for disease, model in models.items():
        input_df = inputs[disease]
        
        # Pull quantitative probability metrics
        prob = model.predict_proba(input_df)[0][1]
        percentage = round(float(prob) * 100, 2)
        
        shap_exp = generate_shap_explanations(model, feature_keys[disease], input_df)
        
        results[disease] = {
            'risk_percentage': percentage,
            'shap_values': shap_exp
        }
        
        # Clinical Risk Threshold Check
        if percentage >= 50.0:
            recommended_doctors.append({
                'disease': disease.capitalize(),
                'specialist': SPECIALIST_MATRIX[disease]['doctor'],
                'reason': SPECIALIST_MATRIX[disease]['desc']
            })
            
    return jsonify({
        'predictions': results,
        'referrals': recommended_doctors
    })

@app.route('/api/download-report', methods=['POST'])
def download_report():
    data = request.json
    predictions = data.get('predictions', {})
    referrals = data.get('referrals', [])
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=20, textColor='#1e3a8a')
    h2_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'], fontSize=16, spaceBefore=15, spaceAfter=10, textColor='#0f172a')
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=11, leading=16, spaceAfter=6)
    alert_style = ParagraphStyle('AlertBody', parent=body_style, textColor='#b91c1c', fontName="Helvetica-Bold")

    story = []
    story.append(Paragraph("Clinical Diagnostics Summary Report", title_style))
    story.append(Paragraph("Automated Screening Results Generated by AI Engine.", body_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("System Diagnostic Profiles", h2_style))
    for disease, metrics in predictions.items():
        txt = f"<b>{disease.capitalize()} Risk Evaluation:</b> {metrics['risk_percentage']}% probability score."
        story.append(Paragraph(txt, body_style))
        
        # Add SHAP metrics summary to PDF
        shap_txt = "<i>Feature Weights Impacting Score:</i> " + ", ".join([f"{k}: {v}" for k, v in metrics['shap_values'].items()])
        story.append(Paragraph(shap_txt, styles['Italic']))
        story.append(Spacer(1, 8))
        
    story.append(Spacer(1, 10))
    story.append(Paragraph("Directed Clinical Referrals", h2_style))
    
    if referrals:
        for ref in referrals:
            txt = f"⚠️ <b>Elevated risk detected for {ref['disease']}:</b> Immediate clinical consultation advised with a <b>{ref['specialist']}</b> ({ref['reason']})."
            story.append(Paragraph(txt, alert_style))
            story.append(Spacer(1, 5))
    else:
        story.append(Paragraph("All evaluated vital matrices reside within standard diagnostic limits. Maintain typical preventive checkups.", body_style))
        
    doc.build(story)
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name='Diagnostic_Health_Summary.pdf', mimetype='application/pdf')

if __name__ == '__main__':
    app.run(port=5000, debug=True)