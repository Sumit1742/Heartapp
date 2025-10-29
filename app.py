import streamlit as st
import numpy as np
import joblib

# Load pre-trained model and scaler
logreg = joblib.load('logreg_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Heart Disease Prediction")

# Friendly feature options/mappings
feature_options = {
    'sex': ('Sex', {'Male': 1, 'Female': 0}),
    'cp': ('Chest Pain Type', {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-Anginal Pain': 2,
        'Asymptomatic': 3
    }),
    'fbs': ('Fasting Blood Sugar > 120 mg/dl', {'Yes': 1, 'No': 0}),
    'restecg': ('Resting ECG Results', {
        'Normal': 0,
        'ST-T Wave Abnormality': 1,
        'Left Ventricular Hypertrophy': 2
    }),
    'exang': ('Exercise Induced Angina', {'Yes': 1, 'No': 0}),
    'slope': ('Slope of Peak Exercise ST Segment', {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }),
    'thal': ('Thalassemia', {
        'Normal': 1,
        'Fixed Defect': 2,
        'Reversible Defect': 3
    })
}

num_features = [
    ('Age', 'age'),
    ('Resting Blood Pressure (mm Hg)', 'trestbps'),
    ('Serum Cholestoral (mg/dl)', 'chol'),
    ('Maximum Heart Rate Achieved', 'thalach'),
    ('Oldpeak (ST depression by exercise)', 'oldpeak'),
    ('Number of Major Vessels (0-3)', 'ca') # slider
]

with st.form(key='input_form'):
    st.subheader("Enter your medical details (fill all fields)")

    user_input = []
    missing_input = False

    # Numeric Inputs
    for label, feat in num_features:
        if feat == 'ca':
            val = st.slider(label, 0, 3, 0, help="Number of major vessels colored by fluoroscopy (0-3)")
        elif feat in ['age', 'trestbps', 'chol', 'thalach']:
            val = st.number_input(label, value=0, step=1)
        else:
            val = st.number_input(label, value=0.0)
        user_input.append(val)


    # Categorical Inputs with 'Select...' placeholder
    for feat in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
        label, opts = feature_options[feat]
        options_list = ['Select...'] + list(opts.keys())
        chosen = st.selectbox(label, options_list)
        if chosen == 'Select...':
            missing_input = True
        else:
            user_input.append(opts[chosen])

    submit_button = st.form_submit_button(label="Predict Heart Disease Risk")

    if submit_button:
        if missing_input:
            st.error("Please fill all fields and select options for each category before predicting.")
        else:
            X_new = np.array(user_input).reshape(1, -1)
            X_new_scaled = scaler.transform(X_new)
            prediction = logreg.predict(X_new_scaled)[0]
            st.success(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
