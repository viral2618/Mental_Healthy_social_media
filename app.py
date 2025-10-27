from flask import Flask, render_template, request
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load('Random_froest_Model_for_mental_health.pkl')
scaler = joblib.load('Scaler_for_mental_health.pkl')

app = Flask(__name__)
app.secret_key = 'my_secret'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Convert gender safely
        gender = 1 if data.get('Gender') == 'Male' else (2 if data.get('Gender') == 'Other' else 0)

        # Active user
        is_active = 1 if data.get('Is_Active_User') == 'Yes' else 0

        # Social media platform one-hot encoding
        platforms = ['Instagram', 'LinkedIn', 'TikTok', 'Twitter', 'YouTube']
        selected_platform = data.get('Social_Media_Platform', 'None')  # default to None
        if selected_platform not in platforms:
            platform_values = [0] * len(platforms)
        else:
            platform_values = [1 if selected_platform == p else 0 for p in platforms]

        # Helper function to safely convert to float
        def safe_float(value):
            if value is None or value.strip() == '':
                return 0.0
            return float(value.replace(',', '.'))

        # Collect features in the order your model expects
        feature = [
            safe_float(data.get('Age')),
            gender,
            safe_float(data.get('Daily_Screen_Time')),
            safe_float(data.get('Sleep_Quality')),
            safe_float(data.get('Stress_Level')),
            safe_float(data.get('Days_Without_Social_Media')),
            safe_float(data.get('Exercise_Frequency')),
            *platform_values,
            safe_float(data.get('Total_Social_Media_Use')),
            is_active,
            safe_float(data.get('Work_Life_Balance_Score')),
            safe_float(data.get('Rest_to_Screen_Ratio')),
            safe_float(data.get('Screen_Stress_Interaction')),
            safe_float(data.get('Stress_to_Sleep_Ratio')),
            safe_float(data.get('Screen_to_Exercise_Ratio'))
        ]

        # Convert to numpy array
        final_feature = np.array([feature])

        # Scale features if scaler exists
        if scaler:
            final_feature = scaler.transform(final_feature)

        # Predict
        prediction = model.predict(final_feature)[0]
        prediction = round(prediction, 2)

        # Categorize mental health based on happiness index
        if prediction < 6:
            status = 'Not Good Mental Health 😟'
        elif prediction < 9:
            status = 'Average Mental Health 🙂'
        else:
            status = 'Good Mental Health 😄'

        return render_template('index.html', prediction_text=f'Predicted Happiness: {prediction} → {status}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'⚠️ Error: {str(e)}')


if __name__ == '__main__':
    app.run(debug=True)
