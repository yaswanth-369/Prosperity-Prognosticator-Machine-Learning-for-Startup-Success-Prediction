from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Feature names in the exact order expected by the model
FEATURE_NAMES = [
    'age_first_funding_year', 'age_last_funding_year', 'age_first_milestone_year',
    'age_last_milestone_year', 'funding_rounds', 'funding_total_usd', 'milestones',
    'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate', 'is_software', 'is_web',
    'is_mobile', 'is_enterprise', 'is_advertising', 'is_gamesvideo', 'is_ecommerce',
    'is_biotech', 'is_consulting', 'is_othercategory', 'has_VC', 'has_angel',
    'has_roundA', 'has_roundB', 'has_roundC', 'has_roundD', 'avg_participants',
    'is_top500', 'has_RoundABCD', 'has_Investor', 'has_both', 'invalid_startup',
    'age_startup_year', 'tier_relationships', 'State_CA', 'State_MA', 'State_NY',
    'State_TX', 'State_WA', 'State_other', 'category_advertising', 'category_biotech',
    'category_enterprise', 'category_games_video', 'category_hardware', 'category_mobile',
    'category_network_hosting', 'category_other', 'category_semiconductor',
    'category_software', 'category_web', 'founded_year_1984', 'founded_year_1985',
    'founded_year_1990', 'founded_year_1992', 'founded_year_1995', 'founded_year_1996',
    'founded_year_1997', 'founded_year_1998', 'founded_year_1999', 'founded_year_2000',
    'founded_year_2001', 'founded_year_2002', 'founded_year_2003', 'founded_year_2004',
    'founded_year_2005', 'founded_year_2006', 'founded_year_2007', 'founded_year_2008',
    'founded_year_2009', 'founded_year_2010', 'founded_year_2011', 'founded_year_2012',
    'founded_year_2013'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    
    try:
        features = []
        for feature in FEATURE_NAMES:
            value = request.form.get(feature, 0)
            if value == '':
                value = 0
            features.append(float(value))
        
        feature_array = np.array(features).reshape(1, -1)
        prediction = model.predict(feature_array)[0]
        prediction_proba = model.predict_proba(feature_array)[0]

        confidence_success = prediction_proba[1] * 100
        confidence_failure = prediction_proba[0] * 100

        if confidence_success >= 80:
            success_level = "Highly Likely to Succeed"
            color_class = "high-success"
        elif confidence_success >= 60:
            success_level = "Likely to Succeed"
            color_class = "medium-success"
        elif confidence_success >= 40:
            success_level = "Moderate Chance"
            color_class = "moderate-success"
        else:
            success_level = "Low Success Probability"
            color_class = "low-success"

        result = {
            'prediction': int(prediction),
            'confidence_success': round(confidence_success, 2),
            'confidence_failure': round(confidence_failure, 2),
            'success_level': success_level,
            'color_class': color_class
        }

        # 🔥 Add input data mapping here for use in results.html
        result['input_data'] = dict(zip(FEATURE_NAMES, features))

        return render_template('results.html', result=result)

    except Exception as e:
        return render_template('results.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        features = [data.get(feature, 0) for feature in FEATURE_NAMES]
        feature_array = np.array(features).reshape(1, -1)

        prediction = model.predict(feature_array)[0]
        prediction_proba = model.predict_proba(feature_array)[0]

        return jsonify({
            'prediction': int(prediction),
            'success_probability': round(prediction_proba[1] * 100, 2),
            'failure_probability': round(prediction_proba[0] * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/adaptivity')
def adaptivity():
    return render_template('adaptivity.html')

if __name__ == '__main__':
    app.run(debug=True)
