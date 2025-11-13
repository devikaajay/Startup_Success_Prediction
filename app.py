
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for

# --- Configuration and Initialization ---
app = Flask(__name__)
MODEL_PATH = 'model.joblib'
FEATURES_PATH = 'features.json'
FILE_NAME = 'prepared_for_model.csv' # Added for loading data in app.py

# Global variables to hold the loaded model and feature list
model = None
feature_names = None

# Global variables for dropdown options, loaded once
unique_categories = []
unique_countries = []
unique_states = []
unique_statuses = []

# List of the original categorical columns that need OHE
OHE_COLS = ['category_list', 'country_code', 'state_code', 'status']

def load_assets():
    # Load the trained model and feature names list.
    global model, feature_names
    global unique_categories, unique_countries, unique_states, unique_statuses # Declare globals

    # 1. Load the Model
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}. Ensure you have uploaded 'model.joblib'.")
        model = None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

    # 2. Load the Feature Names (from training data)
    try:
        with open(FEATURES_PATH, 'r') as f:
            feature_names = json.load(f)
        print(f"✅ Features list loaded successfully. Total OHE features: {len(feature_names)}")
    except FileNotFoundError:
        print(f"❌ Error: Features file not found at {FEATURES_PATH}. Ensure you have uploaded 'features.json'.")
        feature_names = []
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        feature_names = []

    # 3. Load data for dropdowns (this will run once when app starts)
    try:
        df_for_dropdowns = pd.read_csv(FILE_NAME)
        unique_categories = sorted(df_for_dropdowns['category_list'].dropna().unique().tolist())
        unique_countries = sorted(df_for_dropdowns['country_code'].dropna().unique().tolist())
        unique_states = sorted(df_for_dropdowns['state_code'].dropna().unique().tolist())
        unique_statuses = sorted(df_for_dropdowns['status'].dropna().unique().tolist())
        print(f"✅ Dropdown values loaded successfully from '{FILE_NAME}'. Found {len(unique_categories)} categories, {len(unique_countries)} countries, {len(unique_states)} states, {len(unique_statuses)} statuses.")
    except FileNotFoundError:
        print(f"❌ Error: '{FILE_NAME}' not found for dropdowns. Please ensure it exists.")
    except Exception as e:
        print(f"❌ Error loading dropdown data: {e}")

load_assets()

@app.route('/')
def cover():
    return render_template('cover.html')

@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Renders the main form and handles prediction on POST request.

    if model is None or not feature_names:
        return render_template('predict.html', error="Model assets are missing or failed to load. Please check the console output.")

    prediction_result = None
    error_message = None

    if request.method == 'POST':
        try:
            raw_input = {
                'category_list': request.form['category_list'],
                'country_code': request.form['country_code'],
                'state_code': request.form['state_code'],
                'funding_rounds': int(request.form['funding_rounds']),
                'status': request.form['status'],
                'funding_total_usd': float(request.form['funding_total_usd']) # New direct numeric input
            }

            input_df = pd.DataFrame([raw_input])

            # Apply one-hot encoding to categorical features
            input_encoded = pd.get_dummies(input_df, columns=OHE_COLS, drop_first=False)

            # Reindex to match the columns from training data, filling missing with 0
            # Ensure numerical columns are not accidentally encoded or dropped
            X_new = input_encoded.reindex(columns=feature_names, fill_value=0)

            prediction_class = model.predict(X_new)[0]

            probabilities = model.predict_proba(X_new)[0]
            class_labels = model.classes_
            prob_map = dict(zip(class_labels, probabilities))

            predicted_probability = prob_map[prediction_class] * 100

            prediction_result = {
                'class': int(prediction_class),
                'probability': f"{predicted_probability:.2f}%",
                'full_probs': {int(k): f"{v*100:.2f}%" for k, v in prob_map.items()}
            }

        except ValueError:
            error_message = "Invalid input. Please ensure 'Funding Rounds' and 'Funding Total (USD)' are numbers and all fields are filled."
        except Exception as e:
            print(f"An unexpected error occurred during prediction: {e}")
            error_message = f"Prediction failed due to an internal error: {e}"

    return render_template(
        'predict.html',
        unique_categories=unique_categories,
        unique_countries=unique_countries,
        unique_states=unique_states,
        unique_statuses=unique_statuses,
        prediction_result=prediction_result,
        error=error_message,
        form_values=request.form
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
