from flask import Flask, request, jsonify
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and necessary preprocessing objects
model = joblib.load("C:/Users/mosta/OneDrive/Desktop/Work(ATW ltd)/app/credit_score_model.pkl")
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['age', 'annual_income', 'num_of_loan', 'num_of_delayed_payment',
                                      'num_credit_inquiries', 'amount_invested_monthly']),
        ('cat', categorical_transformer, ['occupation', 'credit_mix', 'payment_of_min_amount', 'payment_behaviour'])])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
                
        # Create a DataFrame from the received JSON data
        df = pd.DataFrame(data, index=[0])
                
        # Preprocess the data
        X_processed = preprocessor.transform(df)
                
        # Predict the credit score
        prediction = model.predict(X_processed)
                
        # Return the predicted credit score as JSON response
        return jsonify({'credit_score_prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8000)