import pandas as pd
import numpy as np
import joblib
import streamlit as st

class FraudDetectionModel:
    def __init__(self, model_path='fraud_detection_model.pkl'):
        self.model_path = model_path
        self.model = self._load_model()
        
    def _load_model(self):
        try:
            loaded_data = joblib.load(self.model_path)
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                model = loaded_data['model']
            else:
                model = loaded_data
            if not hasattr(model, 'predict_proba'):
                raise ValueError("Loaded object is not a valid model with predict_proba method")
            return model
        except FileNotFoundError:
            print(f"Model file not found at {self.model_path}")
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def preprocess_input(self, input_data):
        df = pd.DataFrame([input_data])
        
        # Ensure categorical columns are strings
        categorical_cols = ['card_type', 'location', 'purchase_category', 'transaction_description']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
            else:
                df[col] = 'Unknown'
        
        # Ensure numeric columns are float
        numeric_cols = ['customer_id', 'merchant_id', 'amount', 'customer_age', 'is_weekend',
                        'customer_avg_amount', 'customer_std_amount', 'customer_max_amount',
                        'customer_transaction_count', 'merchant_avg_amount', 'merchant_std_amount',
                        'merchant_max_amount', 'merchant_transaction_count', 'merchant_fraud_rate',
                        'amount_deviation', 'hour', 'day_of_week', 'month']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
            else:
                df[col] = 0.0
        
        return df
    
    def predict(self, input_data):
        if self.model is None:
            return {"error": "Model not loaded"}
        
        try:
            processed_data = self.preprocess_input(input_data)
            prob_scores = self.model.predict_proba(processed_data)
            fraud_prob = prob_scores[0][1]
            prediction = self.model.predict(processed_data)[0]
            
            return {
                "fraud_probability": float(fraud_prob),
                "is_fraud": bool(prediction)
            }
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def generate_sample_data(self, n_samples=5):
        np.random.seed(42)
        data = []
        for i in range(n_samples):
            amount = np.random.randint(10, 1000)
            card_types = ['Visa', 'Mastercard', 'Amex', 'Discover']
            locations = ['Online', 'In-store', 'Mobile', 'ATM']
            categories = ['Retail', 'Grocery', 'Travel', 'Entertainment', 'Restaurant']
            descriptions = ['Regular Purchase', 'Subscription', 'One-time Payment']
            
            if i % 2 == 0:
                is_fraud = 0
                amount = np.random.randint(10, 200)
                location = np.random.choice(['In-store', 'Mobile'])
            else:
                is_fraud = 1
                amount = np.random.randint(500, 2000)
                location = 'Online'
                
            transaction = {
                'customer_id': np.random.randint(10000, 99999),
                'merchant_id': np.random.randint(1000, 9999),
                'amount': amount,
                'is_fraudulent': is_fraud,
                'card_type': np.random.choice(card_types),
                'location': location,
                'purchase_category': np.random.choice(categories),
                'customer_age': np.random.randint(18, 85),
                'transaction_description': np.random.choice(descriptions),
                'is_weekend': np.random.randint(0, 2),
                'customer_avg_amount': np.random.uniform(50, 500),
                'customer_std_amount': np.random.uniform(10, 100),
                'customer_max_amount': np.random.uniform(100, 1000),
                'customer_transaction_count': np.random.randint(1, 50),
                'merchant_avg_amount': np.random.uniform(50, 500),
                'merchant_std_amount': np.random.uniform(10, 100),
                'merchant_max_amount': np.random.uniform(100, 1000),
                'merchant_transaction_count': np.random.randint(1, 100),
                'merchant_fraud_rate': np.random.uniform(0, 0.1),
                'amount_deviation': np.random.uniform(-100, 100),
                'hour': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'month': np.random.randint(1, 13)
            }
            data.append(transaction)
        
        return pd.DataFrame(data)

# Streamlit Interface
def main():
    st.title("Fraud Detection System")
    
    # Initialize model
    fraud_model = FraudDetectionModel()
    
    # Input form
    with st.form("transaction_form"):
        st.header("Enter Transaction Details")
        
        # Numeric inputs
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
        customer_id = st.number_input("Customer ID", min_value=10000, max_value=99999, value=10000)
        merchant_id = st.number_input("Merchant ID", min_value=1000, max_value=9999, value=1000)
        customer_age = st.number_input("Customer Age", min_value=18, max_value=85, value=30)
        
        # Categorical inputs
        card_type = st.selectbox("Card Type", ['Visa', 'Mastercard', 'Amex', 'Discover'])
        location = st.selectbox("Location", ['Online', 'In-store', 'Mobile', 'ATM'])
        purchase_category = st.selectbox("Purchase Category", ['Retail', 'Grocery', 'Travel', 'Entertainment', 'Restaurant'])
        transaction_description = st.selectbox("Transaction Description", ['Regular Purchase', 'Subscription', 'One-time Payment'])
        
        # Additional numeric inputs
        is_weekend = st.selectbox("Is Weekend", [0, 1])
        hour = st.number_input("Hour of Day", min_value=0, max_value=23, value=12)
        day_of_week = st.number_input("Day of Week (0-6)", min_value=0, max_value=6, value=3)
        month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)
        
        # Placeholder values for features not exposed in UI
        customer_avg_amount = st.number_input("Customer Avg Amount", min_value=50.0, max_value=500.0, value=100.0)
        customer_std_amount = st.number_input("Customer Std Amount", min_value=10.0, max_value=100.0, value=20.0)
        customer_max_amount = st.number_input("Customer Max Amount", min_value=100.0, max_value=1000.0, value=500.0)
        customer_transaction_count = st.number_input("Customer Transaction Count", min_value=1, max_value=50, value=10)
        merchant_avg_amount = st.number_input("Merchant Avg Amount", min_value=50.0, max_value=500.0, value=100.0)
        merchant_std_amount = st.number_input("Merchant Std Amount", min_value=10.0, max_value=100.0, value=20.0)
        merchant_max_amount = st.number_input("Merchant Max Amount", min_value=100.0, max_value=1000.0, value=500.0)
        merchant_transaction_count = st.number_input("Merchant Transaction Count", min_value=1, max_value=100, value=20)
        merchant_fraud_rate = st.number_input("Merchant Fraud Rate", min_value=0.0, max_value=0.1, value=0.01)
        amount_deviation = st.number_input("Amount Deviation", min_value=-100.0, max_value=100.0, value=0.0)
        
        # Submit button
        submitted = st.form_submit_button("Predict Fraud")
        
        if submitted:
            # Create input dictionary
            input_data = {
                'customer_id': customer_id,
                'merchant_id': merchant_id,
                'amount': amount,
                'card_type': card_type,
                'location': location,
                'purchase_category': purchase_category,
                'customer_age': customer_age,
                'transaction_description': transaction_description,
                'is_weekend': is_weekend,
                'customer_avg_amount': customer_avg_amount,
                'customer_std_amount': customer_std_amount,
                'customer_max_amount': customer_max_amount,
                'customer_transaction_count': customer_transaction_count,
                'merchant_avg_amount': merchant_avg_amount,
                'merchant_std_amount': merchant_std_amount,
                'merchant_max_amount': merchant_max_amount,
                'merchant_transaction_count': merchant_transaction_count,
                'merchant_fraud_rate': merchant_fraud_rate,
                'amount_deviation': amount_deviation,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month
            }
            
            # Make prediction
            result = fraud_model.predict(input_data)
            
            # Display results
            if "error" in result:
                st.error(result["error"])
            else:
                st.subheader("Prediction Results")
                st.write(f"Fraud Probability: {result['fraud_probability']:.2%}")
                st.write(f"Is Fraudulent: {'Yes' if result['is_fraud'] else 'No'}")
                
                # Visual indicator
                if result['is_fraud']:
                    st.error("⚠️ High Fraud Risk Detected!")
                else:
                    st.success("✅ Transaction Appears Safe")

if __name__ == "__main__":
    main()
