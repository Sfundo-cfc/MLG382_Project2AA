import pandas as pd
import numpy as np
import joblib

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