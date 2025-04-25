
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import joblib
import os

# 1. Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()
    return df

# 2. Data preprocessing and feature engineering
def preprocess_data(df):
    df_processed = df.copy()
    
    df_processed['transaction_time'] = pd.to_datetime(df_processed['transaction_time'])
    df_processed['hour'] = df_processed['transaction_time'].dt.hour
    df_processed['day_of_week'] = df_processed['transaction_time'].dt.dayofweek
    df_processed['month'] = df_processed['transaction_time'].dt.month
    df_processed['is_weekend'] = df_processed['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    customer_features = df_processed.groupby('customer_id').agg({
        'amount': ['mean', 'std', 'max'],
        'transaction_id': 'count'
    })
    customer_features.columns = ['customer_avg_amount', 'customer_std_amount', 'customer_max_amount', 'customer_transaction_count']
    customer_features = customer_features.reset_index()
    
    merchant_features = df_processed.groupby('merchant_id').agg({
        'amount': ['mean', 'std', 'max'],
        'transaction_id': 'count',
        'is_fraudulent': 'mean'
    })
    merchant_features.columns = ['merchant_avg_amount', 'merchant_std_amount', 'merchant_max_amount', 'merchant_transaction_count', 'merchant_fraud_rate']
    merchant_features = merchant_features.reset_index()
    
    df_processed = pd.merge(df_processed, customer_features, on='customer_id', how='left')
    df_processed = pd.merge(df_processed, merchant_features, on='merchant_id', how='left')
    
    df_processed['amount_deviation'] = abs(df_processed['amount'] - df_processed['customer_avg_amount']) / (df_processed['customer_std_amount'] + 1)
    
    return df_processed

# 3. Feature selection
def select_features(df_processed):
    drop_cols = ['transaction_id', 'transaction_time']
    X = df_processed.drop(drop_cols + ['is_fraudulent'], axis=1)
    y = df_processed['is_fraudulent']
    return X, y

# 4. Model building and evaluation
def build_and_evaluate_pipeline(X_train, X_test, y_train, y_test):
    numeric_features = ['customer_id', 'merchant_id', 'amount', 'customer_age', 'is_weekend',
                        'customer_avg_amount', 'customer_std_amount', 'customer_max_amount',
                        'customer_transaction_count', 'merchant_avg_amount', 'merchant_std_amount',
                        'merchant_max_amount', 'merchant_transaction_count', 'merchant_fraud_rate',
                        'amount_deviation', 'hour', 'day_of_week', 'month']
    categorical_features = ['card_type', 'location', 'purchase_category', 'transaction_description']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {roc_auc}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return pipeline

# 5. Main entry
def detect_fraud(file_path: str):
    print("Loading data...")
    df = load_data(file_path)
    
    print("\nPreprocessing data and engineering features...")
    df_processed = preprocess_data(df)
    
    X, y = select_features(df_processed)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining pipeline...")
    pipeline = build_and_evaluate_pipeline(X_train, X_test, y_train, y_test)
    
    # Save the full pipeline
    pipeline_file = 'fraud_detection_pipeline.pkl'
    joblib.dump(pipeline, pipeline_file, compress=1)
    print(f"\nSaved pipeline to: {os.path.abspath(pipeline_file)}")
    return pipeline

pipeline = detect_fraud('synthetic_financial_data.csv')
