
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# 1. Load and prepare basic dataset
def load_simple_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()

    # Only keep basic useful columns
    keep_cols = [
        'customer_id', 'merchant_id', 'amount', 'card_type',
        'location', 'purchase_category', 'customer_age',
        'transaction_description', 'is_fraudulent'
    ]
    df = df[keep_cols]
    return df

# 2. Feature selection
def select_features(df):
    X = df.drop('is_fraudulent', axis=1)
    y = df['is_fraudulent']
    return X, y

# 3. Build and train a simple pipeline
def build_pipeline(X_train, X_test, y_train, y_test):
    numeric_features = ['customer_id', 'merchant_id', 'amount', 'customer_age']
    categorical_features = ['card_type', 'location', 'purchase_category', 'transaction_description']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return pipeline

# 4. Train and save pipeline
def detect_fraud_simple(file_path):
    print("Loading simple dataset...")
    df = load_simple_data(file_path)

    X, y = select_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training simple pipeline...")
    pipeline = build_pipeline(X_train, X_test, y_train, y_test)

    output_path = "fraud_detection_simple_pipeline.pkl"
    joblib.dump(pipeline, output_path, compress=1)
    print(f"Saved pipeline to: {os.path.abspath(output_path)}")

    return pipeline

# Run
pipeline = detect_fraud_simple('synthetic_financial_data.csv')
