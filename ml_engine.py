import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import os

DATA_FILE = "raw_data.parquet"
PROCESSED_FILE = "processed_data.parquet"

def process_data():
    print("Loading raw data...")
    if not os.path.exists(DATA_FILE):
        print("Raw data not found. Run data_engine.py first.")
        return
        
    df = pd.read_parquet(DATA_FILE)
    
    print("Aggregating data by state and date...")
    # Group by state and date to make the dataset manageable for real-time Streamlit charting
    daily_stats = df.groupby(['state', 'date']).agg({
        'enrollments': 'sum',
        'updates': 'sum',
        'biometric_failures': 'sum',
        'document_rejections': 'sum'
    }).reset_index()
    
    # Feature Engineering
    daily_stats['failure_rate'] = daily_stats['biometric_failures'] / (daily_stats['enrollments'] + 1)
    daily_stats['rejection_rate'] = daily_stats['document_rejections'] / (daily_stats['enrollments'] + 1)
    
    print("Running Machine Learning Models...")
    
    # Isolation Forest for Anomaly Detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    features = ['enrollments', 'failure_rate', 'rejection_rate']
    daily_stats['anomaly_score'] = iso.fit_predict(daily_stats[features])
    daily_stats['is_anomaly'] = daily_stats['anomaly_score'] == -1
    
    # XGBoost for Risk Forecasting
    # We will simulate a risk target based on historical patterns to train XGBoost
    np.random.seed(42)
    daily_stats['historical_risk_target'] = (
        (daily_stats['failure_rate'] * 0.4) + 
        (daily_stats['rejection_rate'] * 0.4) + 
        (daily_stats['is_anomaly'] * 0.2) + 
        np.random.normal(0, 0.05, len(daily_stats))
    ).clip(0, 1)
    
    dtrain = xgb.DMatrix(daily_stats[features], label=daily_stats['historical_risk_target'])
    params = {'objective': 'reg:squarederror', 'max_depth': 4, 'learning_rate': 0.1}
    xgb_model = xgb.train(params, dtrain, num_boost_round=50)
    
    daily_stats['forecasted_risk_score'] = xgb_model.predict(xgb.DMatrix(daily_stats[features]))
    daily_stats['risk_level'] = pd.cut(
        daily_stats['forecasted_risk_score'], 
        bins=[-np.inf, 0.3, 0.6, 0.8, np.inf], 
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    print("Saving processed data...")
    daily_stats.to_parquet(PROCESSED_FILE, engine='pyarrow')
    print(f"ML pipeline complete. Data saved to {PROCESSED_FILE}")

if __name__ == "__main__":
    process_data()
