import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

DATA_FILE = "raw_data.parquet"

def generate_data(num_rows=1000000):
    np.random.seed(42)
    
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
        "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
        "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
        "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi"
    ]
    
    # Generate dates over the last 90 days
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(90)]
    
    print(f"Generating {num_rows} records...")
    
    # Vectorized generation for extreme performance
    data = {
        'date': np.random.choice(dates, num_rows),
        'state': np.random.choice(states, num_rows),
        'enrollments': np.random.poisson(lam=50, size=num_rows),
        'updates': np.random.poisson(lam=20, size=num_rows),
        'biometric_failures': np.random.poisson(lam=2, size=num_rows),
        'document_rejections': np.random.poisson(lam=3, size=num_rows),
        'operator_id': np.random.randint(1000, 9999, size=num_rows),
        'registrar_id': np.random.randint(100, 999, size=num_rows)
    }
    
    df = pd.DataFrame(data)
    
    print("Injecting anomalies...")
    anomaly_indices = np.random.choice(df.index, size=int(num_rows * 0.05), replace=False)
    
    df['enrollments'] = df['enrollments'].astype(float)
    df['biometric_failures'] = df['biometric_failures'].astype(float)
    
    df.loc[anomaly_indices, 'enrollments'] *= np.random.uniform(3, 8, size=len(anomaly_indices))
    df.loc[anomaly_indices, 'biometric_failures'] *= np.random.uniform(5, 10, size=len(anomaly_indices))
    
    # Save as parquet for ultra-fast loading (100x faster than CSV)
    print("Saving to parquet...")
    df.to_parquet(DATA_FILE, engine='pyarrow')
    print(f"Data generation complete. Saved to {DATA_FILE}")

if __name__ == "__main__":
    generate_data()
