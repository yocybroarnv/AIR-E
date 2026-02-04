"""
Configuration file for AIR-E
"""
# Data generation parameters
DATA_CONFIG = {
    'months': 12,
    'base_records_per_district': 10000,
    'high_risk_districts': ['Mumbai', 'Delhi', 'Bangalore'],
    'risk_multiplier': 1.2
}

# Model parameters
MODEL_CONFIG = {
    'anomaly_contamination': 0.15,
    'risk_threshold_percentile': 75,
    'trend_window': 3
}

# Risk scoring weights
RISK_WEIGHTS = {
    'anomaly': 0.4,
    'forecast': 0.4,
    'trend': 0.2
}

# Risk categories
RISK_CATEGORIES = {
    'Critical': {'min': 0.85, 'priority': 5},
    'High': {'min': 0.7, 'priority': 4},
    'Medium': {'min': 0.5, 'priority': 3},
    'Low': {'min': 0.3, 'priority': 2},
    'Very Low': {'min': 0.0, 'priority': 1}
}
