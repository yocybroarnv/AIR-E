"""
Sample Data Generator for AIR-E
Generates anonymized synthetic Aadhaar enrollment and update data
Matches UIDAI aggregate dataset structure
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# State to districts mapping (mirrored from geography.py to avoid circular imports)
STATE_DISTRICT_MAP = {
    'Bihar': ['Araria', 'Kishanganj', 'Purnia', 'Patna', 'Muzaffarpur', 'Madhubani'],
    'West Bengal': ['Kolkata', 'Cooch Behar', 'Jalpaiguri', 'Malda', 'Murshidabad', 'Nadia'],
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Thane', 'Nashik'],
    'Delhi': ['Delhi'],
    'Karnataka': ['Bangalore'],
    'Tamil Nadu': ['Chennai'],
    'Telangana': ['Hyderabad'],
    'Gujarat': ['Ahmedabad', 'Surat', 'Rajkot', 'Vadodara'],
    'Rajasthan': ['Jaipur'],
    'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Agra', 'Varanasi', 'Meerut', 'Ghaziabad', 'Noida', 'Faridabad'],
    'Jammu and Kashmir': ['Srinagar'],
    'Punjab': ['Amritsar', 'Ludhiana'],
    'Madhya Pradesh': ['Indore', 'Bhopal'],
    'Odisha': ['Visakhapatnam']
}

BORDER_DISTRICTS = {
    'Araria', 'Kishanganj', 'Purnia', 'Madhubani', 'Supaul', 
    'Sitamarhi', 'Sheohar', 'Muzaffarpur', 'East Champaran', 'West Champaran',
    'Cooch Behar', 'Jalpaiguri', 'Alipurduar', 'Malda', 'Murshidabad',
    'Nadia', 'North 24 Parganas', 'South 24 Parganas'
}

# Get all districts from state mapping
ALL_DISTRICTS = []
for districts in STATE_DISTRICT_MAP.values():
    ALL_DISTRICTS.extend(districts)

DISTRICTS = sorted(list(set(ALL_DISTRICTS)))

# Enrollment operators
OPERATORS = [f'OP{i:03d}' for i in range(1, 101)]

def generate_enrollment_data(months=12, districts=None, base_records=50000):
    """Generate synthetic enrollment data matching UIDAI structure
    Columns: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
    """
    if districts is None:
        districts = DISTRICTS
    
    data = []
    start_date = datetime.now() - timedelta(days=months*30)
    
    for month in range(months):
        month_date = start_date + timedelta(days=month*30)
        
        for district in districts:
            # Base records per district per month (aggregated format)
            base_count = int(np.random.poisson(base_records / len(districts)))
            
            # Border districts show higher adult enrollment spikes (risk pattern)
            is_border = district in BORDER_DISTRICTS
            if is_border:
                adult_multiplier = np.random.uniform(1.5, 2.5)  # 1.5-2.5x higher adult enrollments
            else:
                adult_multiplier = np.random.uniform(0.8, 1.2)
            
            # Age group distribution
            age_0_5 = int(base_count * np.random.uniform(0.08, 0.12))
            age_5_17 = int(base_count * np.random.uniform(0.15, 0.25))
            age_18_greater = int(base_count * np.random.uniform(0.65, 0.77) * adult_multiplier)
            
            # Get state for district
            state = None
            for st, dists in STATE_DISTRICT_MAP.items():
                if district in dists:
                    state = st
                    break
            
            # Generate multiple records per month (daily aggregates)
            records_per_month = np.random.randint(20, 30)  # ~daily aggregates
            
            for day in range(records_per_month):
                record_date = month_date + timedelta(days=day)
                daily_factor = np.random.uniform(0.8, 1.2)
                
                data.append({
                    'date': record_date.strftime('%d %m %Y'),  # DD MM YYYY format
                    'state': state or 'Unknown',
                    'district': district,
                    'pincode': f"{np.random.randint(100000, 999999)}",
                    'age_0_5': max(0, int(age_0_5 * daily_factor / records_per_month)),
                    'age_5_17': max(0, int(age_5_17 * daily_factor / records_per_month)),
                    'age_18_greater': max(0, int(age_18_greater * daily_factor / records_per_month)),
                })
    
    df = pd.DataFrame(data)
    return df

def generate_demographic_updates(enrollment_df, months=12):
    """Generate demographic update data (name, address, DOB updates)
    Matches UIDAI structure: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
    """
    updates = []
    start_date = datetime.now() - timedelta(days=months*30)
    
    # Get unique district-state pairs
    district_states = enrollment_df[['district', 'state']].drop_duplicates()
    
    for month in range(months):
        month_date = start_date + timedelta(days=month*30)
        
        for _, row in district_states.iterrows():
            district = row['district']
            state = row['state']
            
            # Border districts show higher update churn (risk pattern)
            is_border = district in BORDER_DISTRICTS
            base_updates = np.random.poisson(2000 / len(district_states))
            
            if is_border:
                base_updates = int(base_updates * np.random.uniform(1.5, 2.0))
            
            # Daily aggregates
            records_per_month = np.random.randint(20, 30)
            
            for day in range(records_per_month):
                record_date = month_date + timedelta(days=day)
                daily_factor = np.random.uniform(0.8, 1.2)
                
                # Age distribution for updates (mostly adults)
                total_updates = int(base_updates * daily_factor / records_per_month)
                age_0_5 = max(0, int(total_updates * 0.05))
                age_5_17 = max(0, int(total_updates * 0.15))
                age_18_greater = max(0, total_updates - age_0_5 - age_5_17)
                
                updates.append({
                    'date': record_date.strftime('%d %m %Y'),
                    'state': state,
                    'district': district,
                    'pincode': f"{np.random.randint(100000, 999999)}",
                    'age_0_5': age_0_5,
                    'age_5_17': age_5_17,
                    'age_18_greater': age_18_greater,
                })
    
    update_df = pd.DataFrame(updates)
    return update_df

def generate_biometric_updates(enrollment_df, months=12):
    """Generate biometric update data (fingerprint, iris, face re-verification)
    Matches UIDAI structure: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
    """
    updates = []
    start_date = datetime.now() - timedelta(days=months*30)
    
    district_states = enrollment_df[['district', 'state']].drop_duplicates()
    
    for month in range(months):
        month_date = start_date + timedelta(days=month*30)
        
        for _, row in district_states.iterrows():
            district = row['district']
            state = row['state']
            
            # Biometric updates are typically lower than demographic
            base_updates = np.random.poisson(500 / len(district_states))
            
            # Border districts may have lower biometric updates (document risk signal)
            is_border = district in BORDER_DISTRICTS
            if is_border:
                base_updates = int(base_updates * np.random.uniform(0.6, 0.8))
            
            records_per_month = np.random.randint(15, 25)
            
            for day in range(records_per_month):
                record_date = month_date + timedelta(days=day)
                daily_factor = np.random.uniform(0.8, 1.2)
                
                total_updates = int(base_updates * daily_factor / records_per_month)
                age_0_5 = max(0, int(total_updates * 0.05))
                age_5_17 = max(0, int(total_updates * 0.15))
                age_18_greater = max(0, total_updates - age_0_5 - age_5_17)
                
                updates.append({
                    'date': record_date.strftime('%d %m %Y'),
                    'state': state,
                    'district': district,
                    'pincode': f"{np.random.randint(100000, 999999)}",
                    'age_0_5': age_0_5,
                    'age_5_17': age_5_17,
                    'age_18_greater': age_18_greater,
                })
    
    update_df = pd.DataFrame(updates)
    return update_df

def parse_date_column(df, date_col='date'):
    """Parse date column from DD MM YYYY format"""
    def parse_date(date_str):
        try:
            parts = str(date_str).strip().split()
            if len(parts) == 3:
                day, month, year = parts
                return pd.to_datetime(f"{year}-{month.zfill(2)}-{day.zfill(2)}")
        except:
            pass
        return pd.NaT
    
    df = df.copy()
    df['parsed_date'] = df[date_col].apply(parse_date)
    df['month'] = df['parsed_date'].dt.to_period('M')
    df['year_month'] = df['parsed_date'].dt.strftime('%Y-%m')
    df['date_dt'] = df['parsed_date']
    return df

def generate_aggregate_metrics(enrollment_df, demographic_df=None, biometric_df=None):
    """Generate district-level aggregated metrics from UIDAI format data"""
    # Parse dates
    enrollment_df = parse_date_column(enrollment_df)
    if demographic_df is not None and not demographic_df.empty:
        demographic_df = parse_date_column(demographic_df)
    if biometric_df is not None and not biometric_df.empty:
        biometric_df = parse_date_column(biometric_df)
    
    metrics = []
    
    # Group by district and month
    for (district, month), group in enrollment_df.groupby(['district', 'year_month']):
        # Enrollment totals by age group
        total_enrollments = (group['age_0_5'].sum() + 
                           group['age_5_17'].sum() + 
                           group['age_18_greater'].sum())
        
        age_0_5 = group['age_0_5'].sum()
        age_5_17 = group['age_5_17'].sum()
        age_18_greater = group['age_18_greater'].sum()
        
        # Get state
        state = group['state'].iloc[0] if 'state' in group.columns else 'Unknown'
        
        # Demographic updates
        demo_updates = 0
        if demographic_df is not None and not demographic_df.empty:
            demo_group = demographic_df[
                (demographic_df['district'] == district) & 
                (demographic_df['year_month'] == month)
            ]
            demo_updates = (demo_group['age_0_5'].sum() + 
                          demo_group['age_5_17'].sum() + 
                          demo_group['age_18_greater'].sum())
        
        # Biometric updates
        bio_updates = 0
        if biometric_df is not None and not biometric_df.empty:
            bio_group = biometric_df[
                (biometric_df['district'] == district) & 
                (biometric_df['year_month'] == month)
            ]
            bio_updates = (bio_group['age_0_5'].sum() + 
                         bio_group['age_5_17'].sum() + 
                         bio_group['age_18_greater'].sum())
        
        metrics.append({
            'district': district,
            'state': state,
            'year_month': month,
            'month': pd.to_datetime(f"{month}-01"),
            'total_enrollments': total_enrollments,
            'age_0_5': age_0_5,
            'age_5_17': age_5_17,
            'age_18_greater': age_18_greater,
            'demographic_updates': demo_updates,
            'biometric_updates': bio_updates,
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Add geographic features (simple version to avoid circular import)
    from utils.geography import add_geographic_features
    metrics_df = add_geographic_features(metrics_df)
    
    # Calculate enrollment rate changes
    for district in metrics_df['district'].unique():
        district_data = metrics_df[metrics_df['district'] == district].sort_values('month')
        if len(district_data) > 1:
            district_data = district_data.copy()
            district_data['enrollment_rate_change'] = district_data['total_enrollments'].pct_change().fillna(0)
            metrics_df.loc[metrics_df['district'] == district, 'enrollment_rate_change'] = \
                district_data['enrollment_rate_change'].values
    
    return metrics_df
