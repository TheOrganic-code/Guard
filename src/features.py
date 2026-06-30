import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define feature groups based on synthetic recovery dataset schema
NUM_FEATURES = [
    'device_risk_score',
    'ip_risk_score',
    'time_since_last_login_days',
    'time_since_account_creation_days',
    'failed_recovery_attempts_7d',
    'hour_of_day'
]

CAT_FEATURES = [
    'recovery_channel'
]

PASSTHROUGH_FEATURES = [
    'is_new_device',
    'is_new_ip',
    'geo_mismatch_flag',
    'is_unusual_hour',
    'onboarding_to_recovery_speed_flag'
]

FEATURE_COLUMNS = NUM_FEATURES + CAT_FEATURES + PASSTHROUGH_FEATURES

def get_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer for preprocessing recovery attempts.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, NUM_FEATURES),
            ('cat', cat_pipeline, CAT_FEATURES),
            ('pass', 'passthrough', PASSTHROUGH_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor

def load_data(filepath):
    """
    Loads recovery dataset and splits it into features and target.
    """
    df = pd.read_csv(filepath)
    # Ensure standard schema columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset.")
            
    X = df[FEATURE_COLUMNS]
    y = df['is_fraud'] if 'is_fraud' in df.columns else None
    return X, y, df
