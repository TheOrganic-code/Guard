import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from features import get_preprocessor, load_data

def train_all_models():
    train_path = "data/processed/recovery_train.csv"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}. Please run preparation first.")

    print(f"Loading training dataset from {train_path}...")
    X_train, y_train, _ = load_data(train_path)
    print(f"Loaded {X_train.shape[0]} training samples with {X_train.shape[1]} features.")

    # Get the feature preprocessor
    preprocessor = get_preprocessor()

    # Define the three models
    models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        'lr': LogisticRegression(max_iter=1000, random_state=42)
    }

    os.makedirs("models", exist_ok=True)

    # Train and save each model
    for name, clf in models.items():
        print(f"Training {name.upper()} Model...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        
        save_path = f"models/{name}_pipeline.pkl"
        joblib.dump(pipeline, save_path)
        print(f"Successfully saved {name.upper()} pipeline to {save_path}")

if __name__ == "__main__":
    train_all_models()
