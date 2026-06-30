import os
import json
import joblib
import pandas as pd
import numpy as np
from features import load_data

C_FP = 800     # Manual review cost in INR
C_FN = 25000   # Fraud loss cost in INR

def compute_detailed_policy_stats(y_true, y_prob, threshold, c_fp=C_FP, c_fn=C_FN):
    """
    Computes confusion matrix, cost, precision, recall, and savings for a specific threshold.
    """
    preds = (y_prob >= threshold).astype(int)
    
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    tp = int(np.sum((y_true == 1) & (preds == 1)))
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Base cost if we approved everything (no fraud detection)
    base_cost = int(np.sum(y_true == 1) * c_fn)
    
    # Current operational cost
    model_cost = int(fp * c_fp + fn * c_fn)
    
    # Savings
    savings = int(base_cost - model_cost)
    
    return {
        'threshold': float(threshold),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'model_cost': model_cost,
        'base_cost': base_cost,
        'savings': savings
    }

def calculate_threshold_policies():
    test_path = "data/processed/recovery_test.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    _, y_test, _ = load_data(test_path)
    
    models = {}
    for name in ['rf', 'gb', 'lr']:
        model_path = f"models/{name}_pipeline.pkl"
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)

    if not models:
        raise RuntimeError("No models found. Train models first.")

    policies_output = {}

    for name, pipeline in models.items():
        probs = pipeline.predict_proba(pipeline.named_steps['preprocessor'].transform(pipeline.feature_names_in_ if hasattr(pipeline, 'feature_names_in_') else pd.read_csv(test_path).drop(columns=['is_fraud','attempt_id','account_id'], errors='ignore')) if False else pd.read_csv(test_path))[:, 1]
        # Wait, the pipeline can be run directly on X_test since preprocessor handles it
        X_test, _, _ = load_data(test_path)
        probs = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate thresholds
        thresholds = np.linspace(0.0, 1.0, 1001)
        
        # 1. Cost Optimized
        best_cost = float('inf')
        opt_cost_thresh = 0.5
        for t in thresholds:
            preds = (probs >= t).astype(int)
            fp = np.sum((y_test == 0) & (preds == 1))
            fn = np.sum((y_test == 1) & (preds == 0))
            cost = fp * C_FP + fn * C_FN
            if cost < best_cost:
                best_cost = cost
                opt_cost_thresh = t
                
        # 2. Balanced F1
        best_f1 = -1
        opt_f1_thresh = 0.5
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = np.sum((y_test == 1) & (preds == 1))
            fp = np.sum((y_test == 0) & (preds == 1))
            fn = np.sum((y_test == 1) & (preds == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                opt_f1_thresh = t

        # 3. High Security (Recall >= 95%)
        # Find the highest threshold that achieves Recall >= 95%
        high_sec_thresh = 0.1
        for t in reversed(thresholds):
            preds = (probs >= t).astype(int)
            tp = np.sum((y_test == 1) & (preds == 1))
            fn = np.sum((y_test == 1) & (preds == 0))
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if rec >= 0.95:
                high_sec_thresh = t
                break

        # 4. Low Friction (Precision >= 95%)
        # Find the lowest threshold that achieves Precision >= 95%
        low_fric_thresh = 0.9
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tp = np.sum((y_test == 1) & (preds == 1))
            fp = np.sum((y_test == 0) & (preds == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if prec >= 0.95:
                low_fric_thresh = t
                break

        # 5. Review Capacity (Top 10%)
        # Sort probabilities and pick threshold that flags exactly top 10%
        top_10_idx = int(len(probs) * 0.1)
        sorted_probs = np.sort(probs)
        review_cap_thresh = sorted_probs[-top_10_idx] if top_10_idx > 0 else 0.5
        
        # Package everything
        policies_output[name] = {
            'cost_optimized': compute_detailed_policy_stats(y_test, probs, opt_cost_thresh),
            'balanced_f1': compute_detailed_policy_stats(y_test, probs, opt_f1_thresh),
            'high_security': compute_detailed_policy_stats(y_test, probs, high_sec_thresh),
            'low_friction': compute_detailed_policy_stats(y_test, probs, low_fric_thresh),
            'review_capacity': compute_detailed_policy_stats(y_test, probs, review_cap_thresh)
        }
        
    os.makedirs("reports/metrics", exist_ok=True)
    with open("reports/metrics/threshold_policies.json", "w") as f:
        json.dump(policies_output, f, indent=4)
        
    print("Precomputed all threshold policies and saved to reports/metrics/threshold_policies.json")

if __name__ == "__main__":
    calculate_threshold_policies()
