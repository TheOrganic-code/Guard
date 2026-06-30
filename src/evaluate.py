import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from features import load_data

def set_style():
    """Sets matplotlib parameters for a premium Obsidian Orange dark theme."""
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#09090b'      # Obsidian base
    plt.rcParams['axes.facecolor'] = '#18181b'        # Slate container
    plt.rcParams['axes.edgecolor'] = '#27272a'        # Card border
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#27272a'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['text.color'] = '#f4f4f5'
    plt.rcParams['axes.labelcolor'] = '#e4e4e7'
    plt.rcParams['xtick.color'] = '#a1a1aa'
    plt.rcParams['ytick.color'] = '#a1a1aa'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'sans-serif']

def evaluate_models():
    set_style()
    test_path = "data/processed/recovery_test.csv"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    print(f"Loading test dataset from {test_path}...")
    X_test, y_test, _ = load_data(test_path)
    
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)

    models = {}
    for name in ['rf', 'gb', 'lr']:
        model_path = f"models/{name}_pipeline.pkl"
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
        else:
            print(f"Warning: {name} model pipeline not found at {model_path}")

    if not models:
        raise RuntimeError("No models found to evaluate. Train models first.")

    comparison_results = {}
    predictions = {}

    for name, pipeline in models.items():
        print(f"Evaluating {name.upper()}...")
        probs = pipeline.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        # Calculate scores
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        
        brier = brier_score_loss(y_test, probs)
        
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        comparison_results[name] = {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'f1_score': float(f1),
            'precision': float(prec),
            'recall': float(rec),
            'brier_score': float(brier)
        }
        
        predictions[name] = {
            'probs': probs,
            'fpr': fpr,
            'tpr': tpr,
            'precision_vals': precision_vals,
            'recall_vals': recall_vals
        }

    # Save metrics JSON
    with open("reports/metrics/model_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=4)
    print("Saved reports/metrics/model_comparison.json")

    # Define color scheme for models
    # Orange (primary), Emerald (accent), Amber (secondary)
    colors = {
        'rf': '#FF5A1F',  # Bank of Baroda Vermillion Orange
        'gb': '#10B981',  # Emerald Green
        'lr': '#F59E0B'   # Amber Yellow
    }
    model_labels = {
        'rf': 'Random Forest',
        'gb': 'Gradient Boosting',
        'lr': 'Logistic Regression'
    }

    # Plot 1: ROC Curves
    plt.figure(figsize=(8, 6))
    for name in models:
        plt.plot(
            predictions[name]['fpr'], 
            predictions[name]['tpr'], 
            color=colors[name], 
            lw=2.5, 
            label=f"{model_labels[name]} (AUC = {comparison_results[name]['roc_auc']:.4f})"
        )
    plt.plot([0, 1], [0, 1], color='#3f3f46', lw=1.5, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (FPR)', fontsize=11, labelpad=10)
    plt.ylabel('True Positive Rate (TPR)', fontsize=11, labelpad=10)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc='lower right', frameon=True, facecolor='#18181b', edgecolor='#27272a')
    plt.tight_layout()
    plt.savefig('reports/figures/roc_curve.png', dpi=300)
    plt.close()

    # Plot 2: Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for name in models:
        plt.plot(
            predictions[name]['recall_vals'], 
            predictions[name]['precision_vals'], 
            color=colors[name], 
            lw=2.5, 
            label=f"{model_labels[name]} (AP = {comparison_results[name]['pr_auc']:.4f})"
        )
    plt.xlabel('Recall', fontsize=11, labelpad=10)
    plt.ylabel('Precision', fontsize=11, labelpad=10)
    plt.title('Precision-Recall Curve', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc='lower left', frameon=True, facecolor='#18181b', edgecolor='#27272a')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig('reports/figures/pr_curve.png', dpi=300)
    plt.close()

    # Plot 3: Calibration Curves (Reliability Curves)
    plt.figure(figsize=(8, 6))
    for name in models:
        prob_true, prob_pred = calibration_curve(y_test, predictions[name]['probs'], n_bins=10)
        plt.plot(
            prob_pred, 
            prob_true, 
            marker='o', 
            color=colors[name], 
            lw=2, 
            label=f"{model_labels[name]} (Brier = {comparison_results[name]['brier_score']:.4f})"
        )
    plt.plot([0, 1], [0, 1], color='#3f3f46', lw=1.5, linestyle='--', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=11, labelpad=10)
    plt.ylabel('Fraction of Positives', fontsize=11, labelpad=10)
    plt.title('Probability Calibration (Reliability Curve)', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc='upper left', frameon=True, facecolor='#18181b', edgecolor='#27272a')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig('reports/figures/calibration_curve.png', dpi=300)
    plt.close()

    # Plot 4: Confusion Matrix for the best model (based on ROC-AUC)
    best_model_name = max(comparison_results, key=lambda k: comparison_results[k]['roc_auc'])
    print(f"Generating Confusion Matrix for best model: {best_model_name.upper()}...")
    
    best_probs = predictions[best_model_name]['probs']
    best_preds = (best_probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, best_preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap=sns.dark_palette(colors[best_model_name], as_cmap=True),
        cbar=False,
        xticklabels=['Safe', 'Fraud'],
        yticklabels=['Safe', 'Fraud'],
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.ylabel('Actual Category', fontsize=11, labelpad=10)
    plt.xlabel('Predicted Category', fontsize=11, labelpad=10)
    plt.title(f'Confusion Matrix ({model_labels[best_model_name]} @ 0.50)', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('reports/figures/confusion_matrix.png', dpi=300)
    plt.close()
    
    print("Successfully generated all evaluation charts under reports/figures/.")

if __name__ == "__main__":
    evaluate_models()
