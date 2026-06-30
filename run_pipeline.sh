#!/bin/bash
# RecoverGuard Orchestration Pipeline Script
# Styled for Bank of Baroda Hackathon 2026

set -e # Terminate on error

echo "=========================================================="
echo "🛡️  RECOVERGUARD: SYSTEM ARCHITECTURE COMPILER & EXECUTION"
echo "=========================================================="

# 1. Ensure Python Path has the src directory
export PYTHONPATH=src

# 2. Run Model Training
echo ""
echo "🚀 [STEP 1/3] Training Models (Random Forest, Gradient Boosting, Logistic Regression)..."
python src/train_model.py

# 3. Run Performance Evaluation & Generate Benchmarks
echo ""
echo "📈 [STEP 2/3] Evaluating Models & Rendering Performance Visualizations..."
python src/evaluate.py

# 4. Precompute Decision Threshold Policies
echo ""
echo "⚙️  [STEP 3/3] Precomputing Financial Utility Threshold Policies..."
python src/threshold_policy.py

echo ""
echo "=========================================================="
echo "✅ PIPELINE COMPILED SUCCESSFULLY!"
echo "=========================================================="
echo "Starting Gradio Dashboard on http://localhost:7860..."
echo "Press Ctrl+C to stop the console server."
echo "=========================================================="

# 5. Launch the Web App
python app.py
