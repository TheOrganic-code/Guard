---
title: RecoverGuard
emoji: 🛡️
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# RecoverGuard — Account Recovery Fraud Detection Console

🛡️ **RecoverGuard** is a state-of-the-art production-grade account recovery auditing system, custom-built for the **Bank of Baroda Hackathon 2026** (Theme: *Identity Trust, Protection & Safety*). 

This space is designed to help bank administrators and security teams monitor credential recovery attempts, identify fraud in real-time, explain AI decisions via SHAP (attribution analysis), and minimize operational expenses in Rupees (₹) using business-utility threshold compilers.

## 🚀 Key Features

1. **🌐 Dynamic English & Hindi Localization**: Real-time toggling between **English** and **Hindi / हिन्दी** (Indic) across all KPI metrics, logs, search fields, and diagnostic cards.
2. **🌓 Theme Switcher**: Toggle between a premium cybersecurity **Obsidian Dark** aesthetic and a high-contrast **Light Mode**.
3. **🧠 Layman-Friendly Indicators**: Variables are renamed to clean operational banking terms, with detailed guides explaining charts and financial sandboxes.
4. **📊 Multi-Architecture Auditor**: Compare and hot-swap between three models: **Gradient Boosting**, **Random Forest**, and **Logistic Regression**.
5. **🕵️‍♂️ Explainable AI (XAI)**: Displays SHAP feature attributions and localized rule-based reason codes (`BOB-RC01` to `BOB-RC05`) for immediate manual verification actions.
6. **👥 Team 4mistakes details**: Developed by members from the **Rajiv Gandhi Institute of Petroleum Technology** (An Institute of National Importance - INI).

---

## 📂 Project Directory Structure

```
├── app.py                                  # Gradio dashboard app entrypoint
├── requirements.txt                        # Python dependencies
├── README.md                               # Space configuration metadata & documentation
├── run_pipeline.sh                         # Automation shell script
├── src/                                    # Modular source code
│   ├── features.py                         # Data loading & scaling
│   ├── train_model.py                      # Multi-model training
│   ├── evaluate.py                         # Model evaluation & curve plots
│   ├── threshold_policy.py                 # Policy compiler in Rupees (₹)
│   ├── reason_codes.py                     # Static heuristic alert triggers
│   └── explain.py                          # SHAP explainability plot wrapper
├── models/                                 # Serialized ML pipeline objects
│   ├── rf_pipeline.pkl
│   ├── gb_pipeline.pkl
│   └── lr_pipeline.pkl
├── data/                                   # Datastore directories
│   └── processed/
│       └── recovery_test.csv               # Live database feed (700+ records)
└── reports/                                # Precomputed benchmarks
    ├── metrics/
    │   ├── model_comparison.json
    │   └── threshold_policies.json
    └── figures/
        ├── roc_curve.png
        ├── pr_curve.png
        ├── calibration_curve.png
        └── confusion_matrix.png
```

---

## 🛠️ Local Verification

To run this application locally, install the requirements and run the pipeline runner script:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete model training, evaluation, and app launch script
./run_pipeline.sh
```

The Gradio server will launch on `http://localhost:7860/`.

---

## 👥 Team 4mistakes
* **Ayush Pandey** (Team Lead & AI Developer) — [LinkedIn Profile](https://www.linkedin.com/in/ayushpandey1801/)
* **Karan** (System Architect & Data Engineer) — [LinkedIn Profile](https://www.linkedin.com/in/twynixkaran/)
* **Anurag Sharma** (Fullstack Engineer & UI Lead) — [LinkedIn Profile](https://www.linkedin.com/in/anurag-sharma-silver/)

**Institution**: Rajiv Gandhi Institute of Petroleum Technology (Rajiv Gandhi Institute of Petroleum Technology, Jais - An Institute of National Importance).
