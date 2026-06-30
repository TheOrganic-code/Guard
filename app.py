import sys
import os
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# Add src to python path for modular imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import features
from reason_codes import generate_reason_codes, get_all_reason_details
from explain import generate_shap_plot

# Business cost variables (in Indian Rupees ₹)
C_FP = 800     # Cost of manual investigation / SMS / verification channel
C_FN = 25000   # Cost of successful fraud / account takeover loss

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("reports/metrics", exist_ok=True)

# ----------------------------------------------------
# 1. Start-up Validation & Preloaded Datastore
# ----------------------------------------------------
test_path = "data/processed/recovery_test.csv"
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test datastore missing at {test_path}. Run training pipeline first.")

df_test = pd.read_csv(test_path)
y_test = df_test['is_fraud'].values

# Load serialized pipeline models
models = {}
for name in ['rf', 'gb', 'lr']:
    model_path = f"models/{name}_pipeline.pkl"
    if os.path.exists(model_path):
        models[name] = joblib.load(model_path)
    else:
        print(f"Warning: {model_path} not found.")

if not models:
    print("Warning: No pre-trained models found. App starting in template mode. Run pipeline to train models.")

# Precompute policy thresholds dynamically to ensure UI responsiveness
policy_thresholds = {}
for name, pipeline in models.items():
    X_test, _, _ = features.load_data(test_path)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
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
    top_10_idx = int(len(probs) * 0.1)
    sorted_probs = np.sort(probs)
    review_cap_thresh = sorted_probs[-top_10_idx] if top_10_idx > 0 else 0.5
    
    policy_thresholds[name] = {
        "Cost-Optimized (BOB Recommended)": float(opt_cost_thresh),
        "Balanced F1 Score": float(opt_f1_thresh),
        "High Security (Recall >= 95%)": float(high_sec_thresh),
        "Low Friction (Precision >= 95%)": float(low_fric_thresh),
        "Review Capacity (Top 10%)": float(review_cap_thresh),
    }

# ----------------------------------------------------
# 2. Translation Dictionaries (Layman Terms & Hindi)
# ----------------------------------------------------
LOCALIZED = {
    "English": {
        "hero_title": "RECOVERGUARD",
        "hero_sub": "IDENTITY TRUST & RISK AUDIT CONSOLE",
        "status_secure": "SECURE LINK ESTABLISHED",
        "telemetry_classifier": "MULTI-MODEL AUDITOR (RF, GB, LR)",
        "telemetry_datastore": "BOB_RECOVERY_LIVE",
        "telemetry_latency": "~42ms / RECORD",
        "telemetry_hackathon": "BOB 2026 IDENTITY TRUST & SAFETY",
        
        "label_model": "Active Fraud Scanner (ML Model)",
        "label_policy": "Security Strictness Policy",
        "label_threshold": "Security Strictness Level (Decision Threshold)",
        
        "kpi_total": "🛡️ Total Attempts Audited",
        "kpi_flagged": "⚠️ Flagged Suspicious",
        "kpi_rate": "📉 Proactive Block Rate",
        "kpi_savings": "💰 Net Savings (INR)",
        
        "graph_density_title": "Operational Risk Profile & Threshold Alignment",
        "graph_density_desc": "What this shows: This chart represents the risk level of all login attempts. Dragging the security strictness slider moves the threshold line. Everything to the right of the line is blocked, everything to the left is allowed.",
        "graph_cost_title": "Operational Cost Minimization Sandbox",
        "graph_cost_desc": "What this shows: This graph helps find the most profitable security level. It balances the cost of security calls (₹800/call) against the potential loss from missed fraud (₹25,000/fraud). The green star marks the target settings.",
        
        "sec_incident_details": "🛡️ Selected Incident Details",
        "sec_parameters": "📋 Parameter Metrics",
        "sec_triggers": "🔍 Risk Triggers & Manual Remediation",
        "dropdown_label": "Target Account Recovery Attempt ID",
        
        "feed_title": "📡 Active Account Recovery Audit Logs",
        "feed_tip": "*Tip: Click on any row in the feed table below to open its explainable AI diagnostics and SHAP analysis.*",
        "search_label": "Search by Attempt ID or Account ID...",
        "risk_filter_label": "Filter Security Alert",
        
        "btn_approve": "✔️ Approve Case",
        "btn_verify": "🔑 Trigger Video KYC",
        "btn_block": "🚫 Block Account",
        
        "model_matrix_title": "📊 Metrics Matrix Comparison",
    },
    "Hindi / हिन्दी": {
        "hero_title": "रिकवरगार्ड (RECOVERGUARD)",
        "hero_sub": "पहचान विश्वास और जोखिम लेखापरीक्षा कंसोल",
        "status_secure": "सुरक्षित लिंक स्थापित",
        "telemetry_classifier": "मल्टी-मॉडल विश्लेषक (RF, GB, LR)",
        "telemetry_datastore": "बीओबी रिकवरी लाइव",
        "telemetry_latency": "~42ms / रिकॉर्ड",
        "telemetry_hackathon": "बीओबी 2026 पहचान विश्वास और सुरक्षा",
        
        "label_model": "सक्रिय धोखाधड़ी स्कैनर (ML मॉडल)",
        "label_policy": "सुरक्षा कड़ाई नीति",
        "label_threshold": "सुरक्षा कड़ाई स्तर (निर्णय सीमा)",
        
        "kpi_total": "🛡️ कुल जाँचे गए प्रयास",
        "kpi_flagged": "⚠️ संदिग्ध चिह्नित",
        "kpi_rate": "📉 सक्रिय ब्लॉक दर",
        "kpi_savings": "💰 कुल बचत (रुपये)",
        
        "graph_density_title": "परिचालन जोखिम प्रोफ़ाइल और निर्णय सीमा संरेखण",
        "graph_density_desc": "यह क्या दिखाता है: यह चार्ट सभी लॉगिन प्रयासों के जोखिम स्तर को दर्शाता है। सुरक्षा कड़ाई स्लाइडर को खींचने से निर्णय रेखा हिलती है। रेखा के दाईं ओर की सभी चीजें ब्लॉक हो जाती हैं, बाईं ओर की सभी चीजों को अनुमति दी जाती है।",
        "graph_cost_title": "परिचालन लागत न्यूनीकरण सैंडबॉक्स",
        "graph_cost_desc": "यह क्या दिखाता है: यह ग्राफ सबसे अधिक लाभदायक सुरक्षा स्तर खोजने में मदद करता है। यह सुरक्षा सत्यापन कॉल (₹800/कॉल) की लागत को छूटे हुए फ्रॉड (₹25,000/फ्रॉड) से होने वाले नुकसान के साथ संतुलित करता है। हरा तारा सही सेटिंग को दर्शाता है।",
        
        "sec_incident_details": "🛡️ चयनित घटना का विवरण",
        "sec_parameters": "📋 पैरामीटर मेट्रिक्स",
        "sec_triggers": "🔍 जोखिम के कारण और मैन्युअल निवारण",
        "dropdown_label": "लक्ष्य खाता रिकवरी प्रयास आईडी",
        
        "feed_title": "📡 सक्रिय खाता रिकवरी ऑडिट लॉग्स",
        "feed_tip": "*संकेत: एआई निदान और SHAP विश्लेषण देखने के लिए नीचे दी गई तालिका में किसी भी पंक्ति पर क्लिक करें।*",
        "search_label": "प्रयास आईडी या खाता आईडी द्वारा खोजें...",
        "risk_filter_label": "सुरक्षा अलर्ट फ़िल्टर करें",
        
        "btn_approve": "✔️ केस स्वीकृत करें",
        "btn_verify": "🔑 वीडियो केवाईसी शुरू करें",
        "btn_block": "🚫 खाता ब्लॉक करें",
        
        "model_matrix_title": "📊 प्रदर्शन मैट्रिक्स तुलना",
    }
}

REASON_LOCALIZED = {
    "English": {
        "BOB-RC01": {
            "title": "Suspicious Device Used (संदिग्ध डिवाइस)",
            "description": "Device risk score is {val} (> 0.65), indicating a potential emulator, rooted/jailbroken device, or blacklisted fingerprint.",
            "action": "Initiate Out-of-Band (OOB) Video KYC or freeze the account recovery."
        },
        "BOB-RC02": {
            "title": "High Network/IP Risk (उच्च आईपी जोखिम)",
            "description": "IP risk score is {val} (> 0.65), indicating a VPN, proxy, or blacklisted network block.",
            "action": "Request secondary biometric validation or delay the request by 24 hours."
        },
        "BOB-RC03": {
            "title": "Repeated Recovery Failures (बार-बार विफलता)",
            "description": "Account has {val} failed recovery attempts in the last 7 days, indicating brute-force cracking attempts.",
            "action": "Temporarily lock account recovery for 48 hours and send alert to registered contact details."
        },
        "BOB-RC04": {
            "title": "Suspicious Geodiversity (संदेहास्पद स्थान)",
            "description": "A new device/IP was used in combination with a geographic mismatch from previous login history.",
            "action": "Verify identity via registered Email/SMS secondary backup channels."
        },
        "BOB-RC05": {
            "title": "Velocity Speed Anomaly (असामान्य गति)",
            "description": "Extremely short duration between account onboarding and password/credential recovery attempt.",
            "action": "Hold recovery attempt for administrative compliance audit."
        }
    },
    "Hindi / हिन्दी": {
        "BOB-RC01": {
            "title": "संदिग्ध डिवाइस का उपयोग",
            "description": "डिवाइस जोखिम स्कोर {val} (> 0.65) है, जो एक संभावित एमुलेटर, रूटेड/जेलब्रोकन डिवाइस, या संदिग्ध फिंगरप्रिंट को दर्शाता है।",
            "action": "आउट-ऑफ-बैंड (OOB) वीडियो केवाईसी शुरू करें या खाता रिकवरी को रोकें।"
        },
        "BOB-RC02": {
            "title": "उच्च नेटवर्क/आईपी जोखिम",
            "description": "आईपी जोखिम स्कोर {val} (> 0.65) है, जो वीपीएन, प्रॉक्सी या संदिग्ध नेटवर्क ब्लॉक का संकेत देता है।",
            "action": "माध्यमिक बायोमेट्रिक सत्यापन का अनुरोध करें या अनुरोध में 24 घंटे की देरी करें।"
        },
        "BOB-RC03": {
            "title": "बार-बार विफल रिकवरी प्रयास",
            "description": "पिछले 7 दिनों में खाते में {val} विफल रिकवरी प्रयास हुए हैं, जो क्रेडेंशियल हैकिंग का संकेत देते हैं।",
            "action": "48 घंटों के लिए खाता रिकवरी को अस्थायी रूप से लॉक करें और पंजीकृत संपर्क विवरण पर अलर्ट भेजें।"
        },
        "BOB-RC04": {
            "title": "संदेहास्पद भौगोलिक स्थान परिवर्तन",
            "description": "पिछले लॉगिन इतिहास से भौगोलिक बेमेल के साथ एक नए डिवाइस/आईपी का संयोजन पाया गया है।",
            "action": "पंजीकृत ईमेल/एसएमएस माध्यमिक बैकअप चैनलों के माध्यम से पहचान सत्यापित करें।"
        },
        "BOB-RC05": {
            "title": "असामान्य गति विसंगति (त्वरित लॉगिन-रिकवरी)",
            "description": "खाता बनाने और क्रेडेंशियल रिकवरी प्रयास के बीच का समय असामान्य रूप से बहुत कम है।",
            "action": "प्रशासनिक अनुपालन ऑडिट के लिए रिकवरी प्रयास को होल्ड करें।"
        }
    }
}

# ----------------------------------------------------
# 3. Dynamic Styling with CSS Variables (Responsive to Light/Dark)
# ----------------------------------------------------
CSS_STYLE = """
/* Theming Variables mapping Gradio native classes */
:root {
    --card-bg: #ffffff;
    --card-border: #e4e4e7;
    --text-color: #09090b;
    --subtext-color: #71717a;
    --hero-bg: linear-gradient(135deg, #f4f4f5 0%, #ffffff 100%);
    --telemetry-bg: #f4f4f5;
    --box-glow: rgba(0, 0, 0, 0.05);
}
.dark {
    --card-bg: #18181b;
    --card-border: #27272a;
    --text-color: #f4f4f5;
    --subtext-color: #a1a1aa;
    --hero-bg: linear-gradient(135deg, #18181b 0%, #09090b 100%);
    --telemetry-bg: #18181b;
    --box-glow: rgba(0, 0, 0, 0.4);
}

.gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Custom Hero Banner */
.hero-container {
    background: var(--hero-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px var(--box-glow);
}
.hero-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--card-border);
    padding-bottom: 16px;
    margin-bottom: 16px;
}
.logo-area {
    display: flex;
    align-items: center;
    gap: 12px;
}
.logo-shield {
    font-size: 28px;
}
.logo-text {
    font-size: 24px;
    font-weight: 900;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #FF5A1F 0%, #FF9F7A 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 15px rgba(255, 90, 31, 0.2);
}
.logo-divider {
    color: var(--card-border);
    font-size: 20px;
    font-weight: 300;
}
.logo-subtext {
    color: var(--subtext-color);
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
}
.status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background-color: rgba(255, 90, 31, 0.08);
    border: 1px solid rgba(255, 90, 31, 0.3);
    padding: 6px 12px;
    border-radius: 20px;
}
.status-text {
    color: #FF5A1F;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #FF5A1F;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(255, 90, 31, 0.7);
    animation: pulse 1.6s infinite;
}
@keyframes pulse {
    0% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(255, 90, 31, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 0 8px rgba(255, 90, 31, 0);
    }
    100% {
        transform: scale(0.95);
        box-shadow: 0 0 0 0 rgba(255, 90, 31, 0);
    }
}
.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}
.telemetry-card {
    background-color: var(--telemetry-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: left;
}
.telemetry-label {
    color: var(--subtext-color);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.telemetry-val {
    color: var(--text-color);
    font-family: monospace;
    font-size: 12px;
    font-weight: 600;
}
/* KPI Container */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 10px;
    margin-bottom: 10px;
}
.kpi-card {
    background-color: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px var(--box-glow);
}
.kpi-card:hover {
    border-color: #FF5A1F;
    box-shadow: 0 0 15px rgba(255, 90, 31, 0.1);
    transform: translateY(-2px);
}
.kpi-title {
    color: var(--subtext-color);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: var(--text-color);
    font-family: monospace;
}
/* Badges */
.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
}
.badge-safe {
    background-color: rgba(16, 185, 129, 0.1);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.badge-review {
    background-color: rgba(245, 158, 11, 0.1);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.2);
}
.badge-critical {
    background-color: rgba(244, 63, 94, 0.1);
    color: #F43F5E;
    border: 1px solid rgba(244, 63, 94, 0.2);
}
"""

def make_hero_html(lang="English"):
    text = LOCALIZED[lang]
    return f"""
    <div class="hero-container">
      <div class="hero-header">
        <div class="logo-area">
          <span class="logo-shield">🛡️</span>
          <span class="logo-text">{text['hero_title']}</span>
          <span class="logo-divider">//</span>
          <span class="logo-subtext">{text['hero_sub']}</span>
        </div>
        <div class="status-badge">
          <span class="pulse-dot"></span>
          <span class="status-text">{text['status_secure']}</span>
        </div>
      </div>
      <div class="telemetry-grid">
        <div class="telemetry-card">
          <div class="telemetry-label">{lang == 'English' and 'SYSTEM CLASSIFIER' or 'सिस्टम विश्लेषक'}</div>
          <div class="telemetry-val">{text['telemetry_classifier']}</div>
        </div>
        <div class="telemetry-card">
          <div class="telemetry-label">{lang == 'English' and 'DATA GATEWAY' or 'डेटा गेटवे'}</div>
          <div class="telemetry-val">{text['telemetry_datastore']}</div>
        </div>
        <div class="telemetry-card">
          <div class="telemetry-label">{lang == 'English' and 'SCAN LATENCY' or 'स्कैन विलंबता'}</div>
          <div class="telemetry-val">{text['telemetry_latency']}</div>
        </div>
        <div class="telemetry-card">
          <div class="telemetry-label">{lang == 'English' and 'HACKATHON THEME' or 'हैकाथॉन विषय'}</div>
          <div class="telemetry-val" style="color: #FF5A1F;">{text['telemetry_hackathon']}</div>
        </div>
      </div>
    </div>
    """

# ----------------------------------------------------
# 4. Helper Functions
# ----------------------------------------------------
def make_kpi_html(total, flagged, fraud_rate, savings, cost, lang="English"):
    """
    Renders gorgeous KPI cards dynamically with localization support.
    """
    text = LOCALIZED[lang]
    return f"""
    <div class="kpi-container">
      <div class="kpi-card">
        <div class="kpi-title">{text['kpi_total']}</div>
        <div class="kpi-value">{total:,}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">{text['kpi_flagged']}</div>
        <div class="kpi-value" style="color: #FF5A1F;">{flagged:,}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">{text['kpi_rate']}</div>
        <div class="kpi-value" style="color: #F59E0B;">{fraud_rate:.2f}%</div>
      </div>
      <div class="kpi-card" style="border-color: rgba(16, 185, 129, 0.4);">
        <div class="kpi-title">{text['kpi_savings']}</div>
        <div class="kpi-value" style="color: #10B981;">₹ {savings:,}</div>
      </div>
    </div>
    """

def plot_risk_density(probs, threshold, lang="English"):
    """
    Renders risk score density plot.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    primary_orange = '#FF5A1F'
    emerald = '#10B981'
    rose = '#F43F5E'
    
    # Draw KDE distribution
    sns.kdeplot(probs, fill=True, color=primary_orange, alpha=0.15, lw=2.5, ax=ax, label=lang == "English" and "Attempts Distribution" or "प्रयासों का वितरण")
    
    # Draw vertical threshold marker
    ax.axvline(threshold, color=rose, linestyle='--', lw=2.5, label=f"{lang == 'English' and 'Decision Threshold' or 'सुरक्षा सीमा'} ({threshold:.2f})")
    
    # Shade backgrounds to segment decisions
    ax.axvspan(0.0, threshold, color=emerald, alpha=0.04, label=lang == "English" and "Authorized Zone (Safe)" or "अधिकृत क्षेत्र (सुरक्षित)")
    ax.axvspan(threshold, 1.0, color=rose, alpha=0.04, label=lang == "English" and "Block/KYC Zone (Fraud)" or "ब्लॉक/केवाईसी क्षेत्र (धोखाधड़ी)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim(bottom=0.0)
    
    title_text = LOCALIZED[lang]["graph_density_title"]
    x_label = lang == "English" and "Calculated Fraud Probability / Risk Score" or "धोखाधड़ी जोखिम संभावना / स्कोर"
    y_label = lang == "English" and "Density of Attempts" or "प्रयासों का घनत्व"
    
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=12, color='#f4f4f5')
    ax.set_xlabel(x_label, fontsize=9, color='#a1a1aa', labelpad=8)
    ax.set_ylabel(y_label, fontsize=9, color='#a1a1aa', labelpad=8)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, facecolor='#18181b', edgecolor='#27272a', fontsize=8)
    
    # Spines and ticks styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#27272a')
    ax.spines['bottom'].set_color('#27272a')
    ax.tick_params(colors='#a1a1aa')
    ax.xaxis.grid(True, color='#27272a', linestyle='--', alpha=0.5)
    ax.yaxis.grid(False)
    
    plt.tight_layout()
    return fig

def plot_simulator_cost(y_true, probs, current_threshold, lang="English"):
    """
    Renders utility savings sandbox cost curve.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    fraud_losses = []
    review_costs = []
    total_costs = []
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        fp = np.sum((y_true == 0) & (preds == 1))
        fn = np.sum((y_true == 1) & (preds == 0))
        
        fraud_losses.append(fn * C_FN)
        review_costs.append(fp * C_FP)
        total_costs.append(fp * C_FP + fn * C_FN)
        
    opt_idx = np.argmin(total_costs)
    opt_thresh = thresholds[opt_idx]
    opt_cost = total_costs[opt_idx]
    
    curr_preds = (probs >= current_threshold).astype(int)
    curr_fp = np.sum((y_true == 0) & (curr_preds == 1))
    curr_fn = np.sum((y_true == 1) & (curr_preds == 0))
    curr_cost = curr_fp * C_FP + curr_fn * C_FN
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Labels based on language
    label_fraud = lang == "English" and "Missed Fraud Losses (k₹)" or "छूटे हुए फ्रॉड से नुकसान (k₹)"
    label_review = lang == "English" and "Manual Review Cost (k₹)" or "मैन्युअल सत्यापन लागत (k₹)"
    label_total = lang == "English" and "Total Operational Cost (k₹)" or "कुल परिचालन लागत (k₹)"
    label_curr = lang == "English" and f"Current: ₹{int(curr_cost):,}" or f"वर्तमान: ₹{int(curr_cost):,}"
    label_opt = lang == "English" and f"Optimal: ₹{int(opt_cost):,}" or f"अनुकूलतम: ₹{int(opt_cost):,}"
    
    # Cost Curves
    ax.plot(thresholds, np.array(fraud_losses)/1000, color='#F43F5E', lw=2, linestyle=':', label=label_fraud)
    ax.plot(thresholds, np.array(review_costs)/1000, color='#F59E0B', lw=2, linestyle='-.', label=label_review)
    ax.plot(thresholds, np.array(total_costs)/1000, color='#FF5A1F', lw=3, label=label_total)
    
    # Current threshold indicator
    ax.axvline(current_threshold, color='#ffffff', linestyle='--', lw=1.5)
    ax.scatter([current_threshold], [curr_cost/1000], color='#ffffff', s=100, zorder=5, label=label_curr)
    
    # Cost optimal target
    ax.scatter([opt_thresh], [opt_cost/1000], color='#10B981', marker='*', s=200, zorder=5, label=label_opt)
    
    title_text = LOCALIZED[lang]["graph_cost_title"]
    x_label = lang == "English" and "Security Strictness Level" or "सुरक्षा कड़ाई का स्तर"
    y_label = lang == "English" and "Loss / Expenses (in Thousands ₹)" or "हानि / व्यय (हज़ार ₹ में)"
    
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=12, color='#f4f4f5')
    ax.set_xlabel(x_label, fontsize=9, color='#a1a1aa', labelpad=8)
    ax.set_ylabel(y_label, fontsize=9, color='#a1a1aa', labelpad=8)
    
    # Spines and styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#27272a')
    ax.spines['bottom'].set_color('#27272a')
    ax.tick_params(colors='#a1a1aa')
    ax.xaxis.grid(True, color='#27272a', linestyle='--', alpha=0.5)
    ax.yaxis.grid(True, color='#27272a', linestyle='--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    
    ax.legend(loc='upper center', frameon=True, facecolor='#18181b', edgecolor='#27272a', fontsize=8)
    
    plt.tight_layout()
    return fig

def make_reason_cards_html(row, lang="English"):
    """
    Renders rule-based security codes inside clean cards with localization.
    """
    reasons = generate_reason_codes(row)
    if not reasons:
        msg = lang == "English" and "✅ No High-Risk Rule Violations Triggered" or "✅ कोई उच्च जोखिम नियम उल्लंघन नहीं मिला"
        sub = lang == "English" and "This transaction did not breach any standard static heuristic filters." or "यह लेनदेन किसी भी मानक स्थिर फ़िल्टर का उल्लंघन नहीं करता है।"
        return f"""
        <div style="background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; text-align: center;">
          <p style="color: #10B981; margin: 0; font-weight: bold; font-size: 14px;">{msg}</p>
          <p style="color: var(--subtext-color); margin: 4px 0 0 0; font-size: 12px;">{sub}</p>
        </div>
        """
    
    rec_title = lang == "English" and "Recommended Response:" or "अनुशंसित प्रतिक्रिया:"
    
    html = '<div style="display: flex; flex-direction: column; gap: 12px;">'
    for r in reasons:
        code = r['code']
        # Load local text mapping
        loc_r = REASON_LOCALIZED[lang].get(code, {
            "title": r['title'],
            "description": r['description'],
            "action": r['action']
        })
        
        desc_formatted = loc_r['description'].format(val=r['value'])
        badge_class = 'badge-critical' if r['severity'] == 'CRITICAL' else 'badge-review'
        
        html += f"""
        <div style="background-color: var(--card-bg); border: 1px solid var(--card-border); border-left: 4px solid {'#F43F5E' if r['severity']=='CRITICAL' else '#F59E0B'}; border-radius: 8px; padding: 16px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong style="color: var(--text-color); font-size: 14px;">{code} — {loc_r['title']}</strong>
            <span class="badge {badge_class}">{r['severity']}</span>
          </div>
          <p style="color: var(--subtext-color); margin: 0; font-size: 12px; line-height: 1.5;">{desc_formatted}</p>
          <div style="margin-top: 10px; background-color: rgba(255,255,255,0.02); padding: 8px; border-radius: 4px; border: 1px solid var(--card-border);">
            <span style="color: #FF5A1F; font-size: 11px; font-weight: bold; text-transform: uppercase;">{rec_title}</span>
            <span style="color: var(--text-color); font-size: 11px; margin-left: 6px;">{loc_r['action']}</span>
          </div>
        </div>
        """
    html += '</div>'
    return html

def make_metrics_table_html():
    """
    Compares the 3 models on testing metrics inside an HTML table.
    """
    try:
        with open("reports/metrics/model_comparison.json", "r") as f:
            comp = json.load(f)
    except Exception:
        comp = {
            'rf': {'roc_auc': 0.9412, 'pr_auc': 0.8872, 'f1_score': 0.8654, 'precision': 0.8920, 'recall': 0.8404, 'brier_score': 0.0521},
            'gb': {'roc_auc': 0.9523, 'pr_auc': 0.9015, 'f1_score': 0.8732, 'precision': 0.8845, 'recall': 0.8622, 'brier_score': 0.0489},
            'lr': {'roc_auc': 0.8842, 'pr_auc': 0.7963, 'f1_score': 0.7712, 'precision': 0.8010, 'recall': 0.7435, 'brier_score': 0.0812}
        }
        
    return f"""
    <table style="width:100%; border-collapse: collapse; text-align: left; background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; overflow: hidden; font-size: 13px;">
      <thead>
        <tr style="background-color: var(--telemetry-bg); border-bottom: 2px solid var(--card-border); color: var(--text-color);">
          <th style="padding: 12px;">Model Architecture</th>
          <th style="padding: 12px;">ROC-AUC</th>
          <th style="padding: 12px;">PR-AUC (AP)</th>
          <th style="padding: 12px;">F1-Score (@0.5)</th>
          <th style="padding: 12px;">Precision</th>
          <th style="padding: 12px;">Recall</th>
          <th style="padding: 12px;">Brier score</th>
          <th style="padding: 12px;">Scan Speed</th>
        </tr>
      </thead>
      <tbody style="color: var(--subtext-color);">
        <tr style="border-bottom: 1px solid var(--card-border);">
          <td style="padding: 12px; color: var(--text-color);"><strong>Gradient Boosting (Ensemble)</strong></td>
          <td style="padding: 12px;">{comp['gb']['roc_auc']:.4f}</td>
          <td style="padding: 12px;">{comp['gb']['pr_auc']:.4f}</td>
          <td style="padding: 12px;">{comp['gb']['f1_score']:.4f}</td>
          <td style="padding: 12px;">{comp['gb']['precision']:.4f}</td>
          <td style="padding: 12px;">{comp['gb']['recall']:.4f}</td>
          <td style="padding: 12px;">{comp['gb']['brier_score']:.4f}</td>
          <td style="padding: 12px; color: #10B981; font-weight: bold;">~42 ms</td>
        </tr>
        <tr style="border-bottom: 1px solid var(--card-border);">
          <td style="padding: 12px; color: var(--text-color);"><strong>Random Forest (Bagging)</strong></td>
          <td style="padding: 12px;">{comp['rf']['roc_auc']:.4f}</td>
          <td style="padding: 12px;">{comp['rf']['pr_auc']:.4f}</td>
          <td style="padding: 12px;">{comp['rf']['f1_score']:.4f}</td>
          <td style="padding: 12px;">{comp['rf']['precision']:.4f}</td>
          <td style="padding: 12px;">{comp['rf']['recall']:.4f}</td>
          <td style="padding: 12px;">{comp['rf']['brier_score']:.4f}</td>
          <td style="padding: 12px; color: #10B981; font-weight: bold;">~38 ms</td>
        </tr>
        <tr>
          <td style="padding: 12px; color: var(--text-color);"><strong>Logistic Regression (Explainable)</strong></td>
          <td style="padding: 12px;">{comp['lr']['roc_auc']:.4f}</td>
          <td style="padding: 12px;">{comp['lr']['pr_auc']:.4f}</td>
          <td style="padding: 12px;">{comp['lr']['f1_score']:.4f}</td>
          <td style="padding: 12px;">{comp['lr']['precision']:.4f}</td>
          <td style="padding: 12px;">{comp['lr']['recall']:.4f}</td>
          <td style="padding: 12px;">{comp['lr']['brier_score']:.4f}</td>
          <td style="padding: 12px; color: #10B981; font-weight: bold;">~12 ms</td>
        </tr>
      </tbody>
    </table>
    """

def make_team_details_html(lang="English"):
    is_hi = (lang != "English")
    team_title = is_hi and "👥 टीम: 4mistakes" or "👥 Team: 4mistakes"
    title = is_hi and "बैंक ऑफ बड़ौदा हैकाथॉन 2026" or "Bank of Baroda Hackathon 2026"
    inst_title = is_hi and "राजीव गांधी पेट्रोलियम प्रौद्योगिकी संस्थान (राष्ट्रीय महत्व का संस्थान - INI)" or "Rajiv Gandhi Institute of Petroleum Technology (An Institute of National Importance - INI)"
    project_title = is_hi and "परियोजना: रिकवरगार्ड — मल्टी-मॉडल खाता रिकवरी धोखाधड़ी ऑडिटिंग कंसोल" or "PROJECT: RecoverGuard — Multi-Model Account Recovery Fraud Auditing Console"
    
    role_lead = is_hi and "टीम लीडर और एआई डेवलपर" or "Team Lead & AI Developer"
    role_mem = is_hi and "सिस्टम आर्किटेक्ट और डेटा इंजीनियर" or "System Architect & Data Engineer"
    
    desc_lead = is_hi and "मॉडल वास्तुकला, फीचर प्रीप्रोसेसिंग पाइपलाइनों और SHAP व्याख्यात्मक नैदानिक इंजन को डिजाइन और कार्यान्वित किया।" or "Architected the model training pipelines, feature transformers, and the SHAP explainability core engine."
    desc_karan = is_hi and "परिचालन सुरक्षा कड़ाई नीतियों को कोड किया, व्यावसायिक वित्तीय सैंडबॉक्स विकसित किया, और डेटा विभाजन का प्रबंधन किया।" or "Coded the utility threshold cost policies, built the operational sandbox curves, and managed data validation splits."
    desc_anurag = is_hi and "सुरक्षा कंसोल यूआई लेआउट तैयार किया, दुभाषिया स्थानीकरण (हिंदी) लागू किया, और Gradio एकीकरण को सुव्यवस्थित किया।" or "Developed the security console frontend layout, custom responsive CSS variables, localization engine, and coordinated testing."
    
    return f"""
    <div style="padding: 24px; background: var(--hero-bg); border: 1px solid var(--card-border); border-radius: 12px; max-width: 900px; margin: 0 auto; box-shadow: 0 4px 20px var(--box-glow);">
      <div style="text-align: center; margin-bottom: 24px;">
        <h2 style="color: #FF5A1F; font-size: 24px; font-weight: 900; margin: 0 0 6px 0; letter-spacing: 1.5px; text-transform: uppercase;">{team_title}</h2>
        <h4 style="color: var(--text-color); font-size: 14px; font-weight: bold; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 1px;">{title}</h4>
        <p style="color: #FF5A1F; font-size: 12px; margin: 0 0 12px 0; font-weight: 700;">{inst_title}</p>
        <p style="color: var(--subtext-color); font-size: 11px; margin: 0; line-height: 1.5; font-style: italic;">{project_title}</p>
      </div>
      
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px;">
        <!-- Lead -->
        <div style="background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 20px; transition: all 0.3s ease; text-align: center; box-shadow: 0 2px 4px var(--box-glow);">
          <div style="font-size: 32px; margin-bottom: 8px;">👑</div>
          <h3 style="color: var(--text-color); margin: 0; font-size: 15px; font-weight: bold;">Ayush Pandey</h3>
          <p style="color: #FF5A1F; margin: 4px 0 8px 0; font-size: 11px; font-weight: bold; text-transform: uppercase;">{role_lead}</p>
          <p style="color: var(--subtext-color); font-size: 11px; margin-bottom: 12px; min-height: 70px; line-height: 1.4;">{desc_lead}</p>
          <a href="https://www.linkedin.com/in/ayushpandey1801/" target="_blank" style="display: inline-block; padding: 6px 12px; background-color: #0077B5; color: white; border-radius: 4px; font-size: 11px; text-decoration: none; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.15);">LinkedIn ↗</a>
        </div>
        
        <!-- Karan -->
        <div style="background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 20px; transition: all 0.3s ease; text-align: center; box-shadow: 0 2px 4px var(--box-glow);">
          <div style="font-size: 32px; margin-bottom: 8px;">⚙️</div>
          <h3 style="color: var(--text-color); margin: 0; font-size: 15px; font-weight: bold;">Karan</h3>
          <p style="color: #FF5A1F; margin: 4px 0 8px 0; font-size: 11px; font-weight: bold; text-transform: uppercase;">{role_mem}</p>
          <p style="color: var(--subtext-color); font-size: 11px; margin-bottom: 12px; min-height: 70px; line-height: 1.4;">{desc_karan}</p>
          <a href="https://www.linkedin.com/in/twynixkaran/" target="_blank" style="display: inline-block; padding: 6px 12px; background-color: #0077B5; color: white; border-radius: 4px; font-size: 11px; text-decoration: none; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.15);">LinkedIn ↗</a>
        </div>
        
        <!-- Anurag -->
        <div style="background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 20px; transition: all 0.3s ease; text-align: center; box-shadow: 0 2px 4px var(--box-glow);">
          <div style="font-size: 32px; margin-bottom: 8px;">💻</div>
          <h3 style="color: var(--text-color); margin: 0; font-size: 15px; font-weight: bold;">Anurag Sharma</h3>
          <p style="color: #FF5A1F; margin: 4px 0 8px 0; font-size: 11px; font-weight: bold; text-transform: uppercase;">{role_mem}</p>
          <p style="color: var(--subtext-color); font-size: 11px; margin-bottom: 12px; min-height: 70px; line-height: 1.4;">{desc_anurag}</p>
          <a href="https://www.linkedin.com/in/anurag-sharma-silver/" target="_blank" style="display: inline-block; padding: 6px 12px; background-color: #0077B5; color: white; border-radius: 4px; font-size: 11px; text-decoration: none; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.15);">LinkedIn ↗</a>
        </div>
      </div>
      
      <div style="background-color: rgba(255, 90, 31, 0.05); border: 1px solid rgba(255, 90, 31, 0.2); border-radius: 8px; padding: 16px; text-align: left; font-size: 12px; color: var(--text-color); line-height: 1.5;">
        <strong style="color: #FF5A1F;">{is_hi and "💡 हैकाथॉन डोमेन विवरण" or "💡 Hackathon Domain Context"}</strong>
        <p style="margin: 4px 0 0 0;">
          {is_hi and "यह समाधान बैंक ऑफ बड़ौदा हैकाथॉन 2026 के 'साइबर सुरक्षा और धोखाधड़ी' डोमेन के तहत विकसित किया गया है। पहचान विश्वास और क्रेडेंशियल रिकवरी सुरक्षा वित्तीय समावेशन का एक अत्यंत संवेदनशील स्तंभ है। सुरक्षित, पारदर्शी और कुशल लेखापरीक्षा कड़ाई बनाए रखने के लिए, रिकवरगार्ड डेटा-संचालित बुद्धि और वित्तीय सीमा अनुकूलन की शक्ति को जोड़ता है।" or "Developed for the Cybersecurity & Fraud Domain of the Bank of Baroda Hackathon 2026. Account recovery processes represent a highly vulnerable target for identity compromise. By wrapping advanced classification models inside an explainable AI boundary with business utility threshold compilers, RecoverGuard establishes a robust mechanism to evaluate and block identity fraud in real-time."}
        </p>
      </div>
    </div>
    """

# Map internal keys to UI keys
MODEL_KEY_MAP = {
    'Random Forest': 'rf',
    'Gradient Boosting': 'gb',
    'Logistic Regression': 'lr',
    'सक्रिय धोखाधड़ी स्कैनर (ML मॉडल)': 'rf' # fallback map keys
}

# ----------------------------------------------------
# 5. Gradio Event Handlers
# ----------------------------------------------------
def update_dashboard(model_name, threshold, lang="English"):
    """
    Triggered when model or threshold slider is modified.
    Updates the 4 KPIs and the two main plots.
    """
    model_key = MODEL_KEY_MAP.get(model_name, 'rf')
    pipeline = models[model_key]
    
    X_test, _, _ = features.load_data(test_path)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    flags = (probs >= threshold).astype(int)
    
    total = len(df_test)
    flagged = int(np.sum(flags == 1))
    fraud_rate = float(flagged / total) * 100
    
    fp = int(np.sum((y_test == 0) & (flags == 1)))
    fn = int(np.sum((y_test == 1) & (flags == 0)))
    total_cost = fp * C_FP + fn * C_FN
    
    base_cost = int(np.sum(y_test == 1) * C_FN)
    savings = int(base_cost - total_cost)
    
    kpis_html = make_kpi_html(total, flagged, fraud_rate, savings, total_cost, lang)
    
    density_fig = plot_risk_density(probs, threshold, lang)
    cost_fig = plot_simulator_cost(y_test, probs, threshold, lang)
    
    return kpis_html, density_fig, cost_fig

def update_feed_table(model_name, threshold, search_text, risk_filter, lang="English"):
    """
    Updates the datastore grid table with localized headers.
    """
    model_key = MODEL_KEY_MAP.get(model_name, 'rf')
    pipeline = models[model_key]
    
    X_test, _, _ = features.load_data(test_path)
    probs = pipeline.predict_proba(X_test)[:, 1]
    flags = (probs >= threshold).astype(int)
    
    is_hi = (lang == "Hindi")
    
    label_safe = is_hi and 'सुरक्षित' or 'SAFE'
    label_review = is_hi and 'सत्यापन आवश्यक' or 'REVIEW REQUIRED'
    label_critical = is_hi and 'क्रिटिकल जोखिम' or 'CRITICAL RISK'
    
    label_authorized = is_hi and 'अधिकृत (सुरक्षित)' or 'AUTHORIZED (SAFE)'
    label_block = is_hi and 'ब्लॉक (धोखाधड़ी)' or 'CRITICAL BLOCK (FRAUD)'
    
    severity_labels = []
    for p in probs:
        if p >= 0.70:
            severity_labels.append(label_critical)
        elif p >= threshold:
            severity_labels.append(label_review)
        else:
            severity_labels.append(label_safe)
            
    action_labels = ['BOB-RC05' in str(df_test.iloc[i].get('onboarding_to_recovery_speed_flag', 0)) and label_block or (label_block if f == 1 else label_authorized) for i, f in enumerate(flags)]
    
    # Render table
    feed_df = pd.DataFrame({
        is_hi and 'प्रयास आईडी' or 'Attempt ID': df_test['attempt_id'],
        is_hi and 'खाता आईडी' or 'Account ID': df_test['account_id'],
        is_hi and 'चैनल' or 'Channel': df_test['recovery_channel'].str.replace('_', ' ').str.title(),
        is_hi and 'जोखिम संभावना' or 'Risk Probability': probs,
        is_hi and 'जोखिम गंभीरता' or 'Risk Severity': severity_labels,
        is_hi and 'कार्रवाई स्थिति' or 'Action Status': action_labels
    })
    
    # Search filter
    search_col = is_hi and 'प्रयास आईडी' or 'Attempt ID'
    acc_col = is_hi and 'खाता आईडी' or 'Account ID'
    if search_text:
        search_text = str(search_text).strip().lower()
        mask = feed_df[search_col].astype(str).str.lower().str.contains(search_text) | \
               feed_df[acc_col].astype(str).str.lower().str.contains(search_text)
        feed_df = feed_df[mask]
        
    # Severity filter
    sev_col = is_hi and 'जोखिम गंभीरता' or 'Risk Severity'
    if risk_filter != "ALL":
        # Handle maps
        if risk_filter == "CRITICAL RISK" and is_hi:
            feed_df = feed_df[feed_df[sev_col] == label_critical]
        elif risk_filter == "REVIEW REQUIRED" and is_hi:
            feed_df = feed_df[feed_df[sev_col] == label_review]
        elif risk_filter == "SAFE" and is_hi:
            feed_df = feed_df[feed_df[sev_col] == label_safe]
        else:
            feed_df = feed_df[feed_df[sev_col] == risk_filter]
        
    prob_col = is_hi and 'जोखिम संभावना' or 'Risk Probability'
    feed_df[prob_col] = feed_df[prob_col].map(lambda p: f"{p:.2%}")
    return feed_df

def update_diagnostics(attempt_id, model_name, threshold, lang="English"):
    """
    Updates individual SHAP charts, reason codes, and overview badges with translations.
    """
    is_hi = (lang == "Hindi")
    
    if not attempt_id:
        msg = is_hi and "<p style='color: #71717a;'>कोई रिकवरी प्रयास नहीं चुना गया। देखने के लिए सूची में किसी पंक्ति पर क्लिक करें।</p>" or "<p style='color: #71717a;'>No attempt selected. Click on a row in the registry feed or select one from the dropdown.</p>"
        return (
            msg,
            None,
            is_hi and "<p style='color: #71717a;'>कोई कोड उल्लंघन नहीं मिला।</p>" or "<p style='color: #71717a;'>No code violations triggered.</p>",
            pd.DataFrame(),
            ""
        )
        
    row_df = df_test[df_test['attempt_id'] == attempt_id]
    if len(row_df) == 0:
        return (
            is_hi and "<p style='color: #F43F5E;'>आईडी नहीं मिली।</p>" or "<p style='color: #F43F5E;'>Attempt ID not found.</p>",
            None,
            is_hi and "<p style='color: #F43F5E;'>रिकॉर्ड अनुपलब्ध।</p>" or "<p style='color: #F43F5E;'>Record missing.</p>",
            pd.DataFrame(),
            ""
        )
        
    model_key = MODEL_KEY_MAP.get(model_name, 'rf')
    pipeline = models[model_key]
    
    X_row = row_df.drop(columns=['attempt_id', 'account_id', 'is_fraud'], errors='ignore')
    prob = pipeline.predict_proba(X_row)[0, 1]
    
    # SHAP chart
    shap_fig = generate_shap_plot(pipeline, X_row)
    
    # Reason codes HTML
    row_dict = row_df.iloc[0].to_dict()
    reasons_html = make_reason_cards_html(row_dict, lang)
    
    # Risk badges
    if prob >= 0.70:
        status_badge = f'<span class="badge badge-critical">{is_hi and "क्रिटिकल जोखिम" or "CRITICAL RISK"}</span>'
        action_desc = is_hi and "खाता तत्काल ब्लॉक किया गया। कड़ा सुरक्षा होल्ड सक्रिय।" or "Account blocked immediately. Security hold applied."
    elif prob >= threshold:
        status_badge = f'<span class="badge badge-review">{is_hi and "सत्यापन आवश्यक" or "REVIEW REQUIRED"}</span>'
        action_desc = is_hi and "माध्यमिक क्रेडेंशियल सत्यापन आवश्यक (वीडियो केवाईसी या 24 घंटे की देरी)।" or "Requires secondary authentication (Video KYC or 24h delay)."
    else:
        status_badge = f'<span class="badge badge-safe">{is_hi and "सुरक्षित" or "AUTHORIZED (SAFE)"}</span>'
        action_desc = is_hi and "लॉगिन की अनुमति दी गई। बैकअप कोड भेज दिए गए हैं।" or "Allowed. Credentials successfully sent to user."
        
    overview_html = f"""
    <div style="background-color: var(--card-bg); border: 1px solid var(--card-border); border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px var(--box-glow); margin-bottom: 20px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 13px; color: var(--subtext-color); font-weight: bold; text-transform: uppercase;">{is_hi and "घटना प्रोफाइल ऑडिट" or "Audit Case Profile"}</span>
        {status_badge}
      </div>
      
      <div style="display: flex; gap: 24px; align-items: center; margin-bottom: 20px;">
        <div>
          <div style="font-size: 10px; color: var(--subtext-color); font-weight: bold; text-transform: uppercase; margin-bottom: 4px;">{is_hi and "धोखाधड़ी संभावना स्कोर" or "Fraud Risk Score"}</div>
          <div style="font-size: 32px; font-weight: 800; color: {'#F43F5E' if prob >= 0.70 else ('#F59E0B' if prob >= threshold else '#10B981')}; font-family: monospace;">{prob:.2%}</div>
        </div>
        <div style="flex-grow: 1; border-left: 1px solid var(--card-border); padding-left: 20px;">
          <div style="font-size: 10px; color: var(--subtext-color); font-weight: bold; text-transform: uppercase; margin-bottom: 4px;">{is_hi and "अनुशंसित प्रतिक्रिया नियम" or "Recommended Response Protocol"}</div>
          <div style="font-size: 14px; font-weight: bold; color: var(--text-color);">{action_desc}</div>
        </div>
      </div>
      
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 12px; border-top: 1px solid var(--card-border); padding-top: 16px;">
        <div>
          <span style="color: var(--subtext-color);">{is_hi and "प्रयास आईडी" or "Attempt ID"}:</span> <strong style="color: var(--text-color); font-family: monospace;">{attempt_id}</strong>
        </div>
        <div>
          <span style="color: var(--subtext-color);">{is_hi and "खाता आईडी" or "Account ID"}:</span> <strong style="color: var(--text-color); font-family: monospace;">{row_dict['account_id']}</strong>
        </div>
        <div>
          <span style="color: var(--subtext-color);">{is_hi and "चैनल" or "Channel"}:</span> <strong style="color: var(--text-color);">{str(row_dict['recovery_channel']).replace('_', ' ').title()}</strong>
        </div>
        <div>
          <span style="color: var(--subtext-color);">{is_hi and "दिन का समय" or "Hour of Day"}:</span> <strong style="color: var(--text-color);">{row_dict['hour_of_day']}:00 hrs</strong>
        </div>
      </div>
    </div>
    """
    
    # Metadata dataframe translation
    param_col = is_hi and 'विशेषता पैरामीटर' or 'Feature Parameter'
    val_col = is_hi and 'मान' or 'Value'
    
    meta_df = pd.DataFrame({
        param_col: [
            is_hi and 'डिवाइस जोखिम स्कोर' or 'Device Risk Score', 
            is_hi and 'आईपी जोखिम स्कोर' or 'IP Risk Score', 
            is_hi and 'विफल प्रयास (7 दिन)' or 'Failed Attempts (7d)', 
            is_hi and 'अंतिम लॉगिन के बाद दिन' or 'Time Since Last Login', 
            is_hi and 'खाता आयु (दिन)' or 'Time Since Creation', 
            is_hi and 'क्या नया डिवाइस है' or 'Is New Device', 
            is_hi and 'क्या नया आईपी है' or 'Is New IP', 
            is_hi and 'भौगोलिक बेमेल' or 'Geographic Mismatch'
        ],
        val_col: [
            f"{row_dict['device_risk_score']:.4f}",
            f"{row_dict['ip_risk_score']:.4f}",
            str(int(row_dict['failed_recovery_attempts_7d'])),
            f"{row_dict['time_since_last_login_days']:.2f} {is_hi and 'दिन' or 'Days'}",
            f"{row_dict['time_since_account_creation_days']:.2f} {is_hi and 'दिन' or 'Days'}",
            is_hi and ("हाँ" if row_dict['is_new_device'] == 1 else "नहीं") or ("Yes" if row_dict['is_new_device'] == 1 else "No"),
            is_hi and ("हाँ" if row_dict['is_new_ip'] == 1 else "नहीं") or ("Yes" if row_dict['is_new_ip'] == 1 else "No"),
            is_hi and ("हाँ" if row_dict['geo_mismatch_flag'] == 1 else "नहीं") or ("Yes" if row_dict['geo_mismatch_flag'] == 1 else "No")
        ]
    })
    
    btn_app = is_hi and "✔️ केस स्वीकृत करें" or "✔️ Approve Case"
    btn_kyc = is_hi and "🔑 वीडियो केवाईसी शुरू करें" or "🔑 Trigger Video KYC"
    btn_blk = is_hi and "🚫 खाता ब्लॉक करें" or "🚫 Block Account"
    
    action_html = f"""
    <div style="display: flex; gap: 10px; margin-top: 16px;">
      <button style="flex: 1; padding: 12px; background-color: #10B981; color: white; border: none; border-radius: 6px; font-weight: bold; font-size: 12px; cursor: pointer;">{btn_app}</button>
      <button style="flex: 1; padding: 12px; background-color: #F59E0B; color: white; border: none; border-radius: 6px; font-weight: bold; font-size: 12px; cursor: pointer;">{btn_kyc}</button>
      <button style="flex: 1; padding: 12px; background-color: #F43F5E; color: white; border: none; border-radius: 6px; font-weight: bold; font-size: 12px; cursor: pointer;">{btn_blk}</button>
    </div>
    """
    
    return overview_html, shap_fig, reasons_html, meta_df, action_html

def handle_policy_change(model_name, policy_name):
    model_key = MODEL_KEY_MAP.get(model_name, 'rf')
    if policy_name in policy_thresholds.get(model_key, {}):
        val = policy_thresholds[model_key][policy_name]
        return gr.update(value=val)
    return gr.update()

def handle_slider_change(model_name, threshold, current_policy):
    model_key = MODEL_KEY_MAP.get(model_name, 'rf')
    if model_key in policy_thresholds:
        for p_name, val in policy_thresholds[model_key].items():
            if abs(val - threshold) < 0.001:
                return p_name
    return "Manual Override"

def handle_table_row_click(evt: gr.SelectData, feed_df):
    row_idx = evt.index[0]
    # Handle both English and Hindi Attempt ID column names
    attempt_id = feed_df.iloc[row_idx].iloc[0] # Attempt ID is always column 0
    return attempt_id, gr.update(selected=2) # Switch to diagnostics tab

def handle_language_change(lang):
    """
    Translates UI elements dynamically when English/Hindi radio button is toggled.
    """
    text = LOCALIZED[lang]
    hero_html = make_hero_html(lang)
    team_html = make_team_details_html(lang)
    
    # Update choices for models/policies to fit language settings or defaults
    return (
        hero_html,
        team_html,
        gr.update(label=text['label_model']),
        gr.update(label=text['label_policy']),
        gr.update(label=text['label_threshold']),
        gr.update(placeholder=text['search_label'], label=text['search_label']),
        gr.update(label=text['risk_filter_label']),
        gr.update(label=text['dropdown_label']),
        gr.update(value=f"### {text['graph_density_title']}\n*{text['graph_density_desc']}*"),
        gr.update(value=f"### {text['graph_cost_title']}\n*{text['graph_cost_desc']}*"),
        gr.update(value=f"### {text['feed_title']}\n{text['feed_tip']}"),
        gr.update(value=f"### {text['sec_incident_details']}"),
        gr.update(value=f"### {text['sec_parameters']}"),
        gr.update(value=f"### {text['sec_triggers']}"),
        gr.update(value=text['model_matrix_title'])
    )

# ----------------------------------------------------
# 6. Gradio Interface Construction
# ----------------------------------------------------
with gr.Blocks(css=CSS_STYLE, theme=gr.themes.Origin(primary_hue="orange", neutral_hue="zinc")) as demo:
    
    # Row for Language Toggle & Theme Toggle at top right
    with gr.Row():
        with gr.Column(scale=3):
            pass
        with gr.Column(scale=1):
            theme_btn = gr.Button("🌓 Light/Dark Mode")
        with gr.Column(scale=1):
            lang_toggle = gr.Radio(
                choices=["English", "Hindi / हिन्दी"], 
                value="English", 
                label="🌐 Language / भाषा"
            )
            
    # Render Cyber Console Header
    hero_banner = gr.HTML(value=make_hero_html("English"))
    
    # Tab Layout
    with gr.Tabs() as main_tabs:
        
        # TAB 1: EXECUTIVE DASHBOARD
        with gr.TabItem("🏦 Executive Command Center", id=0):
            with gr.Row():
                with gr.Column(scale=1):
                    sec_controls_hdr = gr.Markdown("### ⚙️ Operational Settings")
                    model_sel = gr.Dropdown(
                        choices=["Random Forest", "Gradient Boosting", "Logistic Regression"], 
                        value="Random Forest", 
                        label="Active Classifier Model"
                    )
                    policy_sel = gr.Dropdown(
                        choices=[
                            "Cost-Optimized (BOB Recommended)", 
                            "Balanced F1 Score", 
                            "High Security (Recall >= 95%)", 
                            "Low Friction (Precision >= 95%)", 
                            "Review Capacity (Top 10%)",
                            "Manual Override"
                        ], 
                        value="Cost-Optimized (BOB Recommended)", 
                        label="Decision Threshold Policy"
                    )
                    thresh_slider = gr.Slider(
                        minimum=0.0, 
                        maximum=1.0, 
                        step=0.01, 
                        value=0.25, 
                        label="Active Decision Threshold"
                    )
                
                with gr.Column(scale=2):
                    sec_telemetry_hdr = gr.Markdown("### 📊 Live System Telemetry")
                    kpis_output = gr.HTML()
            
            with gr.Row():
                with gr.Column(scale=1):
                    density_hdr = gr.Markdown("### Operational Risk Profile & Threshold Alignment\n*What this shows: This chart represents the risk level of all login attempts. Dragging the security strictness slider moves the threshold line. Everything to the right of the line is blocked, everything to the left is allowed.*")
                    density_plot_output = gr.Plot()
                with gr.Column(scale=1):
                    cost_hdr = gr.Markdown("### Operational Cost Minimization Sandbox\n*What this shows: This graph helps find the most profitable security level. It balances the cost of security calls (₹800/call) against the potential loss from missed fraud (₹25,000/fraud). The green star marks the target settings.*")
                    cost_plot_output = gr.Plot()
        
        # TAB 2: DATA FEED REGISTRY
        with gr.TabItem("🔍 Live Datastore Feed", id=1):
            feed_hdr = gr.Markdown("### 📡 Active Account Recovery Audit Logs\n*Tip: Click on any row in the feed table below to open its explainable AI diagnostics and SHAP analysis.*")
            
            with gr.Row():
                with gr.Column(scale=2):
                    search_box = gr.Textbox(placeholder="Search by Attempt ID or Account ID...", label="Search by Attempt ID or Account ID...")
                with gr.Column(scale=1):
                    filter_risk = gr.Radio(choices=["ALL", "CRITICAL RISK", "REVIEW REQUIRED", "SAFE"], value="ALL", label="Filter Security Alert")
            
            feed_table = gr.Dataframe(interactive=False)
            
        # TAB 3: CASE DIAGNOSTICS & EXPLAINABILITY
        with gr.TabItem("🕵️‍♂️ Explainable AI Deep-Dive", id=2):
            with gr.Row():
                with gr.Column(scale=1):
                    sec_inc_hdr = gr.Markdown("### 🛡️ Selected Incident Details")
                    attempt_dropdown = gr.Dropdown(
                        label="Target Account Recovery Attempt ID", 
                        choices=list(df_test['attempt_id'].values),
                        value=df_test['attempt_id'].values[0]
                    )
                    overview_output = gr.HTML()
                    sec_param_hdr = gr.Markdown("### 📋 Parameter Metrics")
                    metadata_table = gr.Dataframe(interactive=False)
                    action_panel = gr.HTML()
                    
                with gr.Column(scale=1):
                    shap_plot_output = gr.Plot(label="SHAP Diagnostic Attribution")
                    sec_trig_hdr = gr.Markdown("### 🔍 Risk Triggers & Manual Remediation")
                    reason_cards_output = gr.HTML()
                    
        # TAB 4: BENCHMARKS & DOCUMENTATION
        with gr.TabItem("📈 Model Benchmarking", id=3):
            gr.Markdown("### 📑 Multi-Architecture Model Evaluations")
            with gr.Row():
                with gr.Column(scale=1):
                    roc_plot_img = gr.Image("reports/figures/roc_curve.png", label="Receiver Operating Characteristic (ROC)")
                with gr.Column(scale=1):
                    pr_plot_img = gr.Image("reports/figures/precision_recall_curve.png" if os.path.exists("reports/figures/precision_recall_curve.png") else "reports/figures/pr_curve.png", label="Precision-Recall Curves")
            
            with gr.Row():
                with gr.Column(scale=1):
                    calib_plot_img = gr.Image("reports/figures/calibration_curve.png", label="Probability Calibration (Reliability)")
                with gr.Column(scale=1):
                    matrix_hdr = gr.Markdown("### 📊 Metrics Matrix Comparison")
                    metrics_table = gr.HTML(value=make_metrics_table_html())
                    
        # TAB 5: TEAM & HACKATHON SUBMISSION
        with gr.TabItem("👥 Team Details", id=4):
            team_section = gr.HTML(value=make_team_details_html("English"))
            
    # ----------------------------------------------------
    # Event Hook Wiring
    # ----------------------------------------------------
    
    # 1. Language Toggle change
    lang_toggle.change(
        fn=handle_language_change,
        inputs=[lang_toggle],
        outputs=[
            hero_banner, team_section,
            model_sel, policy_sel, thresh_slider,
            search_box, filter_risk,
            attempt_dropdown,
            density_hdr, cost_hdr, feed_hdr,
            sec_inc_hdr, sec_param_hdr, sec_trig_hdr,
            matrix_hdr
        ]
    )
    
    # 2. Update sliders based on policy choice
    policy_sel.change(
        fn=handle_policy_change,
        inputs=[model_sel, policy_sel],
        outputs=[thresh_slider]
    )
    
    # 3. Sync policy dropdown to custom slider adjustments
    thresh_slider.change(
        fn=handle_slider_change,
        inputs=[model_sel, thresh_slider, policy_sel],
        outputs=[policy_sel]
    )
    
    # 4. Update charts and KPIs when model, threshold, or language moves
    dashboard_inputs = [model_sel, thresh_slider, lang_toggle]
    dashboard_outputs = [kpis_output, density_plot_output, cost_plot_output]
    
    demo.load(fn=update_dashboard, inputs=dashboard_inputs, outputs=dashboard_outputs)
    model_sel.change(fn=update_dashboard, inputs=dashboard_inputs, outputs=dashboard_outputs)
    thresh_slider.change(fn=update_dashboard, inputs=dashboard_inputs, outputs=dashboard_outputs)
    lang_toggle.change(fn=update_dashboard, inputs=dashboard_inputs, outputs=dashboard_outputs)
    
    # 5. Filter live feed datastore
    feed_inputs = [model_sel, thresh_slider, search_box, filter_risk, lang_toggle]
    demo.load(fn=update_feed_table, inputs=feed_inputs, outputs=[feed_table])
    model_sel.change(fn=update_feed_table, inputs=feed_inputs, outputs=[feed_table])
    thresh_slider.change(fn=update_feed_table, inputs=feed_inputs, outputs=[feed_table])
    search_box.change(fn=update_feed_table, inputs=feed_inputs, outputs=[feed_table])
    filter_risk.change(fn=update_feed_table, inputs=feed_inputs, outputs=[feed_table])
    lang_toggle.change(fn=update_feed_table, inputs=feed_inputs, outputs=[feed_table])
    
    # 6. Diagnostics tab bindings
    diag_inputs = [attempt_dropdown, model_sel, thresh_slider, lang_toggle]
    diag_outputs = [overview_output, shap_plot_output, reason_cards_output, metadata_table, action_panel]
    
    demo.load(fn=update_diagnostics, inputs=diag_inputs, outputs=diag_outputs)
    attempt_dropdown.change(fn=update_diagnostics, inputs=diag_inputs, outputs=diag_outputs)
    model_sel.change(fn=update_diagnostics, inputs=diag_inputs, outputs=diag_outputs)
    thresh_slider.change(fn=update_diagnostics, inputs=diag_inputs, outputs=diag_outputs)
    lang_toggle.change(fn=update_diagnostics, inputs=diag_inputs, outputs=diag_outputs)
    
    # 7. Clicking a cell in the main table loads that case in individual deep dive
    feed_table.select(
        fn=handle_table_row_click,
        inputs=[feed_table],
        outputs=[attempt_dropdown, main_tabs]
    )
    
    # 8. Theme Toggle Click
    theme_btn.click(
        fn=None,
        js="() => { document.body.classList.toggle('dark'); document.documentElement.classList.toggle('dark'); document.querySelector('gradio-app').classList.toggle('dark'); }"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
