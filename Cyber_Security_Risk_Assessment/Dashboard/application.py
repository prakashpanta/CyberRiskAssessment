import os
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SME Cyber Risk Intelligence Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../Models"))

# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.stApp {
    background: radial-gradient(circle at top right, rgba(56, 189, 248, 0.07), transparent 28%), #07111f;
    color: #f8fafc;
}

section[data-testid="stSidebar"] {
    background: #0b1727;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc;
}

.header-box {
    background: linear-gradient(135deg, rgba(30, 64, 175, 0.35), rgba(15, 23, 42, 0.95));
    border: 1px solid #263b57;
    border-radius: 18px;
    padding: 30px 32px;
    margin-bottom: 24px;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25);
}

.header-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f8fafc;
    margin: 0;
}

.header-subtitle {
    margin-top: 8px;
    font-size: 0.95rem;
    line-height: 1.6;
    color: #94a3b8;
}

.section-title {
    color: #38bdf8;
    font-size: 0.85rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 12px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.kpi-card {
    background: #0b1727;
    border: 1px solid #20324a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.kpi-label {
    color: #7f93aa;
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.kpi-value {
    color: #f8fafc;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 5px;
}

.model-card {
    background: #0b1727;
    border: 1px solid #263b57;
    border-radius: 12px;
    padding: 15px 18px;
    margin-bottom: 12px;
}

.model-name {
    color: #f8fafc;
    font-size: 0.95rem;
    font-weight: 700;
}

.model-type {
    color: #8497ad;
    font-size: 0.75rem;
    margin-top: 4px;
}

.model-accuracy {
    color: #38bdf8;
    font-size: 1.05rem;
    font-weight: 800;
    margin-top: 7px;
}

.recommended {
    border: 1px solid #22c55e;
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.08), rgba(15, 23, 42, 0.9));
}

.recommended-badge {
    display: inline-block;
    background: rgba(34, 197, 94, 0.12);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-radius: 999px;
    padding: 4px 9px;
    font-size: 0.65rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.risk-banner {
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    margin: 18px 0;
}

.risk-high {
    background: rgba(239, 68, 68, 0.10);
    border: 1px solid rgba(239, 68, 68, 0.55);
    color: #fca5a5;
}

.risk-medium {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.55);
    color: #fde68a;
}

.risk-low {
    background: rgba(34, 197, 94, 0.10);
    border: 1px solid rgba(34, 197, 94, 0.55);
    color: #86efac;
}

.action-row {
    background: #0b1727;
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 12px 15px;
    margin-bottom: 9px;
    color: #cbd5e1;
    font-size: 0.86rem;
    line-height: 1.55;
}

.confidence-box {
    background: #0b1727;
    border: 1px solid #20324a;
    border-radius: 10px;
    padding: 13px 16px;
    margin-top: 14px;
    color: #94a3b8;
    font-size: 0.82rem;
}

.confidence-value {
    color: #f8fafc;
    font-weight: 800;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.72rem;
    margin-top: 35px;
    padding: 15px;
    border-top: 1px solid #1e293b;
}

label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
}

div[data-baseweb="input"],
div[data-baseweb="select"] {
    border-radius: 9px;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 48px;
}

@media (max-width: 900px) {
    .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .header-title {
        font-size: 1.5rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# 4. LOAD MODELS
# ============================================================


@st.cache_resource
def load_all_models():
    def get_model(filename):
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    models_dict = {
        "Decision Tree": {
            "model": get_model("decision_tree.pkl"),
            "accuracy": 76.44,
            "type": "Rule-Based • Interpretable",
        },
        "Random Forest": {
            "model": get_model("random_forest.pkl"),
            "accuracy": 92.25,
            "type": "Ensemble • Bagging",
        },
        "XGBoost": {
            "model": get_model("xgboost.pkl"),
            "accuracy": 92.19,
            "type": "Gradient Boosting",
        },
        "Stacking Ensemble": {
            "model": get_model("Augmented_stacking_model.pkl"),
            "accuracy": 95.75,
            "type": "DT + RF + XGB",
        },
    }

    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    return models_dict, encoder


MODELS, label_encoder = load_all_models()

# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div style="font-size:1.2rem; font-weight:800; color:#f8fafc; margin-bottom:4px;">🛡️ Cyber Risk Intelligence</div>'
        '<div style="font-size:0.75rem; color:#7f93aa; margin-bottom:20px;">SME Decision Support Platform</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Model Selection")
    selected_model = st.selectbox(
        "Choose inference model",
        options=list(MODELS.keys()),
        index=3,
    )

    if selected_model == "Stacking Ensemble":
        st.success(
            "Recommended model\n\nHighest validation accuracy among the implemented models."
        )

    st.markdown("---")
    st.markdown("### Implemented Models")

    for model_name, model_info in MODELS.items():
        recommended = model_name == "Stacking Ensemble"
        badge = " ⭐ Recommended" if recommended else ""
        sidebar_html = (
            '<div style="padding:8px 0; border-bottom:1px solid #1e293b;">'
            f'<div style="color:#e2e8f0; font-size:0.78rem; font-weight:600;">{model_name}{badge}</div>'
            f'<div style="color:#64748b; font-size:0.68rem;">{model_info["accuracy"]:.2f}% accuracy</div>'
            "</div>"
        )
        st.markdown(sidebar_html, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("A Data-Driven Cyber Risk Assessment Framework for SMEs")

# ============================================================
# 6. HEADER
# ============================================================

st.markdown(
    '<div class="header-box">'
    '<div class="header-title">🛡️ SME Cyber Risk Assessment</div>'
    '<div class="header-subtitle">Data-driven cyber risk assessment using socio-technical risk modelling and machine learning classification.</div>'
    "</div>",
    unsafe_allow_html=True,
)

# ============================================================
# 7. INPUT SECTION
# ============================================================

input_col1, input_col2 = st.columns([1, 1], gap="large")

with input_col1:
    st.markdown(
        '<div class="section-title">💥 Operational & Financial Impact</div>',
        unsafe_allow_html=True,
    )

    financial_loss = st.number_input(
        "Financial Loss (USD)",
        min_value=0.0,
        max_value=100000.0,
        value=10000.0,
        step=1000.0,
    )

    operational_disruption = st.number_input(
        "Operational Disruption (Hours)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )

    reputation_damage = st.slider(
        "Reputation Damage Score",
        min_value=0,
        max_value=12,
        value=6,
        help="0 = No damage, 12 = Critical damage",
    )

    response_col, recovery_col = st.columns(2)
    with response_col:
        incident_response_time = st.number_input(
            "Response Time (Hours)",
            min_value=0.0,
            max_value=300.0,
            value=20.0,
            step=2.0,
        )
    with recovery_col:
        recovery_time = st.number_input(
            "Recovery Time (Days)",
            min_value=0.0,
            max_value=15.0,
            value=2.0,
            step=0.5,
        )

with input_col2:
    st.markdown(
        '<div class="section-title">🛡️ Socio-Technical Resilience</div>',
        unsafe_allow_html=True,
    )

    cybersecurity_budget = st.number_input(
        "Cybersecurity Budget (USD)",
        min_value=0.0,
        max_value=600000.0,
        value=100000.0,
        step=5000.0,
    )

    st.markdown(
        '<div style="color:#7f93aa; font-size:0.76rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; margin:15px 0 9px 0;">Security Controls</div>',
        unsafe_allow_html=True,
    )

    employee_training = st.checkbox("Employee Security Awareness Training")
    use_of_mfa = st.checkbox("Multi-Factor Authentication (MFA)")
    data_backup = st.checkbox("Data Backup Availability")

# ============================================================
# 8. NORMALISATION CONSTANTS
# ============================================================

MAX_FINANCIAL_LOSS = 60125.44672268917
MAX_OPERATIONAL_DISRUPTION = 57.985857605582126
MAX_REPUTATION_DAMAGE = 12.0
MAX_INCIDENT_RESPONSE_TIME = 213.38721219642844
MAX_RECOVERY_TIME = 8.775810710089594
MAX_CYBERSECURITY_BUDGET = 557320.5858869473

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button(
    "🚀 Run Cyber Risk Assessment",
    use_container_width=True,
    type="primary",
)

# ============================================================
# 9. CALCULATION & PREDICTION
# ============================================================

if predict_btn:
    selected_model_entry = MODELS.get(selected_model, {})
    selected_model_object = selected_model_entry.get("model")

    if selected_model_object is None:
        st.error(
            f"Model file for '{selected_model}' was not found in `{MODEL_DIR}`. Please verify your model files."
        )
    else:
        financial_loss_norm = min(financial_loss / MAX_FINANCIAL_LOSS, 1.0)
        operational_disruption_norm = min(
            operational_disruption / MAX_OPERATIONAL_DISRUPTION, 1.0
        )
        reputation_damage_norm = min(
            reputation_damage / MAX_REPUTATION_DAMAGE, 1.0
        )
        incident_response_norm = min(
            incident_response_time / MAX_INCIDENT_RESPONSE_TIME, 1.0
        )
        recovery_time_norm = min(recovery_time / MAX_RECOVERY_TIME, 1.0)
        cybersecurity_budget_norm = min(
            cybersecurity_budget / MAX_CYBERSECURITY_BUDGET, 1.0
        )

        technical_score = 0.5 * int(use_of_mfa) + 0.5 * int(data_backup)
        human_score = float(int(employee_training))
        org_score = cybersecurity_budget_norm

        impact_score = (
            0.4 * financial_loss_norm
            + 0.2 * operational_disruption_norm
            + 0.2 * reputation_damage_norm
            + 0.1 * incident_response_norm
            + 0.1 * recovery_time_norm
        )

        control_score = (
            0.4 * technical_score + 0.3 * human_score + 0.3 * org_score
        )
        risk_score = max(0.0, impact_score - control_score)

        input_data = pd.DataFrame(
            [[
                financial_loss_norm,
                operational_disruption_norm,
                reputation_damage_norm,
                incident_response_norm,
                recovery_time_norm,
                cybersecurity_budget_norm,
                int(employee_training),
                int(use_of_mfa),
                int(data_backup),
                technical_score,
                human_score,
                org_score,
            ]],
            columns=[
                "Financial_Loss",
                "Operational_Disruption",
                "Reputation_Damage_Score",
                "Incident_Response_Time",
                "Recovery_Time",
                "Cybersecurity_Budget",
                "Employee_Training",
                "Use_of_MFA",
                "Data_Backup_Availability",
                "Technical_Score",
                "Human_Score",
                "Org_Score",
            ],
        )

        prediction = selected_model_object.predict(input_data)[0]

        if label_encoder is not None:
            try:
                risk_level = label_encoder.inverse_transform([prediction])[0]
            except Exception:
                risk_level = str(prediction)
        else:
            risk_level = str(prediction)

        risk_level = str(risk_level).strip().title()

        confidence = None
        if hasattr(selected_model_object, "predict_proba"):
            try:
                probabilities = selected_model_object.predict_proba(input_data)[
                    0
                ]
                confidence = max(probabilities) * 100
            except Exception:
                confidence = None

        st.markdown("---")
        results_col1, results_col2 = st.columns([1.15, 0.85], gap="large")

        with results_col1:
            st.markdown(
                '<div class="section-title">🎯 Assessment Result</div>',
                unsafe_allow_html=True,
            )

            if risk_level == "High":
                st.markdown(
                    '<div class="risk-banner risk-high">🔴 HIGH CYBER RISK</div>',
                    unsafe_allow_html=True,
                )
                summary = "The assessment indicates a high residual cybersecurity risk. The estimated impact currently exceeds the organisation's resilience controls."
            elif risk_level == "Medium":
                st.markdown(
                    '<div class="risk-banner risk-medium">🟡 MEDIUM CYBER RISK</div>',
                    unsafe_allow_html=True,
                )
                summary = "The assessment indicates a moderate level of residual cybersecurity risk. Additional controls may be required to strengthen the organisation's security posture."
            else:
                st.markdown(
                    '<div class="risk-banner risk-low">🟢 LOW CYBER RISK</div>',
                    unsafe_allow_html=True,
                )
                summary = "The assessment indicates a comparatively low residual cybersecurity risk based on the supplied impact and resilience factors."

            st.markdown(
                f'<p style="color:#cbd5e1; font-size:0.88rem; line-height:1.65;"><b>Assessment Summary:</b> {summary}</p>',
                unsafe_allow_html=True,
            )

            kpi_html = (
                '<div class="kpi-grid">'
                f'<div class="kpi-card"><div class="kpi-label">Technical</div><div class="kpi-value">{technical_score:.2f}</div></div>'
                f'<div class="kpi-card"><div class="kpi-label">Human</div><div class="kpi-value">{human_score:.2f}</div></div>'
                f'<div class="kpi-card"><div class="kpi-label">Organisation</div><div class="kpi-value">{org_score:.2f}</div></div>'
                f'<div class="kpi-card"><div class="kpi-label">Impact</div><div class="kpi-value">{impact_score:.2f}</div></div>'
                "</div>"
            )
            st.markdown(kpi_html, unsafe_allow_html=True)

            risk_score_html = (
                '<div style="margin-top:18px; padding:15px; background:#0b1727; border:1px solid #20324a; border-radius:12px;">'
                '<div style="color:#7f93aa; font-size:0.70rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em;">Residual Risk Score</div>'
                f'<div style="color:#f8fafc; font-size:2rem; font-weight:800; margin-top:3px;">{risk_score:.3f}</div>'
                '<div style="color:#64748b; font-size:0.72rem; margin-top:3px;">Impact Score − Control Score</div>'
                "</div>"
            )
            st.markdown(risk_score_html, unsafe_allow_html=True)

            model_accuracy = selected_model_entry.get("accuracy", 0.0)
            confidence_text = (
                f"{confidence:.2f}%"
                if confidence is not None
                else "Not available"
            )

            conf_html = (
                '<div class="confidence-box">'
                f'<div>Inference Model: <span class="confidence-value">{selected_model}</span></div>'
                f'<div style="margin-top:5px;">Model Validation Accuracy: <span class="confidence-value">{model_accuracy:.2f}%</span></div>'
                f'<div style="margin-top:5px;">Prediction Confidence: <span class="confidence-value">{confidence_text}</span></div>'
                "</div>"
            )
            st.markdown(conf_html, unsafe_allow_html=True)

        with results_col2:
            st.markdown(
                '<div class="section-title">💡 Recommended Actions</div>',
                unsafe_allow_html=True,
            )

            actions = []
            if not use_of_mfa:
                actions.append(
                    "🔐 <b>Enable MFA:</b> Deploy multi-factor authentication across critical user and administrative accounts."
                )
            if not data_backup:
                actions.append(
                    "💾 <b>Strengthen Backup:</b> Implement reliable and tested backups using an appropriate 3-2-1 or equivalent strategy."
                )
            if not employee_training:
                actions.append(
                    "👥 <b>Improve Employee Awareness:</b> Conduct regular cybersecurity awareness and phishing-resistance training."
                )
            if cybersecurity_budget < 50000:
                actions.append(
                    "💼 <b>Review Security Investment:</b> Consider increasing cybersecurity resources according to the organisation's risk exposure."
                )
            if incident_response_time > 24:
                actions.append(
                    "⚡ <b>Improve Incident Response:</b> Establish and regularly test an incident response procedure."
                )
            if recovery_time > 2:
                actions.append(
                    "⏱️ <b>Improve Recovery Capability:</b> Test recovery procedures regularly and reduce the expected restoration time."
                )
            if financial_loss > 30000:
                actions.append(
                    "🛡️ <b>Reduce Financial Exposure:</b> Review business continuity arrangements and cyber insurance requirements."
                )

            if risk_level == "Low":
                actions = [
                    "✅ <b>Maintain Security Awareness:</b> Continue regular employee security training.",
                    "✅ <b>Review Controls:</b> Periodically review MFA, backups and access controls.",
                    "✅ <b>Test Recovery:</b> Conduct periodic incident-response and recovery exercises.",
                ]
            elif len(actions) == 0:
                actions.append(
                    "✅ <b>Maintain Current Controls:</b> Continue monitoring the organisation's cybersecurity posture and review controls periodically."
                )

            for action in actions:
                st.markdown(
                    f'<div class="action-row">{action}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div style="margin-top:16px; color:#64748b; font-size:0.70rem; line-height:1.5;">Recommendations are generated from the supplied assessment inputs and should be treated as decision-support guidance rather than a substitute for a professional cybersecurity assessment.</div>',
                unsafe_allow_html=True,
            )

# ============================================================
# 10. MODEL OVERVIEW
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">📊 Implemented Machine Learning Models</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#94a3b8; font-size:0.82rem; line-height:1.6; margin-bottom:18px;">Four classification approaches are available in the decision platform. Decision Tree provides the primary interpretable model, while Random Forest, XGBoost and Stacking Ensemble provide comparative ensemble approaches.</p>',
    unsafe_allow_html=True,
)

model_columns = st.columns(4)
for index, (model_name, model_info) in enumerate(MODELS.items()):
    with model_columns[index]:
        is_recommended = model_name == "Stacking Ensemble"
        card_class = (
            "model-card recommended" if is_recommended else "model-card"
        )
        badge = (
            '<div class="recommended-badge">⭐ Recommended</div>'
            if is_recommended
            else ""
        )

        card_html = (
            f'<div class="{card_class}">'
            f"{badge}"
            f'<div class="model-name">{model_name}</div>'
            f'<div class="model-type">{model_info["type"]}</div>'
            f'<div class="model-accuracy">{model_info["accuracy"]:.2f}%</div>'
            "</div>"
        )

        st.markdown(card_html, unsafe_allow_html=True)

# ============================================================
# 11. METHODOLOGY SUMMARY & FOOTER
# ============================================================

with st.expander("ℹ️ View Risk Scoring Methodology"):
    st.markdown(
        """
### Risk Scoring Framework
The dashboard uses the proposed residual cyber risk framework.

**Impact Score**  
Financial Loss, Operational Disruption, Reputation Damage, Incident Response Time and Recovery Time contribute to the estimated impact.

**Control Score**  
Technical, human and organisational resilience factors contribute to risk reduction.

**Residual Risk**  
Residual Risk = Impact Score − Control Score *(Negative residual scores are bounded at zero)*

The machine learning model then classifies the supplied feature profile into:
- 🟢 Low
- 🟡 Medium
- 🔴 High
"""
    )

st.markdown(
    '<div class="footer">SME Cyber Risk Intelligence Hub • Data-Driven Cyber Risk Assessment Framework<br>Decision-support prototype for academic research</div>',
    unsafe_allow_html=True,
)