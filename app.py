"""
Mini Digital Health Twin — Live Web Application
Author: Varaganti Vamshi | M.Sc. Industrial AI | Hochschule Albstadt-Sigmaringen
Built as part of HOPn Technical Assessment
"""

import streamlit as st
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mini Digital Health Twin",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #4CAF50;
    }
    .risk-critical { color: #e74c3c; font-weight: bold; }
    .risk-warning { color: #f39c12; font-weight: bold; }
    .risk-healthy { color: #2ecc71; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_patient_data():
    """Load all patient JSON files into a DataFrame."""
    DATA_PATH = "patient_data"
    patient_records = []
    json_files = sorted(glob.glob(os.path.join(DATA_PATH, "P*.json")))

    for filepath in json_files:
        with open(filepath, 'r') as f:
            patient = json.load(f)
            patient_records.append(patient)

    df = pd.DataFrame(patient_records)
    df = df.set_index('patient_id')
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# RISK ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_patient_risks(row):
    """Analyze a single patient against clinical thresholds."""
    risks = []

    # Heart Rate (AHA guidelines)
    if row['heart_rate'] > 120:
        risks.append({'metric': 'Heart Rate', 'value': f"{row['heart_rate']} bpm",
                      'severity': 'CRITICAL', 'detail': 'Severe tachycardia'})
    elif row['heart_rate'] > 100:
        risks.append({'metric': 'Heart Rate', 'value': f"{row['heart_rate']} bpm",
                      'severity': 'WARNING', 'detail': 'Tachycardia — above normal range'})
    elif row['heart_rate'] < 60:
        risks.append({'metric': 'Heart Rate', 'value': f"{row['heart_rate']} bpm",
                      'severity': 'WARNING', 'detail': 'Bradycardia — below normal range'})

    # Blood Pressure (AHA)
    if row['blood_pressure_systolic'] >= 140 or row['blood_pressure_diastolic'] >= 90:
        risks.append({'metric': 'Blood Pressure',
                      'value': f"{row['blood_pressure_systolic']}/{row['blood_pressure_diastolic']} mmHg",
                      'severity': 'CRITICAL', 'detail': 'Hypertension Stage 2'})
    elif row['blood_pressure_systolic'] >= 130 or row['blood_pressure_diastolic'] >= 80:
        risks.append({'metric': 'Blood Pressure',
                      'value': f"{row['blood_pressure_systolic']}/{row['blood_pressure_diastolic']} mmHg",
                      'severity': 'WARNING', 'detail': 'Elevated / Hypertension Stage 1'})

    # Sleep (National Sleep Foundation)
    if row['sleep_hours'] < 5:
        risks.append({'metric': 'Sleep', 'value': f"{row['sleep_hours']} hours",
                      'severity': 'CRITICAL', 'detail': 'Severe sleep deprivation'})
    elif row['sleep_hours'] < 6:
        risks.append({'metric': 'Sleep', 'value': f"{row['sleep_hours']} hours",
                      'severity': 'WARNING', 'detail': 'Below recommended 7-9 hours'})

    # Activity (CDC)
    if row['daily_steps'] < 3000:
        risks.append({'metric': 'Physical Activity', 'value': f"{row['daily_steps']} steps",
                      'severity': 'CRITICAL', 'detail': 'Very low activity level'})
    elif row['daily_steps'] < 5000:
        risks.append({'metric': 'Physical Activity', 'value': f"{row['daily_steps']} steps",
                      'severity': 'WARNING', 'detail': 'Sedentary lifestyle'})

    # BMI (WHO)
    if row['bmi'] >= 30:
        risks.append({'metric': 'BMI', 'value': f"{row['bmi']} kg/m²",
                      'severity': 'CRITICAL', 'detail': 'Obesity'})
    elif row['bmi'] >= 25:
        risks.append({'metric': 'BMI', 'value': f"{row['bmi']} kg/m²",
                      'severity': 'WARNING', 'detail': 'Overweight'})

    # Blood Glucose (ADA)
    if row['blood_glucose'] >= 126:
        risks.append({'metric': 'Blood Glucose', 'value': f"{row['blood_glucose']} mg/dL",
                      'severity': 'CRITICAL', 'detail': 'Diabetic range'})
    elif row['blood_glucose'] >= 100:
        risks.append({'metric': 'Blood Glucose', 'value': f"{row['blood_glucose']} mg/dL",
                      'severity': 'WARNING', 'detail': 'Pre-diabetic range'})

    # Stress
    if row['stress_level'] >= 8:
        risks.append({'metric': 'Stress Level', 'value': f"{row['stress_level']}/10",
                      'severity': 'CRITICAL', 'detail': 'Very high stress'})
    elif row['stress_level'] >= 6:
        risks.append({'metric': 'Stress Level', 'value': f"{row['stress_level']}/10",
                      'severity': 'WARNING', 'detail': 'Elevated stress'})

    return risks


def categorize_risk(score):
    if score == 0: return 'Healthy'
    elif score <= 2: return 'Low Risk'
    elif score <= 5: return 'Moderate Risk'
    else: return 'High Risk'


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

NUMERIC_COLS = ['heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                'sleep_hours', 'daily_steps', 'stress_level', 'bmi', 'blood_glucose']

@st.cache_data
def run_anomaly_detection(df):
    """Run Z-score and Isolation Forest anomaly detection."""
    # Z-Scores
    z_scores = pd.DataFrame(
        stats.zscore(df[NUMERIC_COLS]),
        columns=NUMERIC_COLS,
        index=df.index
    )

    # Isolation Forest
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[NUMERIC_COLS])
    iso_forest = IsolationForest(contamination=0.2, random_state=42, n_estimators=100)
    predictions = iso_forest.fit_predict(features_scaled)
    anomaly_scores = iso_forest.decision_function(features_scaled)

    df_result = df.copy()
    df_result['anomaly_label'] = predictions
    df_result['anomaly_score'] = anomaly_scores.round(4)

    return df_result, z_scores


# ═══════════════════════════════════════════════════════════════════════════════
# AI INSIGHT GENERATOR (Template-based)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_health_insight(row, risks):
    """Generate template-based health insights for a patient."""
    critical = [r for r in risks if r['severity'] == 'CRITICAL']
    warnings_list = [r for r in risks if r['severity'] == 'WARNING']
    is_anomaly = row.get('anomaly_label', 1) == -1
    name = row['name'].split()[0]

    # Overall assessment
    if len(critical) >= 3:
        overall = (f"{name} presents a high-risk profile with {len(critical)} critical indicators "
                   f"requiring prompt medical attention. "
                   f"{'ML analysis also flagged an unusual multi-metric pattern. ' if is_anomaly else ''}"
                   f"Clinical review is strongly recommended.")
    elif len(critical) >= 1:
        overall = (f"{name}'s health data shows {len(critical)} critical area(s) requiring targeted "
                   f"intervention alongside lifestyle adjustments. "
                   f"{'An unusual overall pattern was detected by anomaly detection. ' if is_anomaly else ''}"
                   f"Proactive management can significantly improve outcomes.")
    elif len(warnings_list) > 0:
        overall = (f"{name}'s overall health status is moderate with {len(warnings_list)} warning(s). "
                   f"Addressing these proactively will prevent escalation to critical levels.")
    else:
        overall = (f"{name} demonstrates an excellent health profile across all monitored metrics. "
                   f"Current vitals, metabolic markers, and lifestyle indicators are all within optimal ranges.")

    # Recommendations
    recs = []
    if row['heart_rate'] > 100:
        recs.append("Regular aerobic exercise (20-30 min brisk walking, 3x/week) to reduce resting heart rate.")
    if row['blood_pressure_systolic'] >= 130:
        recs.append("Adopt DASH diet — reduce sodium, increase potassium-rich foods (bananas, spinach, legumes).")
    if row['sleep_hours'] < 6:
        recs.append("Establish consistent sleep schedule. Avoid screens 60 minutes before bed.")
    if row['daily_steps'] < 5000:
        recs.append("Start with 4,000 steps daily and increase by 500 steps every two weeks.")
    if row['blood_glucose'] >= 100:
        recs.append("Reduce simple sugars and refined carbs. Choose low-glycaemic-index foods.")
    if row['bmi'] >= 25:
        recs.append("Structured weight management: dietary adjustment + progressive physical activity.")
    if row['stress_level'] >= 7:
        recs.append("10-minute daily mindfulness or deep-breathing practice to reduce cortisol levels.")
    if row['smoking']:
        recs.append("Smoking cessation — single highest-impact health intervention available.")
    if not recs:
        recs.append("Continue maintaining balanced diet, regular physical activity, and consistent sleep.")

    return overall, recs[:4]


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_heart_rate(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    hr_values = df['heart_rate'].values
    patient_ids = df.index.values
    patient_names = df['name'].values

    colors = []
    for hr in hr_values:
        if hr > 120: colors.append('#e74c3c')
        elif hr > 100: colors.append('#f39c12')
        elif hr < 60: colors.append('#3498db')
        else: colors.append('#2ecc71')

    ax.bar(range(len(hr_values)), hr_values, color=colors, edgecolor='white', linewidth=1.5)
    ax.axhspan(60, 100, alpha=0.05, color='green')
    ax.axhline(y=100, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=60, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)

    for i, val in enumerate(hr_values):
        ax.text(i, val + 2, f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xticks(range(len(patient_ids)))
    ax.set_xticklabels([f"{pid}\n{n.split()[0]}" for pid, n in zip(patient_ids, patient_names)], fontsize=9)
    ax.set_ylabel('Heart Rate (bpm)')
    ax.set_title('Heart Rate Distribution with Clinical Zones', fontweight='bold')
    ax.set_ylim(0, max(hr_values) + 15)

    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Normal (60-100)'),
        mpatches.Patch(color='#f39c12', label='Warning (100-120)'),
        mpatches.Patch(color='#e74c3c', label='Critical (>120)'),
        mpatches.Patch(color='#3498db', label='Low (<60)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    plt.tight_layout()
    return fig


def plot_risk_heatmap(z_scores, df):
    fig, ax = plt.subplots(figsize=(14, 6))
    heatmap_data = z_scores.copy()
    heatmap_data.columns = ['Heart Rate', 'Systolic BP', 'Diastolic BP', 'Sleep',
                            'Steps', 'Stress', 'BMI', 'Glucose']
    labels = [f"{pid} ({df.loc[pid, 'name'].split()[0]})" for pid in heatmap_data.index]
    sns.heatmap(heatmap_data.values, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
                linewidths=0.5, linecolor='white', xticklabels=heatmap_data.columns,
                yticklabels=labels, cbar_kws={'label': 'Z-Score'}, ax=ax)
    ax.set_title('Risk Score Heatmap (Z-Scores)', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_sleep_vs_stress(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['sleep_hours'], df['stress_level'],
                         s=df['bmi'] * 10, c=df['heart_rate'],
                         cmap='RdYlGn_r', edgecolors='black', linewidth=1, alpha=0.8)
    for pid, row in df.iterrows():
        ax.annotate(pid, (row['sleep_hours'], row['stress_level']),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    plt.colorbar(scatter, ax=ax, label='Heart Rate (bpm)')
    ax.set_xlabel('Sleep Hours per Night')
    ax.set_ylabel('Stress Level (1-10)')
    ax.set_title('Sleep vs. Stress Correlation (size=BMI, color=HR)', fontweight='bold')

    corr = df['sleep_hours'].corr(df['stress_level'])
    ax.text(0.02, 0.98, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    return fig


def plot_anomaly_detection(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    normal = df[df['anomaly_label'] == 1]
    anomalies = df[df['anomaly_label'] == -1]

    ax.scatter(normal['daily_steps'], normal['heart_rate'], c='#2ecc71',
               s=120, edgecolors='black', linewidth=1, label='Normal', alpha=0.8, zorder=3)
    ax.scatter(anomalies['daily_steps'], anomalies['heart_rate'], c='#e74c3c',
               s=200, edgecolors='black', linewidth=2, label='Anomaly', marker='X', zorder=4)

    for pid, row in df.iterrows():
        ax.annotate(f"{pid}", (row['daily_steps'], row['heart_rate']),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax.set_xlabel('Daily Steps')
    ax.set_ylabel('Heart Rate (bpm)')
    ax.set_title('Isolation Forest: Anomaly Detection Results', fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_data = df[NUMERIC_COLS].corr()
    display_cols = ['Heart Rate', 'Sys. BP', 'Dia. BP', 'Sleep',
                    'Steps', 'Stress', 'BMI', 'Glucose']
    mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, xticklabels=display_cols,
                yticklabels=display_cols, ax=ax,
                cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title('Health Metric Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_radar_chart(df, patient_id):
    """Generate a radar chart for a specific patient."""
    row = df.loc[patient_id]
    categories = ['Heart Rate', 'Systolic BP', 'Sleep', 'Steps', 'Stress', 'BMI', 'Glucose']
    raw_values = [row['heart_rate'], row['blood_pressure_systolic'], row['sleep_hours'],
                  row['daily_steps'], row['stress_level'], row['bmi'], row['blood_glucose']]

    # Normalize to 0-1 range for each metric
    mins = [50, 90, 4, 1000, 1, 15, 70]
    maxs = [130, 165, 10, 15000, 10, 40, 200]
    norm_values = [(v - mn) / (mx - mn) for v, mn, mx in zip(raw_values, mins, maxs)]
    norm_values = [max(0, min(1, v)) for v in norm_values]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    norm_values_closed = norm_values + [norm_values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles_closed, norm_values_closed, alpha=0.25, color='#3498db')
    ax.plot(angles_closed, norm_values_closed, 'o-', linewidth=2, color='#2980b9')

    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Digital Twin Radar — {row['name']} ({patient_id})",
                 fontweight='bold', fontsize=13, pad=20)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<div class="main-header">🏥 Mini Digital Health Twin</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Industry 4.0 Digital Twin Concept Applied to Healthcare<br>'
                '<b>Varaganti Vamshi</b> | M.Sc. Industrial AI | Hochschule Albstadt-Sigmaringen</div>',
                unsafe_allow_html=True)

    # Load data
    df = load_patient_data()
    df_analyzed, z_scores = run_anomaly_detection(df)

    # Run risk analysis
    all_risks = {}
    risk_counts = []
    for patient_id, row in df_analyzed.iterrows():
        risks = analyze_patient_risks(row)
        all_risks[patient_id] = risks
        c = sum(1 for r in risks if r['severity'] == 'CRITICAL')
        w = sum(1 for r in risks if r['severity'] == 'WARNING')
        risk_counts.append({'patient_id': patient_id, 'name': row['name'],
                            'critical_risks': c, 'warning_risks': w, 'total_risks': len(risks)})

    risk_df = pd.DataFrame(risk_counts).set_index('patient_id')
    risk_df['risk_score'] = risk_df['critical_risks'] * 2 + risk_df['warning_risks']
    risk_df['risk_category'] = risk_df['risk_score'].apply(categorize_risk)
    risk_df = risk_df.sort_values('risk_score', ascending=False)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to:", [
            "📊 Dashboard Overview",
            "👤 Patient Explorer",
            "⚠️ Risk Analysis",
            "🔍 Anomaly Detection",
            "🤖 AI Health Insights",
            "📈 Visualizations",
            "ℹ️ About"
        ])

    # ── Page: Dashboard Overview ─────────────────────────────────────────────
    if page == "📊 Dashboard Overview":
        st.header("Dashboard Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(df_analyzed))
        with col2:
            high_risk = len(risk_df[risk_df['risk_category'] == 'High Risk'])
            st.metric("High Risk", high_risk, delta=None)
        with col3:
            anomalies = len(df_analyzed[df_analyzed['anomaly_label'] == -1])
            st.metric("ML Anomalies", anomalies)
        with col4:
            healthy = len(risk_df[risk_df['risk_category'] == 'Healthy'])
            st.metric("Healthy", healthy)

        st.subheader("Patient Dataset")
        display_cols = ['name', 'age', 'gender', 'heart_rate', 'blood_pressure_systolic',
                        'blood_pressure_diastolic', 'sleep_hours', 'daily_steps',
                        'stress_level', 'bmi', 'blood_glucose']
        st.dataframe(df_analyzed[display_cols], use_container_width=True)

        st.subheader("Risk Summary Table")
        st.dataframe(risk_df[['name', 'critical_risks', 'warning_risks', 'risk_score', 'risk_category']],
                      use_container_width=True)

    # ── Page: Patient Explorer ───────────────────────────────────────────────
    elif page == "👤 Patient Explorer":
        st.header("Patient Explorer")

        patient_id = st.selectbox("Select Patient:",
                                   df_analyzed.index.tolist(),
                                   format_func=lambda x: f"{x} — {df_analyzed.loc[x, 'name']}")
        row = df_analyzed.loc[patient_id]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{row['name']}")
            st.write(f"**Age:** {row['age']} | **Gender:** {row['gender']}")
            st.write(f"**Medical History:** {', '.join(row['medical_history']) if row['medical_history'] else 'None'}")
            st.write(f"**Medications:** {', '.join(row['medication']) if row['medication'] else 'None'}")
            st.write(f"**Smoking:** {'Yes' if row['smoking'] else 'No'} | **Alcohol:** {row['alcohol_consumption']}")

        with col2:
            st.subheader("Vitals & Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Heart Rate", f"{row['heart_rate']} bpm")
            m2.metric("Blood Pressure", f"{row['blood_pressure_systolic']}/{row['blood_pressure_diastolic']}")
            m3.metric("Blood Glucose", f"{row['blood_glucose']} mg/dL")

            m4, m5, m6 = st.columns(3)
            m4.metric("Sleep", f"{row['sleep_hours']} hrs")
            m5.metric("BMI", f"{row['bmi']}")
            m6.metric("Stress", f"{row['stress_level']}/10")
            st.metric("Daily Steps", f"{row['daily_steps']:,}")

        # Radar chart for this patient
        st.subheader("Digital Twin Radar Profile")
        fig = plot_radar_chart(df_analyzed, patient_id)
        st.pyplot(fig)
        plt.close()

        # Risk flags
        risks = all_risks.get(patient_id, [])
        if risks:
            st.subheader("Risk Flags")
            for r in risks:
                icon = "🔴" if r['severity'] == 'CRITICAL' else "🟡"
                st.write(f"{icon} **{r['metric']}** ({r['value']}): {r['detail']}")
        else:
            st.success("No risk flags detected — all metrics within healthy ranges.")

    # ── Page: Risk Analysis ──────────────────────────────────────────────────
    elif page == "⚠️ Risk Analysis":
        st.header("Rule-Based Health Risk Analysis")
        st.write("Each patient's vitals are checked against clinical thresholds from WHO, AHA, ADA, and CDC guidelines.")

        for patient_id, row in df_analyzed.iterrows():
            risks = all_risks.get(patient_id, [])
            c = sum(1 for r in risks if r['severity'] == 'CRITICAL')
            w = sum(1 for r in risks if r['severity'] == 'WARNING')

            if len(risks) == 0:
                status = "✅ HEALTHY"
            elif c > 0:
                status = "🔴 HIGH RISK"
            else:
                status = "🟡 AT RISK"

            with st.expander(f"{status} — {patient_id} ({row['name']}) | {c} critical, {w} warnings"):
                if risks:
                    for r in risks:
                        sev_color = "risk-critical" if r['severity'] == 'CRITICAL' else "risk-warning"
                        st.markdown(f'<span class="{sev_color}">[{r["severity"]}]</span> '
                                    f'**{r["metric"]}**: {r["value"]} — {r["detail"]}',
                                    unsafe_allow_html=True)
                else:
                    st.write("All metrics within healthy ranges.")

    # ── Page: Anomaly Detection ──────────────────────────────────────────────
    elif page == "🔍 Anomaly Detection":
        st.header("Anomaly Detection")

        tab1, tab2 = st.tabs(["Z-Score Analysis", "Isolation Forest"])

        with tab1:
            st.subheader("Z-Score Analysis")
            st.write("Z-scores show how far each patient value is from the group average. |Z| > 2 = anomaly.")
            st.dataframe(z_scores.round(2), use_container_width=True)

            st.subheader("Anomalous Values (|Z| > 2)")
            found = False
            for pid in z_scores.index:
                for col in NUMERIC_COLS:
                    z_val = z_scores.loc[pid, col]
                    if abs(z_val) > 2:
                        found = True
                        direction = "ABOVE" if z_val > 0 else "BELOW"
                        st.write(f"**{pid}** ({df_analyzed.loc[pid, 'name']}): "
                                 f"{col} = {df_analyzed.loc[pid, col]} (Z={z_val:.2f}, {direction} average)")
            if not found:
                st.info("No extreme outliers found with threshold |Z| > 2.")

        with tab2:
            st.subheader("Isolation Forest Results")
            st.write("Contamination = 0.2 (expects ~20% anomalous). Uses 100 isolation trees.")
            for pid, row in df_analyzed.iterrows():
                label = "🔴 ANOMALY" if row['anomaly_label'] == -1 else "🟢 Normal"
                st.write(f"{label} — **{pid}** ({row['name']}) | Score: {row['anomaly_score']:+.4f}")

    # ── Page: AI Health Insights ─────────────────────────────────────────────
    elif page == "🤖 AI Health Insights":
        st.header("AI Health Insight Generator")
        st.write("Template-based health insights generated using clinical logic. "
                 "In the notebook, real Claude API calls are also demonstrated.")

        patient_id = st.selectbox("Select Patient for AI Insight:",
                                   df_analyzed.index.tolist(),
                                   format_func=lambda x: f"{x} — {df_analyzed.loc[x, 'name']}")
        row = df_analyzed.loc[patient_id]
        risks = all_risks.get(patient_id, [])

        if st.button("Generate Health Insight", type="primary"):
            overall, recs = generate_health_insight(row, risks)

            st.subheader("Overall Assessment")
            st.info(overall)

            st.subheader("Recommendations")
            for i, rec in enumerate(recs, 1):
                st.write(f"**{i}.** {rec}")

    # ── Page: Visualizations ─────────────────────────────────────────────────
    elif page == "📈 Visualizations":
        st.header("Health Data Visualizations")

        viz_choice = st.selectbox("Select Visualization:", [
            "Heart Rate Distribution",
            "Risk Heatmap (Z-Scores)",
            "Sleep vs. Stress Correlation",
            "Anomaly Detection Plot",
            "Correlation Matrix",
            "Patient Radar Chart"
        ])

        if viz_choice == "Heart Rate Distribution":
            fig = plot_heart_rate(df_analyzed)
            st.pyplot(fig)
            plt.close()
            st.caption("Bar chart showing each patient's resting heart rate with clinical zone overlays.")

        elif viz_choice == "Risk Heatmap (Z-Scores)":
            fig = plot_risk_heatmap(z_scores, df_analyzed)
            st.pyplot(fig)
            plt.close()
            st.caption("Heatmap showing Z-score deviations from the group mean for all health metrics.")

        elif viz_choice == "Sleep vs. Stress Correlation":
            fig = plot_sleep_vs_stress(df_analyzed)
            st.pyplot(fig)
            plt.close()
            st.caption("Scatter plot correlating sleep hours with stress level. Bubble size = BMI, color = heart rate.")

        elif viz_choice == "Anomaly Detection Plot":
            fig = plot_anomaly_detection(df_analyzed)
            st.pyplot(fig)
            plt.close()
            st.caption("Isolation Forest results projected onto steps vs. heart rate space.")

        elif viz_choice == "Correlation Matrix":
            fig = plot_correlation_matrix(df_analyzed)
            st.pyplot(fig)
            plt.close()
            st.caption("Pearson correlation coefficients between all numerical health metrics.")

        elif viz_choice == "Patient Radar Chart":
            pid = st.selectbox("Select Patient for Radar:",
                                df_analyzed.index.tolist(),
                                format_func=lambda x: f"{x} — {df_analyzed.loc[x, 'name']}")
            fig = plot_radar_chart(df_analyzed, pid)
            st.pyplot(fig)
            plt.close()

    # ── Page: About ──────────────────────────────────────────────────────────
    elif page == "ℹ️ About":
        st.header("About This Project")
        st.write("""
        **Mini Digital Health Twin Prototype** is a proof-of-concept application that demonstrates
        how Industry 4.0 Digital Twin concepts can be applied to healthcare.

        **What it does:**
        - Loads patient health data from structured JSON files (FHIR-aligned)
        - Performs rule-based clinical risk analysis against WHO, AHA, ADA, CDC guidelines
        - Runs statistical (Z-score) and ML-based (Isolation Forest) anomaly detection
        - Generates AI-powered health insights using Anthropic Claude API (with template fallback)
        - Provides 6 interactive visualizations for clinical decision support

        **Technology Stack:**
        Python, Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn, Streamlit, Anthropic API

        **Author:** Varaganti Vamshi
        **Program:** M.Sc. Industrial Artificial Intelligence
        **University:** Hochschule Albstadt-Sigmaringen
        **Assessment:** HOPn Technical Assessment — Prof. Dr. Ahmed Ebada
        """)


if __name__ == "__main__":
    main()
