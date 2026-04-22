import streamlit as st
import numpy as np
import os
import joblib
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "pipeline.pkl")

if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

pipeline = joblib.load(model_path)

# Page config
st.set_page_config(page_title="Startup Intelligence System", layout="centered", page_icon="🚀")

# Title
st.markdown("# 🚀 Startup Success Intelligence System")
st.caption("ML-powered system to predict startup success using funding insights")

# Sidebar
st.sidebar.header("📊 About This Project")
st.sidebar.write("""
• **Model:** XGBoost (Pipeline)  
• **Features:** Funding-based engineered features  
• **Includes:** Custom Health Score  
""")

st.sidebar.divider()
st.sidebar.header("🧠 How It Works")
st.sidebar.write("""
1. Enter your startup's funding details  
2. Model computes engineered features  
3. XGBoost predicts success probability  
4. Custom Health Score is blended in  
5. Final verdict is shown instantly  
""")

st.sidebar.divider()
st.sidebar.header("📐 Features Used")
st.sidebar.write("""
• **Funding Efficiency** = Total Funding / (Rounds + 1)  
• **Funding per Year** = Total Funding / (Age + 1)  
• **Log Funding** = log(1 + Total Funding)  
""")

st.sidebar.divider()
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-View%20Source-black?logo=github)](https://github.com/Nischal-Mahajan/Startup-Health-Intelligence-System)")

# Input Section
st.markdown("## 📝 Enter Startup Details")

col1, col2 = st.columns(2)

with col1:
    total_funding = st.number_input("💰 Total Funding ($)", 0.0, 1000000.0, step=1000.0)

with col2:
    funding_rounds = st.number_input("🔄 Funding Rounds", min_value=0)

startup_age = st.slider("📅 Startup Age (years)", 0, 30, 3)

st.divider()

# Prediction Button
if st.button("🔍 Predict Success", use_container_width=True):

    # Feature Engineering
    funding_efficiency = total_funding / (funding_rounds + 1)
    funding_per_year = total_funding / (startup_age + 1)
    log_funding = np.log1p(total_funding)

    # Health Score (normalized)
    raw_score = (
        0.4 * np.log1p(funding_efficiency) +
        0.3 * np.log1p(funding_per_year) +
        0.3 * log_funding
    )
    health_score = min(raw_score / 10, 1.0) * 100

    # Model Prediction
    base_prob = float(pipeline.predict_proba(
        np.array([[funding_efficiency, funding_per_year, log_funding]])
    )[0][1])

    # Adjusted Probability
    adjusted_prob = 0.7 * base_prob + 0.3 * (health_score / 100)

    # Results
    st.markdown("## 📊 Prediction Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Success Probability", f"{adjusted_prob*100:.1f}%")
    with col2:
        st.metric("🧠 Health Score", f"{health_score:.1f}/100")

    # --- Gauge Chart ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(adjusted_prob * 100, 1),
        title={"text": "Success Probability", "font": {"size": 20}},
        # delta={"reference": 50, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "royalblue"},
            "steps": [
                {"range": [0, 50], "color": "#ffcccc"},
                {"range": [50, 70], "color": "#fff3cc"},
                {"range": [70, 100], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": adjusted_prob * 100,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True)

    # Decision
    if adjusted_prob >= 0.7:
        st.success("🔥 Highly Likely to Succeed")
        st.write("➡ Strong funding pattern and consistent growth.")
    elif adjusted_prob >= 0.5:
        st.info("🙂 Moderate Chance of Success")
        st.write("➡ Mixed signals — moderate potential.")
    else:
        st.error("❌ Low Success Probability")
        st.write("➡ Weak financial indicators.")

    # Insights
    st.markdown("### 🔍 Key Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("💡 Funding Efficiency", f"${funding_efficiency:,.0f}")
    c2.metric("📆 Funding per Year", f"${funding_per_year:,.0f}")
    c3.metric("📈 Log Funding", f"{log_funding:.2f}")

    # Warning
    if total_funding == 0:
        st.warning("⚠️ Zero funding may lead to unreliable predictions.")

# Footer
st.markdown("---")
st.caption("Built by Nischal Mahajan | Machine Learning Project 🚀 | [GitHub](https://github.com/Nischal-Mahajan/Startup-Health-Intelligence-System)")