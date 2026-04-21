import streamlit as st
import numpy as np
import pickle

# Load pipeline
pipeline = pickle.load(open("../models/pipeline.pkl", "rb"))

# Page config
st.set_page_config(page_title="Startup Intelligence System", layout="centered")

# Title
st.markdown("# 🚀 Startup Success Intelligence System")
st.caption("ML-powered system to predict startup success using funding insights")

# Sidebar
st.sidebar.header("📊 About")
st.sidebar.write("""
• Model: XGBoost (Pipeline)  
• Features: Funding-based engineered features  
• Includes: Custom Health Score  
""")

# Input Section
st.markdown("## 📝 Enter Startup Details")

col1, col2 = st.columns(2)

with col1:
    total_funding = st.number_input("💰 Total Funding ($)", 0.0, 1000000.0)

with col2:
    funding_rounds = st.number_input("🔁 Funding Rounds", 0, 2)

startup_age = st.slider("📅 Startup Age (years)", 0, 30, 3)

st.divider()

# Prediction Button
if st.button("🔍 Predict Success"):

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

    # 🔥 Adjusted Probability (important)
    adjusted_prob = 0.7 * base_prob + 0.3 * (health_score / 100)

    # Results
    st.markdown("## 📊 Prediction Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("🎯 Success Probability", f"{adjusted_prob*100:.1f}%")

    with col2:
        st.metric("🧠 Health Score", f"{health_score:.1f}/100")

    # Progress bar
    st.progress(min(max(adjusted_prob, 0.0), 1.0))

    # Decision
    if adjusted_prob >= 0.7:
        st.success("🔥 Highly Likely to Succeed")
    elif adjusted_prob >= 0.5:
        st.info("🙂 Moderate Chance of Success")
    else:
        st.error("❌ Low Success Probability")

    # Interpretation
    if adjusted_prob >= 0.7:
        st.write("➡ Strong funding pattern and consistent growth.")
    elif adjusted_prob >= 0.5:
        st.write("➡ Mixed signals — moderate potential.")
    else:
        st.write("➡ Weak financial indicators.")

    # Insights
    st.markdown("### 🔍 Key Insights")

    st.write(f"• Funding Efficiency: ${funding_efficiency:,.0f}")
    st.write(f"• Funding per Year: ${funding_per_year:,.0f}")
    st.write(f"• Log Funding: {log_funding:.2f}")

    # Warning
    if total_funding == 0:
        st.warning("⚠️ Zero funding may lead to unreliable predictions.")

# Footer
st.markdown("---")
st.caption("Built by Nischal Mahajan | Machine Learning Project 🚀")