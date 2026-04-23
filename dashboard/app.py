import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
# import EVERYTHING used in pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import traceback
import streamlit as st

model_path = "models/pipeline.pkl"  # your correct path

try:
    pipeline = joblib.load(model_path)
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error("REAL ERROR BELOW 👇")
    st.code(traceback.format_exc())


# if you created custom class:
# from ..utils.feature_engineering import FeatureEngineer  # adjust path

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FundIQ · Startup Intelligence",
    layout="wide",
    page_icon="◈",
    initial_sidebar_state="collapsed",
)

# ─── Load Model & Scaler ───────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path  = os.path.join(BASE_DIR, "models", "pipeline.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

if not os.path.exists(model_path):
    st.error(f"Model not found: {model_path}")
    st.stop()

pipeline = joblib.load(model_path)
try:
    pipeline = joblib.load(model_path)
except Exception as e:
    st.error(str(e))
    st.text(traceback.format_exc())

# ─── Feature Engineering (matches training pipeline exactly) ───────────────────
DEPLOY_FEATURES = [
    "funding_efficiency",
    "funding_per_year",
    "log_funding",
    "milestones",              # FIX
    "relationships",
    "investment_rounds",
    "startup_age",
    "funding_rounds_count",    # FIX
    "burn_rate",
    "runway",
    "growth_intensity",
    "health_score",
]
SCALE_FEATURES = [
    "funding_efficiency",
    "funding_per_year",
    "log_funding",
    "burn_rate",
    "runway",
    "growth_intensity",
]



def create_features(funding, rounds, inv_rounds, age, milestones, relationships):
    """Compute all 12 deploy features — mirrors training pipeline."""
    funding_efficiency = funding / (rounds + 1)
    funding_per_year   = funding / (age + 1)
    log_funding        = np.log1p(funding)

    # Derived signals
    burn_rate        = funding / max(age * 12, 1)          # monthly burn
    runway           = funding / max(burn_rate, 1)          # months of runway
    growth_intensity = (milestones + relationships) / max(age, 1)

    # Composite health score (0–100)
    # CHANGE 2: Updated health score scaling formula
    raw = (
        0.4 * np.log1p(funding_efficiency)
        + 0.3 * np.log1p(funding_per_year)
        + 0.3 * log_funding
    )
    health_score = (raw / (raw + 5)) * 100  # Updated from: min(raw / 10, 1.0) * 100

    feature_dict = {
    "funding_efficiency": funding_efficiency,
    "funding_per_year":   funding_per_year,
    "log_funding":        log_funding,
    "milestones":         float(milestones),        # ✅ FIXED
    "relationships":      float(relationships),
    "investment_rounds":  float(inv_rounds),
    "startup_age":        float(age),
    "funding_rounds_count": float(rounds),          # ✅ FIXED
    "burn_rate":          burn_rate,
    "runway":             runway,
    "growth_intensity":   growth_intensity,
    "health_score":       health_score,
    }
    return feature_dict, health_score, log_funding


# CHANGE 1: Full rewrite of predict() — calibrated-safe outputs, improved radar, added risk_score
def predict(funding, rounds, inv_rounds, age, milestones, relationships):
    """Run inference using same feature order as training."""
    feat, health_score, log_f = create_features(
        funding, rounds, inv_rounds, age, milestones, relationships
    )

    X_raw = np.array([[feat[k] for k in DEPLOY_FEATURES]])

    # Apply scaler only to scale-required features if scaler is present
    X_input = pd.DataFrame([feat])[DEPLOY_FEATURES]

    base_prob = float(pipeline.predict_proba(X_input)[0][1])

    # Validate — no NaN / inf
    X_input = pd.DataFrame([feat])[DEPLOY_FEATURES]

    if not np.isfinite(X_input.values).all():
        X_input = X_input.replace([np.inf, -np.inf], 0).fillna(0)

    base_prob = float(pipeline.predict_proba(X_input)[0][1])
    # CHANGE 1: Clamp probability to realistic range
    prob = min(max(base_prob, 0.05), 0.85)

    # CHANGE 1: Improved radar scaling (linear, not log)
    radar_vals = [
        min(funding / 1_000_000, 1) * 100,
        min(rounds / 10, 1) * 100,
        min(age / 10, 1) * 100,
        min(milestones / 10, 1) * 100,
        min(relationships / 20, 1) * 100,
    ]

    # CHANGE 1: Compute risk score
    risk_score = (feat["burn_rate"] / (feat["runway"] + 1)) * 100

    # CHANGE 3: Return now includes risk_score
    return prob, health_score, base_prob, feat["funding_efficiency"], feat["funding_per_year"], log_f, radar_vals, risk_score


# ─── Top Feature Drivers ───────────────────────────────────────────────────────
def get_top_drivers(feat: dict, prob: float):
    """Return top 3 features driving this prediction (rule-based heuristic)."""
    scores = {
        "Funding Efficiency":  min(np.log1p(feat["funding_efficiency"]) / 15, 1),
        "Funding per Year":    min(np.log1p(feat["funding_per_year"]) / 15, 1),
        "Milestones Achieved": min(feat["milestones"] / 10, 1),
        "Investor Network":    min(feat["relationships"] / 20, 1),
        "Runway":              min(feat["runway"] / 36, 1),
        "Growth Intensity":    min(feat["growth_intensity"] / 5, 1),
    }
    sorted_drivers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_drivers[:3]


# ─── Recommendations ──────────────────────────────────────────────────────────
# CHANGE 5: Improved recommendations using engineered features
def get_recommendations(funding, rounds, milestones, rels, age, feat=None):
    recs = []
    seen = set()

    def add(priority, title, desc):
        if title not in seen:
            seen.add(title)
            recs.append((priority, title, desc))

    # CHANGE 5: Feature-based recommendations (engineered signals)
    if feat is not None:
        if feat["funding_efficiency"] < 50000:
            add("HIGH", "Low Funding Efficiency",
                "Your funding efficiency is below $50K per round. Focus on fewer, higher-conviction rounds to maximise capital per raise and signal quality to institutional investors.")
        if feat["growth_intensity"] < 1:
            add("HIGH", "Low Growth Intensity",
                "Growth intensity (milestones + relationships per year) is critically low. Accelerate milestone execution and expand your investor network to improve momentum signals.")

    # Funding
    if funding == 0:
        add("HIGH", "No Funding Detected",
            "Securing seed funding is the critical first step. Target angel investors or accelerators to establish credibility and runway.")
    elif funding < 100_000:
        add("MED", "Funding Below $100K",
            "Target angel investors or accelerators to reach meaningful seed-stage funding before burn erodes optionality.")
    elif funding < 1_000_000:
        add("MED", "Pre-Series A Range",
            "Strong early momentum — prepare your Series A pitch deck and identify 10–15 institutional investors to approach.")
    else:
        add("OK", "Funding Level Healthy",
            "Maintain strong investor relations and deploy capital efficiently to maximize runway and signal growth.")

    # Rounds
    if rounds == 0:
        add("HIGH", "No Funding Rounds Completed",
            "Initiating your first round signals market credibility. Focus on a clean SAFE or convertible note structure.")
    elif rounds < 3:
        add("MED", "Limited Funding Rounds",
            "Additional rounds significantly increase investor trust and valuation benchmarks. Plan your next round 12 months ahead.")
    else:
        add("OK", "Multiple Rounds Completed",
            "Strong investor conviction demonstrated. Focus capital deployment on growth metrics that underpin your next raise.")

    # Milestones
    if milestones == 0:
        add("HIGH", "No Milestones Achieved",
            "Product and market milestones are the primary proof points in investor diligence. Prioritize shipping a v1 and acquiring initial customers.")
    elif milestones < 3:
        add("MED", "Few Milestones Recorded",
            "Accelerate product launches and market traction milestones. Milestone velocity is a leading indicator of execution quality.")
    else:
        add("OK", "Strong Milestone Record",
            "Excellent execution track record. Use milestone history as the centrepiece of your investor narrative.")

    # Network / relationships
    if rels < 2:
        add("HIGH", "Weak Investor Network",
            "Expand your investor and partner network urgently. Warm introductions convert 3× better than cold outreach.")
    elif rels < 5:
        add("MED", "Moderate Network",
            "Invest consistently in relationship building — attend relevant events, join accelerator cohorts, and leverage LinkedIn strategically.")
    else:
        add("OK", "Strong Network",
            "Leverage your network for the next growth phase: co-investors, strategic partnerships, and senior talent acquisition.")

    # Age / stage
    if age < 2:
        add("MED", "Early-Stage Startup",
            "Focus ruthlessly on product-market fit before scaling headcount or spend. Validate before you amplify.")
    elif age >= 7:
        add("OK", "Proven Longevity",
            "7+ years demonstrates market adaptability — a powerful signal. Emphasise this resilience in investor conversations.")
    else:
        add("OK", "Established Startup",
            "Solid trajectory — double down on what's working and sunset what's not. Unit economics clarity is your next unlock.")

    return recs[:8]   # cap at 8 actionable insights


# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg-main:       #0B0F14;
    --bg-card:       #111827;
    --bg-elevated:   #1F2937;
    --border:        #1F2937;
    --text-primary:  #F9FAFB;
    --text-secondary:#9CA3AF;
    --accent:        #D4AF37;
    --success:       #22C55E;
    --danger:        #EF4444;
    --warning:       #F59E0B;
    --neutral:       #6B7280;
    --sky:           #38BDF8;
    --r-xs: 3px; --r-sm: 6px; --r-md: 10px; --r-lg: 14px; --r-xl: 20px;
    --font-sans: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .main > div,
section.main, .block-container,
[data-testid="stMainBlockContainer"] {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

body, p, span, div, h1, h2, h3, h4, h5, h6, label {
    color: var(--text-primary) !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarNav"] {
    display: none !important;
    visibility: hidden !important;
}

.block-container {
    padding: 0 2.5rem 6rem 2.5rem !important;
    max-width: 100% !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}

/* ══════════════════════════════════════
   PAGE HEADER
══════════════════════════════════════ */
.page-header {
    padding: 1.2rem 0 1.6rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
}
.page-eyebrow {
    font-family: var(--font-mono) !important;
    font-size: 8px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-bottom: 8px;
    opacity: 0.85;
}
.page-title {
    font-family: var(--font-sans) !important;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.045em;
    color: var(--text-primary) !important;
    line-height: 1.05;
}
.page-title em { font-style: normal; color: var(--accent) !important; }
.page-desc {
    font-size: 0.72rem;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    margin-top: 8px;
    letter-spacing: 0.3px;
}
.header-tags { display: flex; gap: 6px; flex-shrink: 0; flex-wrap: wrap; justify-content: flex-end; }
.tag {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 11px;
    border-radius: var(--r-sm);
    font-family: var(--font-mono) !important;
    font-size: 8px; letter-spacing: 1px; text-transform: uppercase;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-secondary) !important;
}
.tag-dot { width: 4px; height: 4px; border-radius: 50%; flex-shrink: 0; }
.tag-gold  { border-color: rgba(212,175,55,0.22); background: rgba(212,175,55,0.06); color: var(--accent) !important; }
.tag-green { border-color: rgba(34,197,94,0.18);  background: rgba(34,197,94,0.10);  color: var(--success) !important; }
.tag-sand  { border-color: rgba(245,158,11,0.22); background: rgba(245,158,11,0.08); color: var(--warning) !important; }
.tag-gray  { border-color: rgba(107,114,128,0.20);background: rgba(107,114,128,0.10);color: var(--neutral) !important; }

/* ══════════════════════════════════════
   SECTION LABEL
══════════════════════════════════════ */
.section-label {
    font-family: var(--font-mono) !important;
    font-size: 7.5px; letter-spacing: 3px; text-transform: uppercase;
    color: var(--text-secondary) !important;
    padding: 1.6rem 0 0.8rem 0;
    display: flex; align-items: center; gap: 12px;
}
.section-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ══════════════════════════════════════
   STAT CARDS
══════════════════════════════════════ */
.card-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.4rem;
}
@media (max-width: 900px) { .card-grid-4 { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 540px) { .card-grid-4 { grid-template-columns: 1fr; } }

.card-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 1.4rem;
}

.stat-card {
    background: var(--bg-card);
    padding: 1.4rem;
    border-radius: var(--r-lg);
    border: 1px solid var(--border);
    transition: all 0.15s;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    opacity: 0;
    transition: opacity 0.2s;
}
.stat-card:hover { background: var(--bg-elevated); transform: translateY(-2px); box-shadow: 0 4px 20px rgba(0,0,0,0.4); border-color: rgba(212,175,55,0.3); }
.stat-card:hover::before { opacity: 1; }
.stat-card.accent-gold::before  { background: var(--accent); }
.stat-card.accent-green::before { background: var(--success); }
.stat-card.accent-sand::before  { background: var(--warning); }
.stat-card.accent-rose::before  { background: var(--danger); }
.stat-card.accent-sky::before   { background: var(--sky); }
.stat-label {
    font-family: var(--font-mono) !important;
    font-size: 7.5px; letter-spacing: 2px; text-transform: uppercase;
    color: var(--text-secondary) !important; margin-bottom: 0.9rem;
}
.stat-value {
    font-family: var(--font-sans) !important;
    font-size: 2rem; font-weight: 700; color: var(--text-primary) !important;
    line-height: 1; letter-spacing: -0.04em; margin-bottom: 0.35rem;
}
.stat-value.c-gold  { color: var(--accent) !important; }
.stat-value.c-green { color: var(--success) !important; }
.stat-value.c-sand  { color: var(--warning) !important; }
.stat-value.c-rose  { color: var(--danger) !important; }
.stat-value.c-sky   { color: var(--sky) !important; }
.stat-value.c-gray  { color: var(--neutral) !important; }
.stat-sub { font-family: var(--font-mono) !important; font-size: 8.5px; color: var(--text-secondary) !important; }

/* ══════════════════════════════════════
   DECISION TAG BANNER
══════════════════════════════════════ */
.decision-banner {
    border-radius: var(--r-lg);
    padding: 1.1rem 1.5rem;
    display: flex; align-items: center; gap: 1rem;
    margin: 0 0 1rem 0;
    border: 1px solid;
    font-family: var(--font-mono) !important;
}
.db-high-potential {
    background: rgba(34,197,94,0.07);
    border-color: rgba(34,197,94,0.22);
}
.db-moderate {
    background: rgba(245,158,11,0.07);
    border-color: rgba(245,158,11,0.22);
}
.db-high-risk {
    background: rgba(239,68,68,0.07);
    border-color: rgba(239,68,68,0.22);
}
.db-tag {
    font-size: 8px; letter-spacing: 2px; text-transform: uppercase;
    font-weight: 700; padding: 4px 10px;
    border-radius: var(--r-xs); flex-shrink: 0;
}
.db-tag-hp  { background: rgba(34,197,94,0.12);  color: #22C55E; border: 1px solid rgba(34,197,94,0.25); }
.db-tag-mod { background: rgba(245,158,11,0.12); color: #F59E0B; border: 1px solid rgba(245,158,11,0.25); }
.db-tag-hr  { background: rgba(239,68,68,0.12);  color: #EF4444; border: 1px solid rgba(239,68,68,0.25); }
.db-title   { font-size: 0.95rem; font-weight: 600; font-family: var(--font-sans) !important; }
.db-sub     { font-size: 9.5px;   color: var(--text-secondary) !important; line-height: 1.65; margin-top: 2px; }

/* ══════════════════════════════════════
   DRIVER CHIPS
══════════════════════════════════════ */
.driver-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.65rem 1rem;
    border-radius: var(--r-md);
    background: var(--bg-card);
    border: 1px solid var(--border);
    margin: 0.3rem 0;
    gap: 1rem;
}
.driver-name { font-family: var(--font-mono) !important; font-size: 9.5px; color: var(--text-primary) !important; }
.driver-bar-wrap { flex: 1; height: 4px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
.driver-bar      { height: 100%; border-radius: 99px; }
.driver-pct { font-family: var(--font-mono) !important; font-size: 9px; color: var(--text-secondary) !important; min-width: 34px; text-align: right; }

/* ══════════════════════════════════════
   VERDICT BLOCK
══════════════════════════════════════ */
.verdict {
    border-radius: var(--r-lg);
    padding: 1.3rem 1.7rem;
    display: flex; align-items: flex-start; gap: 1.1rem;
    margin: 0.8rem 0 0.4rem 0;
}
.verdict-success  { background: rgba(34,197,94,0.08);  border: 1px solid rgba(34,197,94,0.16); }
.verdict-moderate { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.20); }
.verdict-fail     { background: rgba(239,68,68,0.08);  border: 1px solid rgba(239,68,68,0.20); }
.verdict-icon {
    font-size: 0.95rem; width: 34px; height: 34px;
    display: flex; align-items: center; justify-content: center;
    border-radius: var(--r-sm);
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    flex-shrink: 0; font-family: var(--font-mono) !important;
    color: var(--text-primary) !important;
}
.verdict-title { font-family: var(--font-sans) !important; font-size: 0.95rem; font-weight: 600; margin-bottom: 0.2rem; }
.verdict-desc  { font-family: var(--font-mono) !important; font-size: 9.5px; color: var(--text-secondary) !important; line-height: 1.7; }
.vc-green { color: var(--success) !important; }
.vc-sand  { color: var(--warning) !important; }
.vc-rose  { color: var(--danger) !important; }

/* ══════════════════════════════════════
   WHY THIS RESULT
══════════════════════════════════════ */
.why-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 1.1rem 1.4rem;
    margin: 0.6rem 0;
}
.why-title {
    font-family: var(--font-mono) !important;
    font-size: 7.5px; letter-spacing: 2.5px; text-transform: uppercase;
    color: var(--text-secondary) !important; margin-bottom: 0.8rem;
}
.why-row {
    display: flex; align-items: flex-start; gap: 0.65rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(31,41,55,0.7);
}
.why-row:last-child { border-bottom: none; }
.why-icon { font-size: 11px; flex-shrink: 0; margin-top: 1px; }
.why-text { font-family: var(--font-mono) !important; font-size: 9.5px; color: var(--text-secondary) !important; line-height: 1.6; }
.why-text strong { color: var(--text-primary) !important; font-weight: 500; }

/* ══════════════════════════════════════
   RECOMMENDATIONS
══════════════════════════════════════ */
.rec-row {
    display: flex; align-items: flex-start; gap: 0.9rem;
    padding: 0.85rem 1.1rem;
    border-radius: var(--r-md);
    border: 1px solid var(--border);
    background: var(--bg-card);
    margin: 0.35rem 0;
    transition: border-color 0.15s, background 0.15s;
}
.rec-row:hover { border-color: rgba(212,175,55,0.3); background: var(--bg-elevated); }
.rec-badge {
    font-family: var(--font-mono) !important;
    font-size: 7.5px; letter-spacing: 1.2px; text-transform: uppercase;
    font-weight: 600; padding: 3px 8px; border-radius: var(--r-xs);
    flex-shrink: 0; margin-top: 2px;
}
.rb-high { background: rgba(239,68,68,0.08);  color: var(--danger) !important;  border: 1px solid rgba(239,68,68,0.20); }
.rb-med  { background: rgba(245,158,11,0.08); color: var(--warning) !important; border: 1px solid rgba(245,158,11,0.22); }
.rb-ok   { background: rgba(34,197,94,0.08);  color: var(--success) !important; border: 1px solid rgba(34,197,94,0.20); }
.rec-title { font-family: var(--font-sans) !important; font-size: 1rem; font-weight: 500; color: var(--text-primary) !important; margin-bottom: 0.25rem; }
.rec-desc  { font-family: var(--font-mono) !important; font-size: 11.5px; color: var(--text-secondary) !important; line-height: 1.65; }

/* ══════════════════════════════════════
   WINNER STRIP
══════════════════════════════════════ */
.winner-strip {
    background: var(--bg-card);
    border: 1px solid rgba(34,197,94,0.18);
    border-radius: 12px;
    padding: 1.5rem 1.9rem;
    display: flex; align-items: center; justify-content: space-between;
    margin: 0.7rem 0 1.2rem 0;
}
.winner-lbl  { font-family: var(--font-mono) !important; font-size: 7.5px; letter-spacing: 2.5px; text-transform: uppercase; color: var(--text-secondary) !important; margin-bottom: 4px; }
.winner-name { font-family: var(--font-sans) !important; font-size: 1.6rem; font-weight: 700; color: var(--success) !important; letter-spacing: -0.04em; }
.winner-prob { font-family: var(--font-mono) !important; font-size: 9px; color: var(--text-secondary) !important; margin-top: 4px; }
.winner-badge { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.22); padding: 5px 12px; border-radius: var(--r-sm); font-family: var(--font-mono) !important; font-size: 9px; color: var(--success) !important; font-weight: 500; }

/* ══════════════════════════════════════
   INPUTS
══════════════════════════════════════ */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
    transition: border-color 0.15s !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: rgba(212,175,55,0.35) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(212,175,55,0.08) !important;
}
[data-testid="stNumberInput"] button {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-secondary) !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label,
label[data-testid="stWidgetLabel"] {
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 8.5px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ══════════════════════════════════════
   BUTTON
══════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #D4AF37 0%, #B8962E 100%) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-mono) !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    letter-spacing: 2.5px !important;
    padding: 0.85rem 2rem !important;
    text-transform: uppercase !important;
    transition: all 0.18s !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(212,175,55,0.2) !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    box-shadow: 0 4px 16px rgba(212,175,55,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ══════════════════════════════════════
   TABS
══════════════════════════════════════ */
[data-testid="stTabs"] { background: transparent !important; }
[data-testid="stTabBar"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 8.5px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    padding: 0.85rem 1.4rem !important;
    border-bottom: 1.5px solid transparent !important;
    transition: all 0.18s !important;
    margin-bottom: -1px !important;
}
button[data-baseweb="tab"]:hover { color: var(--text-primary) !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ══════════════════════════════════════
   METRICS
══════════════════════════════════════ */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
    transition: border-color 0.15s !important;
}
[data-testid="metric-container"]:hover { border-color: rgba(212,175,55,0.3) !important; }
[data-testid="metric-container"] label,
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
    letter-spacing: -0.04em !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
}

/* ══════════════════════════════════════
   MISC
══════════════════════════════════════ */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
[data-testid="stAlert"] {
    background: rgba(245,158,11,0.08) !important;
    border: 1px solid rgba(245,158,11,0.22) !important;
    border-radius: var(--r-md) !important;
    color: var(--warning) !important;
    font-family: var(--font-mono) !important;
    font-size: 9.5px !important;
}
[data-testid="stSpinner"] { color: var(--accent) !important; }
.page-footer {
    margin-top: 4rem; padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    font-family: var(--font-mono) !important;
    font-size: 8.5px; color: var(--text-secondary) !important;
    display: flex; align-items: center; justify-content: space-between;
}
.page-footer a { color: var(--text-secondary) !important; text-decoration: none; }
.page-footer a:hover { color: var(--accent) !important; }
.js-plotly-plot, .plotly, .plot-container { overflow: visible !important; }
[data-testid="stPlotlyChart"] { overflow: visible !important; }

/* ══ Success toast ══ */
.toast {
    background: rgba(34,197,94,0.10);
    border: 1px solid rgba(34,197,94,0.22);
    border-radius: var(--r-md);
    padding: 0.75rem 1.1rem;
    font-family: var(--font-mono) !important;
    font-size: 9.5px;
    color: var(--success) !important;
    display: flex; align-items: center; gap: 0.6rem;
    margin: 0.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Chart helpers ─────────────────────────────────────────────────────────────
C_GOLD    = "#D4AF37"
C_SUCCESS = "#22C55E"
C_DANGER  = "#EF4444"
C_WARNING = "#F59E0B"
C_NEUTRAL = "#6B7280"
C_SKY     = "#38BDF8"
_BG   = "rgba(0,0,0,0)"
_TICK = "#9CA3AF"
_GRID = "rgba(255,255,255,0.05)"

def hex_to_rgba(h, a=0.15):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{a})"

def make_gauge(value, title=""):
    color = C_SUCCESS if value >= 70 else (C_WARNING if value >= 50 else C_DANGER)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 44, "color": "#F9FAFB", "family": "DM Sans"}, "suffix": "%"},
        title={"text": title, "font": {"size": 9, "color": "#9CA3AF", "family": "DM Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6B7280",
                     "tickfont": {"color": "#9CA3AF", "size": 9, "family": "DM Mono"}, "nticks": 6},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "#0B0F14", "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "rgba(239,68,68,0.15)"},
                {"range": [50, 70], "color": "rgba(245,158,11,0.18)"},
                {"range": [70,100], "color": "rgba(34,197,94,0.20)"},
            ],
            "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.80, "value": value},
        },
    ))
    fig.update_layout(
        height=340, margin=dict(t=60, b=20, l=40, r=40),
        paper_bgcolor="#0B0F14", plot_bgcolor="#0B0F14",
        font=dict(family="DM Mono", color="#9CA3AF"),
    )
    return fig

def make_radar(values, name, color_hex):
    cats = ["Funding", "Rounds", "Age", "Milestones", "Network"]
    v = values + [values[0]]; c = cats + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v, theta=c, fill="toself",
        fillcolor=hex_to_rgba(color_hex, 0.45),
        line=dict(color=color_hex, width=3.0), name=name,
        marker=dict(size=6, color=color_hex, line=dict(color="#0B0F14", width=1.5)),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0B0F14",
            radialaxis=dict(visible=True, range=[0, 100],
                tickfont=dict(color="#9CA3AF", size=9, family="DM Mono"),
                gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)",
                tickvals=[0,25,50,75,100], ticktext=["0","25","50","75","100"]),
            angularaxis=dict(tickfont=dict(color="#FFFFFF", size=12, family="DM Sans"),
                gridcolor="rgba(255,255,255,0.10)", linecolor="rgba(255,255,255,0.10)"),
        ),
        showlegend=False, height=340, margin=dict(t=20, b=20, l=55, r=55),
        paper_bgcolor="#0B0F14", plot_bgcolor="#0B0F14",
        font=dict(family="DM Mono", color="#FFFFFF")
    )
    return fig

def make_forecast_bar(current, projected):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Current", "2-Year Forecast"],
        y=[current, projected],
        marker=dict(color=["#6B7280", "#22C55E"], cornerradius=4),
        text=[f"{current:.1f}%", f"{projected:.1f}%"],
        textposition="outside",
        textfont=dict(color="#F9FAFB", size=14, family="DM Mono"),
        width=0.32,
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=250,
        yaxis=dict(range=[0,120], tickfont=dict(color="#9CA3AF", size=10, family="DM Mono"),
                   ticksuffix="%", gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        xaxis=dict(tickfont=dict(color="#F9FAFB", size=12, family="DM Mono"), showgrid=False),
        showlegend=False, margin=dict(t=35, b=12, l=45, r=15),
        paper_bgcolor="#0B0F14", plot_bgcolor="#0B0F14",
        font=dict(family="DM Mono", color="#9CA3AF")
    )
    return fig

def make_compare_bar(a_name, b_name, a_vals, b_vals, metrics):
    fig = go.Figure()
    for name, vals, color in [(a_name, a_vals, C_GOLD), (b_name, b_vals, C_NEUTRAL)]:
        fig.add_trace(go.Bar(
            name=name, x=metrics, y=vals,
            marker=dict(color=color, cornerradius=3),
            text=[f"{v:.1f}" for v in vals],
            textposition="outside",
            textfont=dict(color="#F9FAFB", size=12, family="DM Mono"),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        barmode="group", height=300,
        yaxis=dict(tickfont=dict(color="#9CA3AF", size=10, family="DM Mono"),
                   gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        xaxis=dict(tickfont=dict(color="#F9FAFB", size=11, family="DM Mono"), showgrid=False),
        legend=dict(font=dict(color="#9CA3AF", size=11, family="DM Mono"),
                    bgcolor="rgba(255,255,255,0.02)", bordercolor="rgba(255,255,255,0.08)",
                    borderwidth=1, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=45, b=12, l=45, r=15),
        paper_bgcolor="#0B0F14", plot_bgcolor="#0B0F14",
        font=dict(family="DM Mono", color="#9CA3AF")
    )
    return fig

def make_radar_compare(a_radar, b_radar, a_name, b_name):
    cats = ["Funding", "Rounds", "Age", "Milestones", "Network"]
    fig = go.Figure()
    for radar, name, color in [(a_radar, a_name, C_GOLD), (b_radar, b_name, C_NEUTRAL)]:
        fig.add_trace(go.Scatterpolar(
            r=radar+[radar[0]], theta=cats+[cats[0]],
            fill="toself", fillcolor=hex_to_rgba(color, 0.28),
            line=dict(color=color, width=2.5), name=name,
            marker=dict(size=5, color=color, line=dict(color="#0B0F14", width=1.5)),
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.01)",
            radialaxis=dict(visible=True, range=[0,100],
                tickfont=dict(color=_TICK, size=7, family="DM Mono"),
                gridcolor=_GRID, linecolor=_GRID,
                tickvals=[0,25,50,75,100], ticktext=["0","25","50","75","100"]),
            angularaxis=dict(tickfont=dict(color="#FFFFFF", size=11, family="DM Sans"),
                gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.08)"),
        ),
        legend=dict(font=dict(color="#9CA3AF", size=10, family="DM Mono"),
                    bgcolor="rgba(255,255,255,0.02)", bordercolor="rgba(255,255,255,0.05)",
                    borderwidth=1, x=1.05, y=0.95),
        height=325, margin=dict(t=12, b=12, l=48, r=96),
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family="DM Mono", color=_TICK),
    )
    return fig


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<style>
[data-testid="stSidebar"] > div:first-child {
    background: #0B0F14 !important;
    border-right: 1px solid #1F2937 !important;
}
[data-testid="stSidebarContent"] { background: #0B0F14 !important; padding: 0 !important; }
</style>
<div style="padding:18px 20px 16px;border-bottom:1px solid #1F2937;display:flex;align-items:center;gap:10px;">
  <div style="width:30px;height:30px;background:rgba(212,175,55,0.1);border:1px solid rgba(212,175,55,0.22);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:14px;color:#D4AF37;font-family:'DM Mono',monospace;flex-shrink:0;">&#9672;</div>
  <div style="font-family:'DM Sans',sans-serif;font-size:15px;font-weight:700;letter-spacing:-0.03em;color:#F9FAFB;line-height:1;">Fund<span style="color:#D4AF37;">IQ</span></div>
</div>
<div style="padding:12px 20px;border-bottom:1px solid #1F2937;">
  <div style="display:inline-flex;align-items:center;gap:7px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.18);border-radius:50px;padding:4px 11px 4px 9px;">
    <span style="width:6px;height:6px;border-radius:50%;background:#22C55E;display:inline-block;"></span>
    <span style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#22C55E;font-weight:600;font-family:'DM Mono',monospace;">Model Online</span>
  </div>
</div>
<div style="margin:16px 20px 0;padding:14px 16px;background:rgba(212,175,55,0.06);border:1px solid rgba(212,175,55,0.16);border-radius:10px;display:flex;align-items:center;justify-content:space-between;">
  <div>
    <span style="font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#9CA3AF;margin-bottom:4px;display:block;font-family:'DM Mono',monospace;">ROC-AUC Score</span>
    <div style="font-family:'DM Sans',sans-serif;font-size:26px;font-weight:700;letter-spacing:-0.04em;color:#D4AF37;line-height:1;">0.965</div>
  </div>
  <div style="font-size:8px;letter-spacing:1.5px;text-transform:uppercase;color:#D4AF37;background:rgba(212,175,55,0.12);border:1px solid rgba(212,175,55,0.22);border-radius:3px;padding:4px 9px;font-weight:600;font-family:'DM Mono',monospace;">Top 2%</div>
</div>
<div style="font-size:7.5px;letter-spacing:2.5px;text-transform:uppercase;color:#9CA3AF;padding:18px 20px 8px;display:block;font-family:'DM Mono',monospace;">Model Statistics</div>
<div style="padding:0 20px;">
  <div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(31,41,55,0.7);"><span style="font-size:9.5px;color:#9CA3AF;font-family:'DM Mono',monospace;">Architecture</span><span style="font-size:9.5px;font-weight:500;color:#F9FAFB;font-family:'DM Mono',monospace;">CatBoost</span></div>
  <div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(31,41,55,0.7);"><span style="font-size:9.5px;color:#9CA3AF;font-family:'DM Mono',monospace;">Train / Test</span><span style="font-size:9.5px;font-weight:500;color:#F9FAFB;font-family:'DM Mono',monospace;">70 / 30</span></div>
  <div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(31,41,55,0.7);"><span style="font-size:9.5px;color:#9CA3AF;font-family:'DM Mono',monospace;">Features</span><span style="font-size:9.5px;font-weight:500;color:#D4AF37;font-family:'DM Mono',monospace;">12 signals</span></div>
  <div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(31,41,55,0.7);"><span style="font-size:9.5px;color:#9CA3AF;font-family:'DM Mono',monospace;">Precision</span><span style="font-size:9.5px;font-weight:500;color:#D4AF37;font-family:'DM Mono',monospace;">94%</span></div>
  <div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(31,41,55,0.7);"><span style="font-size:9.5px;color:#9CA3AF;font-family:'DM Mono',monospace;">Recall</span><span style="font-size:9.5px;font-weight:500;color:#22C55E;font-family:'DM Mono',monospace;">81%</span></div>
  <div style="display:flex;align-items:center;justify-content:space-between;padding:7px 0;"><span style="font-size:9.5px;color:#9CA3AF;font-family:'DM Mono',monospace;">Records</span><span style="font-size:9.5px;font-weight:500;color:#F9FAFB;font-family:'DM Mono',monospace;">462K+</span></div>
</div>
<div style="font-size:7.5px;letter-spacing:2.5px;text-transform:uppercase;color:#9CA3AF;padding:18px 20px 8px;display:block;font-family:'DM Mono',monospace;">12 Input Features</div>
<div style="padding:0 20px 16px;">
""" + "".join([
    f'<div style="display:flex;align-items:center;gap:9px;padding:4px 0;font-size:9.5px;color:#F9FAFB;font-family:\'DM Mono\',monospace;"><div style="width:14px;height:14px;border-radius:3px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.18);display:flex;align-items:center;justify-content:center;font-size:8px;color:#22C55E;flex-shrink:0;">&#10003;</div>{f}</div>'
    for f in ["Funding Efficiency","Funding per Year","Log Funding","Milestones Achieved","Relationships","Investment Rounds","Startup Age","Funding Rounds","Burn Rate","Runway","Growth Intensity","Health Score"]
]) + """
</div>
<div style="padding:14px 20px 20px;border-top:1px solid #1F2937;margin-top:8px;">
  <a href="https://github.com/Nischal-Mahajan/Startup-Health-Intelligence-System" target="_blank" style="display:flex;align-items:center;gap:7px;font-size:9px;letter-spacing:1px;text-transform:uppercase;color:#9CA3AF;text-decoration:none;font-family:'DM Mono',monospace;">&#8599;&nbsp; View on GitHub</a>
  <div style="font-size:8.5px;color:#9CA3AF;margin-top:7px;letter-spacing:0.5px;font-family:'DM Mono',monospace;">Built by Nischal Mahajan</div>
</div>
""", unsafe_allow_html=True)


# ─── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div>
    <div class="page-eyebrow">Startup Decision Support System</div>
    <div class="page-title">Funding <em>Intelligence</em></div>
    <div class="page-desc">CatBoost · 12-signal engine · 462K+ startups analyzed · ROC-AUC 0.965</div>
  </div>
  <div class="header-tags">
    <div class="tag tag-gold"><span class="tag-dot" style="background:#D4AF37"></span>AUC 0.965</div>
    <div class="tag tag-green"><span class="tag-dot" style="background:#22C55E"></span>Precision 94%</div>
    <div class="tag tag-sand"><span class="tag-dot" style="background:#F59E0B"></span>Recall 81%</div>
    <div class="tag tag-gray"><span class="tag-dot" style="background:#6B7280"></span>462K Records</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["  Single Analysis  ", "  Compare Startups  "])


# ══════════════════════════════════════════════════════
# TAB 1 — SINGLE ANALYSIS
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Startup Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: total_funding  = st.number_input("Total Funding ($)", 0.0, 1e10, value=500_000.0, step=10_000.0, format="%.0f")
    with c2: funding_rounds = st.number_input("Funding Rounds",    min_value=0, max_value=50, value=2)
    with c3: inv_rounds     = st.number_input("Investment Rounds", min_value=0, max_value=50, value=1)
    c4, c5, c6 = st.columns(3)
    with c4: startup_age   = st.slider("Startup Age (years)", 0, 30, 3)
    with c5: milestones    = st.number_input("Milestones Achieved",       min_value=0, max_value=100, value=3)
    with c6: relationships = st.number_input("Relationships / Investors", min_value=0, max_value=500, value=5)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Analysis", use_container_width=True)

    if run_btn:
        # Input validation
        if total_funding < 0 or funding_rounds < 0 or milestones < 0 or relationships < 0:
            st.error("All numeric inputs must be non-negative.")
            st.stop()

        with st.spinner("Running CatBoost inference …"):
            # CHANGE 3: Unpack 8 values now (added risk_score)
            prob, hs, base_prob, eff, pyr, log_f, radar_vals, risk_score = predict(
                total_funding, funding_rounds, inv_rounds, startup_age, milestones, relationships
            )

        prob_pct = prob * 100

        # ── Decision classification ──
        if prob_pct >= 70:
            decision_cls = "db-high-potential"
            tag_cls      = "db-tag-hp"
            tag_lbl      = "HIGH POTENTIAL"
            risk, risk_css = "LOW RISK",  "c-green"
            vcls, vcol     = "verdict-success",  "vc-green"
            vicon, vtitle  = "↑", "Highly Likely to Succeed"
            vdesc = "Strong funding pattern and consistent growth signals detected across all key metrics."
            borders = ["accent-gold", "accent-green", "accent-sky", "accent-green"]
            dtitle  = "This startup demonstrates strong fundamentals across multiple signals."
            dsub    = "Model confidence is high — recommend proceeding to due diligence."
        elif prob_pct >= 50:
            decision_cls = "db-moderate"
            tag_cls      = "db-tag-mod"
            tag_lbl      = "MODERATE"
            risk, risk_css = "MED RISK",  "c-sand"
            vcls, vcol     = "verdict-moderate", "vc-sand"
            vicon, vtitle  = "→", "Moderate Potential"
            vdesc = "Mixed signals detected — solid foundation with clear areas for improvement."
            borders = ["accent-gold", "accent-sky", "accent-sky", "accent-sky"]
            dtitle  = "Mixed signals — proceed with targeted improvement actions."
            dsub    = "Focus on the HIGH-priority recommendations below to move into the high-potential tier."
        else:
            decision_cls = "db-high-risk"
            tag_cls      = "db-tag-hr"
            tag_lbl      = "HIGH RISK"
            risk, risk_css = "HIGH RISK", "c-rose"
            vcls, vcol     = "verdict-fail",     "vc-rose"
            vicon, vtitle  = "!", "Elevated Risk Profile"
            vdesc = "Weak financial indicators across multiple dimensions. Significant improvement needed."
            borders = ["accent-gold", "accent-rose", "accent-sky", "accent-rose"]
            dtitle  = "High-risk profile detected across multiple signals."
            dsub    = "Prioritise the HIGH-severity recommendations before approaching institutional investors."

        # ── Success toast ──
        st.markdown('<div class="toast">✓ Analysis complete — results rendered below</div>', unsafe_allow_html=True)

        # ── Decision banner ──
        st.markdown(f"""
        <div class="decision-banner {decision_cls}">
          <span class="db-tag {tag_cls}">{tag_lbl}</span>
          <div>
            <div class="db-title">{dtitle}</div>
            <div class="db-sub">{dsub}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Stat cards ──
        st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card-grid-4">
          <div class="stat-card {borders[0]}">
            <div class="stat-label">Success Probability</div>
            <div class="stat-value c-gold">{prob_pct:.1f}<span style="font-size:0.95rem;font-family:'DM Mono',monospace;font-weight:300;color:var(--text-secondary)">%</span></div>
            <div class="stat-sub">CatBoost · 12-Signal Engine</div>
          </div>
          <div class="stat-card {borders[1]}">
            <div class="stat-label">Health Score</div>
            <div class="stat-value">{hs:.1f}<span style="font-size:0.8rem;color:var(--text-secondary);font-family:'DM Mono',monospace;"> /100</span></div>
            <div class="stat-sub">Composite Signal Index</div>
          </div>
          <div class="stat-card {borders[2]}">
            <div class="stat-label">Model Confidence</div>
            <div class="stat-value c-sky">{base_prob*100:.1f}<span style="font-size:0.95rem;font-family:'DM Mono',monospace;font-weight:300;color:var(--text-secondary)">%</span></div>
            <div class="stat-sub">Raw CatBoost Output</div>
          </div>
          <div class="stat-card {borders[3]}">
            <div class="stat-label">Risk Classification</div>
            <div class="stat-value {risk_css}" style="font-size:0.9rem;font-family:'DM Mono',monospace;padding-top:0.3rem;letter-spacing:1.5px;">{risk}</div>
            <div class="stat-sub">Investor Rating</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Charts ──
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(make_gauge(round(prob_pct, 1), "SUCCESS PROBABILITY"),
                use_container_width=True, config={"displayModeBar": False})
        with ch2:
            st.plotly_chart(make_radar(radar_vals, "Startup", C_GOLD),
                use_container_width=True, config={"displayModeBar": False})

        # ── Verdict ──
        st.markdown(f"""
        <div class="verdict {vcls}">
          <div class="verdict-icon">{vicon}</div>
          <div>
            <div class="verdict-title {vcol}">{vtitle}</div>
            <div class="verdict-desc">{vdesc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Why This Result ──
        st.markdown('<div class="section-label">Why This Result?</div>', unsafe_allow_html=True)
        feat_dict, _, _ = create_features(total_funding, funding_rounds, inv_rounds, startup_age, milestones, relationships)
        drivers = get_top_drivers(feat_dict, prob)

        why_rows = ""
        for feat_name, score in drivers:
            pct    = int(score * 100)
            bar_c  = C_SUCCESS if pct >= 60 else (C_WARNING if pct >= 35 else C_DANGER)
            why_rows += f"""
            <div class="why-row">
              <span class="why-icon">{'✅' if pct>=60 else ('⚠️' if pct>=35 else '🔴')}</span>
              <span class="why-text"><strong>{feat_name}</strong> — scored {pct}/100</span>
              <div style="flex:1;margin-left:0.8rem;">
                <div class="driver-bar-wrap">
                  <div class="driver-bar" style="width:{pct}%;background:{bar_c};"></div>
                </div>
              </div>
              <span class="driver-pct">{pct}%</span>
            </div>"""

        st.markdown(f'<div class="why-box"><div class="why-title">Top 3 Feature Drivers</div>{why_rows}</div>', unsafe_allow_html=True)

        # ── Derived metrics ──
        st.markdown('<div class="section-label">Derived Metrics</div>', unsafe_allow_html=True)
        # CHANGE 4: Added Risk Score metric (k6 column)
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Funding Efficiency",  f"${eff:,.0f}",                    help="Total Funding / (Rounds + 1)")
        k2.metric("Funding per Year",    f"${pyr:,.0f}",                    help="Total Funding / (Age + 1)")
        k3.metric("Log Funding",         f"{log_f:.2f}",                    help="log(1 + Total Funding)")
        k4.metric("Monthly Burn Rate",   f"${feat_dict['burn_rate']:,.0f}", help="Funding / (Age × 12)")
        k5.metric("Runway (months)",     f"{feat_dict['runway']:.0f}",      help="Funding / Burn Rate")
        k6.metric("Risk Score",          f"{risk_score:.1f}",               help="(Burn Rate / (Runway + 1)) × 100")

        # ── Growth Forecast ──
        st.markdown('<div class="section-label">Growth Forecast — Next 2 Years</div>', unsafe_allow_html=True)
        # CHANGE 3: Updated unpack for forecast predict call
        f_prob, f_hs, _, _, _, _, _, _ = predict(
            total_funding * 1.3, funding_rounds + 1, inv_rounds, startup_age + 2, milestones + 2, relationships
        )
        delta = (f_prob - prob) * 100
        g1, g2, g3 = st.columns(3)
        g1.metric("Projected Probability",  f"{f_prob*100:.1f}%", f"{delta:+.1f}%")
        g2.metric("Projected Health Score", f"{f_hs:.1f}/100",    f"{f_hs-hs:+.1f}")
        g3.metric("Growth Delta",           f"{delta:+.1f}%",     "Positive" if delta > 0 else "Flat")
        st.plotly_chart(make_forecast_bar(prob_pct, f_prob*100),
            use_container_width=True, config={"displayModeBar": False})

        # ── Recommendations (rule-based + ML hybrid) ──
        st.markdown('<div class="section-label">Investment Recommendations</div>', unsafe_allow_html=True)
        # CHANGE 5: Pass feat_dict to get_recommendations for engineered-feature based recs
        for dot, title, desc in get_recommendations(total_funding, funding_rounds, milestones, relationships, startup_age, feat=feat_dict):
            badge_cls = {"HIGH": "rb-high", "MED": "rb-med", "OK": "rb-ok"}.get(dot, "rb-ok")
            badge_lbl = {"HIGH": "High",    "MED": "Medium", "OK": "Good"}.get(dot, dot)
            st.markdown(f"""
            <div class="rec-row">
              <span class="rec-badge {badge_cls}">{badge_lbl}</span>
              <div>
                <div class="rec-title">{title}</div>
                <div class="rec-desc">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Edge case warnings ──
        if total_funding == 0:
            st.warning("Zero funding detected — predictions carry higher uncertainty for unfunded startups.")
        if funding_rounds == 0 and milestones == 0:
            st.warning("No rounds or milestones recorded — ensure inputs are accurate before sharing results.")

        # CHANGE 7: Optional SHAP Integration (commented out, ready to enable)
        # explainer = joblib.load("models/shap_explainer.pkl")
        # shap_values = explainer.shap_values(X_scaled)
        # top_idx = np.argsort(np.abs(shap_values[0]))[::-1][:3]


# ══════════════════════════════════════════════════════
# TAB 2 — COMPARE STARTUPS
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">Configure Startups</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:8px;letter-spacing:2.5px;text-transform:uppercase;color:#D4AF37;margin-bottom:0.85rem;">— Startup A</p>', unsafe_allow_html=True)
        a_name   = st.text_input("Name", "Startup A", key="an")
        a_fund   = st.number_input("Total Funding ($)", 0.0, 1e10, value=1_000_000.0, step=10_000.0, key="af", format="%.0f")
        a_rounds = st.number_input("Funding Rounds", 0, 50, value=3, key="ar")
        a_inv    = st.number_input("Investment Rounds", 0, 50, value=2, key="ai")
        a_age    = st.slider("Age (years)", 0, 30, 4, key="aa")
        a_miles  = st.number_input("Milestones", 0, 100, value=5, key="am")
        a_rels   = st.number_input("Relationships", 0, 500, value=8, key="arel")

    with col_b:
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:8px;letter-spacing:2.5px;text-transform:uppercase;color:#38BDF8;margin-bottom:0.85rem;">— Startup B</p>', unsafe_allow_html=True)
        b_name   = st.text_input("Name", "Startup B", key="bn")
        b_fund   = st.number_input("Total Funding ($)", 0.0, 1e10, value=300_000.0, step=10_000.0, key="bf", format="%.0f")
        b_rounds = st.number_input("Funding Rounds", 0, 50, value=1, key="br")
        b_inv    = st.number_input("Investment Rounds", 0, 50, value=1, key="bi")
        b_age    = st.slider("Age (years)", 0, 30, 2, key="ba")
        b_miles  = st.number_input("Milestones", 0, 100, value=2, key="bm")
        b_rels   = st.number_input("Relationships", 0, 500, value=3, key="brel")

    st.markdown("<br>", unsafe_allow_html=True)
    cmp_btn = st.button("Compare Startups", use_container_width=True)

    if cmp_btn:
        with st.spinner("Computing comparison …"):
            # CHANGE 3: Unpack 8 values for both startups
            a_prob, a_hs, _, _, _, a_log_f, a_radar, a_risk = predict(a_fund, a_rounds, a_inv, a_age, a_miles, a_rels)
            b_prob, b_hs, _, _, _, b_log_f, b_radar, b_risk = predict(b_fund, b_rounds, b_inv, b_age, b_miles, b_rels)

        winner = a_name if a_prob >= b_prob else b_name
        w_prob = max(a_prob, b_prob)

        st.markdown('<div class="toast">✓ Comparison complete</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="winner-strip">
          <div>
            <div class="winner-lbl">Recommended Investment</div>
            <div class="winner-name">{winner}</div>
            <div class="winner-prob">Success Probability: {w_prob*100:.1f}%</div>
          </div>
          <div class="winner-badge">↑ Preferred</div>
        </div>""", unsafe_allow_html=True)

        # CHANGE 6: Improved compare reasoning
        a_feat, _, _ = create_features(a_fund, a_rounds, a_inv, a_age, a_miles, a_rels)
        b_feat, _, _ = create_features(b_fund, b_rounds, b_inv, b_age, b_miles, b_rels)
        if a_prob > b_prob:
            reason = "Higher funding efficiency and stronger milestone velocity" if a_feat["funding_efficiency"] > b_feat["funding_efficiency"] else "Better investor network and capital deployment"
        else:
            reason = "Higher funding efficiency and stronger milestone velocity" if b_feat["funding_efficiency"] > a_feat["funding_efficiency"] else "Better investor network and capital deployment"
        st.info(f"Why {winner} is preferred: {reason}")

        st.markdown('<div class="section-label">Head-to-Head Metrics</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{a_name} — Prob",   f"{a_prob*100:.1f}%")
        m2.metric(f"{b_name} — Prob",   f"{b_prob*100:.1f}%")
        m3.metric(f"{a_name} — Health", f"{a_hs:.1f}/100")
        m4.metric(f"{b_name} — Health", f"{b_hs:.1f}/100")

        st.markdown('<div class="section-label">Key Metric Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(make_compare_bar(a_name, b_name,
            [a_prob*100, a_hs, a_log_f*5],
            [b_prob*100, b_hs, b_log_f*5],
            ["Success %", "Health Score", "Log Funding ×5"]),
            use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label">Signal Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar_compare(a_radar, b_radar, a_name, b_name),
            use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-label">Probability Gauges</div>', unsafe_allow_html=True)
        ga, gb = st.columns(2)
        with ga:
            st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:8px;letter-spacing:2.5px;text-transform:uppercase;color:#D4AF37;text-align:center;margin-bottom:0.35rem;">{a_name}</p>', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(round(a_prob*100, 1), a_name.upper()),
                use_container_width=True, config={"displayModeBar": False})
        with gb:
            st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:8px;letter-spacing:2.5px;text-transform:uppercase;color:#38BDF8;text-align:center;margin-bottom:0.35rem;">{b_name}</p>', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(round(b_prob*100, 1), b_name.upper()),
                use_container_width=True, config={"displayModeBar": False})

        # ── Recommendations for each startup ──
        st.markdown('<div class="section-label">Recommendations</div>', unsafe_allow_html=True)
        ra, rb = st.columns(2)
        with ra:
            st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#D4AF37;margin-bottom:0.6rem;">{a_name}</p>', unsafe_allow_html=True)
            # CHANGE 5: Pass feat dict to recommendations
            for dot, title, desc in get_recommendations(a_fund, a_rounds, a_miles, a_rels, a_age, feat=a_feat):
                badge_cls = {"HIGH":"rb-high","MED":"rb-med","OK":"rb-ok"}.get(dot,"rb-ok")
                badge_lbl = {"HIGH":"High","MED":"Medium","OK":"Good"}.get(dot,dot)
                st.markdown(f'<div class="rec-row"><span class="rec-badge {badge_cls}">{badge_lbl}</span><div><div class="rec-title">{title}</div><div class="rec-desc">{desc}</div></div></div>', unsafe_allow_html=True)
        with rb:
            st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:8px;letter-spacing:2px;text-transform:uppercase;color:#38BDF8;margin-bottom:0.6rem;">{b_name}</p>', unsafe_allow_html=True)
            # CHANGE 5: Pass feat dict to recommendations
            for dot, title, desc in get_recommendations(b_fund, b_rounds, b_miles, b_rels, b_age, feat=b_feat):
                badge_cls = {"HIGH":"rb-high","MED":"rb-med","OK":"rb-ok"}.get(dot,"rb-ok")
                badge_lbl = {"HIGH":"High","MED":"Medium","OK":"Good"}.get(dot,dot)
                st.markdown(f'<div class="rec-row"><span class="rec-badge {badge_cls}">{badge_lbl}</span><div><div class="rec-title">{title}</div><div class="rec-desc">{desc}</div></div></div>', unsafe_allow_html=True)

        # ── Edge case warnings ──
        if a_fund == 0:
            st.warning(f"{a_name}: Zero funding — prediction reliability is lower.")
        if b_fund == 0:
            st.warning(f"{b_name}: Zero funding — prediction reliability is lower.")


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-footer">
  <span>FundIQ · Startup Decision Support System · Built by Nischal Mahajan</span>
  <span>CatBoost · ROC-AUC 0.965 · 12 Signals · <a href="https://github.com/Nischal-Mahajan/Startup-Health-Intelligence-System">GitHub ↗</a></span>
</div>
""", unsafe_allow_html=True)