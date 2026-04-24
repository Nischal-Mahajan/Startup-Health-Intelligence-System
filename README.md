# 🚀 Startup Health Intelligence System
An end-to-end Machine Learning system that predicts startup success probability, failure risk, and generates a custom health score using funding-based signals.
---
## 🔗 Live Demo
https://startup-health-intelligence-system-4ikqyscwe2trtuq8bkfmwz.streamlit.app/
---
## 🧠 Problem Statement
Investors and founders often struggle to evaluate startup potential in early stages.
This system leverages historical funding patterns, growth signals, and engineered features to provide data-driven insights on startup success, risk, and financial health.
---
## ⚡ Key Features
### 🔧 Feature Engineering (Core USP)
- Funding Efficiency  
- Funding per Year  
- Log Funding  
- Runway (Months)  
- Burn Rate Estimation  
---
### 🤖 Model Training
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- XGBoost / CatBoost (Final Optimized Model)
---
### 📊 Intelligent Outputs
- Success Probability (with calibrated confidence)  
- Startup Health Score (0–100)  
- Risk Classification (Low / Medium / High)  
- Feature Driver Analysis (Why this prediction?)  
---
### 🧩 Product-Level Enhancements (Latest Improvements)
- Introduced probability cap (~85%) to avoid overconfident predictions  
- Added feature-level explainability (Runway, Funding Efficiency, etc.)  
- Built multi-startup comparison system  
- Added derived financial metrics (Burn Rate, Runway, Funding Efficiency)  
- Improved interpretability with insights and recommendations  
---
## 📈 Key Insights
- Feature engineering significantly improved model performance  
- Dataset was highly imbalanced → prioritized Recall and ROC-AUC  
- Tree-based models captured complex funding patterns effectively  
- XGBoost / CatBoost achieved best overall performance  
---
## 🧪 Model Evaluation

| Model               | Insight                                       | ROC-AUC       |
|--------------------|----------------------------------------------|---------------|
| Logistic Regression | Limited with non-linear relationships        | Baseline      |
| Random Forest       | Strong baseline, handles feature interactions | Good          |
| Gradient Boosting   | Better on hard-to-classify cases             | Better        |
| XGBoost / CatBoost  | Best overall performance                    | **0.96** ⭐   |

---
## 📸 App Screenshots

### 🔍 Startup Prediction Dashboard
![Prediction Dashboard](https://startup-health-intelligence-system-4ikqyscwe2trtuq8bkfmwz.streamlit.app/)

> 💡 **[👉 Try the Live App Here](https://startup-health-intelligence-system-4ikqyscwe2trtuq8bkfmwz.streamlit.app/)** — Enter any startup's funding details and get instant predictions.

---
## 🏗️ System Architecture
User Input → Feature Engineering → ML Model → Prediction Engine → Insights + Visualization
---
## 📁 Project Structure
```
Startup-Health-Intelligence-System/
│
├── dashboard/        # Streamlit application (UI + inference)
├── models/           # Trained pipelines and model files
├── notebooks/        # EDA, feature engineering, training
├── README.md
├── requirements.txt
```
---
## ▶️ Run Locally
```bash
cd dashboard
streamlit run app.py
```
---
## Tech Stack
* Python
* Scikit-learn
* XGBoost / CatBoost
* Pandas, NumPy
* Streamlit
* SHAP (Explainability)

---
## 💡 What Makes This Project Unique
- Strong focus on feature engineering as a key differentiator  
- Combines machine learning predictions with business-level insights  
- Goes beyond prediction by including explainability and decision support  
- Designed as a complete product, not just a standalone model  
---
## 🚀 Future Improvements
- Integrate LLM-based explanations for natural language insights  
- Build an Agentic AI layer for automated startup analysis  
- Add real-time data integration using a RAG pipeline  
- Improve probability calibration and overall model reliability
---
## Author
Nischal Mahajan
---
## Summary
This project demonstrates a complete machine learning pipeline including data preprocessing, feature engineering, model selection, and deployment in a real-world scenario.
