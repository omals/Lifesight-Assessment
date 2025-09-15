# Assessment 2: Marketing Mix Modeling with Mediation Assumption

## 📌 Context
Dataset: 2 years of weekly data including:
- Paid media (Facebook, Google, TikTok, Instagram, Snapchat)
- Direct levers (Emails, SMS)
- Commercial levers (Price, Promotions)
- Social followers and Revenue

## 🎯 Task
Build a machine learning model that:
- Explains revenue as a function of the above inputs.
- Treats **Google spend as a mediator** between social channels (FB/TikTok/Snapchat) and Revenue.

## 🛠️ Approach
1. **Data Preparation**
   - Handled weekly seasonality and trend (week-of-year, lags).
   - Applied log transforms and zero-spend indicators.
   - Added adstock transformations with tuned decay factors.
2. **Modeling**
   - Baseline: Ridge regression with rolling CV.
   - Mediation-aware: Two-stage pipeline (Social → Google → Revenue).
   - Non-linear check: XGBoost + SHAP explanations.
3. **Causal Framing**
   - Explicit DAG structure.
   - Mediation tests (Baron & Kenny, Sobel, bootstrap).
4. **Diagnostics**
   - Rolling CV performance metrics (RMSE, R², MAPE).
   - Residual plots and ACF checks.
   - Sensitivity to price and promotions.
5. **Insights**
   - Identified main revenue drivers and mediated effects.
   - Price elasticity and promo impact quantified.
   - Risks: multicollinearity, model assumptions.

## 📊 Tools
- Python: Pandas, Numpy, Scikit-learn, Statsmodels, XGBoost, SHAP, Matplotlib, Seaborn.
- Jupyter Notebook: `Marketing_MMM_Assessment_final.ipynb`.

## 🚀 Run Locally
```bash
cd assessment2_mmm
jupyter notebook Marketing_MMAssessment_Modeling.ipynb
