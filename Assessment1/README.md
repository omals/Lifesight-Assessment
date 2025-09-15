# Assessment 1: Marketing Intelligence Dashboard

## 📌 Context
You are provided four datasets covering 120 days of activity:
- **Facebook.csv, Google.csv, TikTok.csv** → Campaign-level marketing data.  
- **Business.csv** → Daily business performance.  

## 🎯 Task
Design and host an interactive BI dashboard that connects marketing activity with business outcomes.

## 🛠️ Approach
1. **Data Preparation**
   - Joined and aggregated campaign datasets with business data.
   - Derived metrics: CTR, CPC, ROAS, CAC, Gross Margin.
2. **Visualization & Storytelling**
   - Interactive dashboard with spend vs revenue trends.
   - Attribution of channel effectiveness.
   - Customer growth funnel visualization.
3. **Product Thinking**
   - Focused on insights a marketing leader would use (e.g., channel ROI, cost efficiency).
   - Clear KPIs and decision-enabling dashboards.

## 📊 Tools
- Python (Pandas, Plotly, Streamlit) **or** BI tools like Power BI / Tableau.
- Deployed via [Streamlit Cloud / PowerBI Service / Tableau Public].

## 🚀 Run Locally (Streamlit Example)
```bash
cd .\Assessment1\Marketing_Intelligence_Dashboard\
streamlit run app.py
