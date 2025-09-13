# Credit Risk Analysis using Financial Ratios

## 📌 Overview
This project evaluates the credit risk of publicly traded companies using financial ratios, Altman Z-Score, and a custom risk scoring model. The analysis helps investors and financial analysts identify safe, moderate, and high-risk companies.

## 🚀 Features
- Fetches **real-time financial data** from Yahoo Finance.
- Computes **key financial ratios** (liquidity, leverage, profitability, efficiency).
- Calculates the **Altman Z-Score** for bankruptcy prediction.
- Builds a **comprehensive risk score (0–100)** combining multiple indicators.
- Generates **visualizations** including:
  - Risk Score Distribution
  - Z-Score by Company
  - Credit Risk Category Pie Chart
  - Risk Score vs Z-Score Scatter Plot
- Provides **investment recommendations** (top 3 safest and riskiest companies).

## 🛠️ Technologies Used
- **Python**  
- **Pandas, NumPy** – Data manipulation and numerical analysis  
- **Matplotlib** – Data visualization  
- **SciPy** – Statistical calculations  
- **yFinance** – Financial data fetching  

## 📊 How It Works
1. Selects a list of companies (tickers).  
2. Fetches their **balance sheet, income statement, and cash flow data**.  
3. Computes financial ratios and **Altman Z-Score**.  
4. Assigns a **composite risk score (0–100)**.  
5. Displays results in tabular format and generates visual insights.  

## 📈 Example Output
- Ranked companies by risk score  
- Safe Zone, Grey Zone, Distress Zone classification  
- Investment recommendations for **top 3 safest and riskiest companies**  

## 📌 Future Improvements
- Expand to include **more companies/sectors**.  
- Integrate **machine learning models** for predictive credit risk analysis.  
- Add **interactive dashboards** with Plotly or Streamlit.  

---
