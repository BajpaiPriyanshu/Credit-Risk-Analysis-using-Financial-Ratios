import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

companies = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'JPM', 'BAC', 'WMT']
print("Analyzing credit risk for companies:", companies)

def get_financial_data(ticker):
    """
    Fetch financial data for a given ticker symbol
    Returns key financial metrics needed for credit risk analysis
    """
    try:
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
        if not balance_sheet.empty and not income_stmt.empty:
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
            current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else 0
            current_liab = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else 0
            total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else 0
            retained_earnings = balance_sheet.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in balance_sheet.index else 0
            revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 0
            ebit = income_stmt.loc['EBIT'].iloc[0] if 'EBIT' in income_stmt.index else 0
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            interest_expense = abs(income_stmt.loc['Interest Expense'].iloc[0]) if 'Interest Expense' in income_stmt.index else 1
            market_cap = stock.info.get('marketCap', total_equity)
            return {
                'ticker': ticker,
                'total_assets': total_assets,
                'current_assets': current_assets,
                'current_liabilities': current_liab,
                'total_debt': total_debt,
                'total_equity': total_equity,
                'retained_earnings': retained_earnings,
                'revenue': revenue,
                'ebit': ebit,
                'net_income': net_income,
                'interest_expense': interest_expense,
                'market_cap': market_cap
            }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_financial_ratios(data):
    """
    Calculate key financial ratios for credit risk analysis
    """
    ratios = {}
    working_capital = data['current_assets'] - data['current_liabilities']
    ratios['wc_to_assets'] = working_capital / data['total_assets'] if data['total_assets'] > 0 else 0
    ratios['re_to_assets'] = data['retained_earnings'] / data['total_assets'] if data['total_assets'] > 0 else 0
    ratios['ebit_to_assets'] = data['ebit'] / data['total_assets'] if data['total_assets'] > 0 else 0
    ratios['equity_to_debt'] = data['market_cap'] / data['total_debt'] if data['total_debt'] > 0 else float('inf')
    ratios['sales_to_assets'] = data['revenue'] / data['total_assets'] if data['total_assets'] > 0 else 0
    ratios['debt_to_equity'] = data['total_debt'] / data['total_equity'] if data['total_equity'] > 0 else float('inf')
    ratios['interest_coverage'] = data['ebit'] / data['interest_expense'] if data['interest_expense'] > 0 else float('inf')
    ratios['roa'] = data['net_income'] / data['total_assets'] if data['total_assets'] > 0 else 0
    return ratios

def calculate_altman_z_score(ratios):
    """
    Calculate Altman Z-Score for bankruptcy prediction
    """
    z_score = (1.2 * ratios['wc_to_assets'] +
               1.4 * ratios['re_to_assets'] +
               3.3 * ratios['ebit_to_assets'] +
               0.6 * min(ratios['equity_to_debt'], 10) +
               1.0 * ratios['sales_to_assets'])
    return z_score

def interpret_z_score(z_score):
    """
    Interpret Altman Z-Score to determine credit risk level
    """
    if z_score > 2.99:
        return "Safe Zone"
    elif z_score > 1.8:
        return "Grey Zone"
    else:
        return "Distress Zone"

def create_risk_score(ratios, z_score):
    """
    Create a comprehensive credit risk score (0-100, higher is better)
    """
    z_component = min(max(z_score * 10, 0), 40)
    ic_component = min(ratios['interest_coverage'] * 2, 25)
    if ratios['debt_to_equity'] == float('inf'):
        de_component = 0
    else:
        de_component = max(20 - ratios['debt_to_equity'] * 5, 0)
    roa_component = min(max(ratios['roa'] * 100, 0), 15)
    risk_score = z_component + ic_component + de_component + roa_component
    return min(risk_score, 100)

print("Fetching financial data for companies...")
financial_data = []

for ticker in companies:
    print(f"Processing {ticker}...")
    data = get_financial_data(ticker)
    if data:
        ratios = calculate_financial_ratios(data)
        z_score = calculate_altman_z_score(ratios)
        risk_interpretation = interpret_z_score(z_score)
        risk_score = create_risk_score(ratios, z_score)
        company_analysis = {
            'Company': ticker,
            'Z_Score': round(z_score, 2),
            'Risk_Category': risk_interpretation,
            'Risk_Score': round(risk_score, 1),
            'Debt_to_Equity': round(ratios['debt_to_equity'], 2) if ratios['debt_to_equity'] != float('inf') else 'N/A',
            'Interest_Coverage': round(ratios['interest_coverage'], 2) if ratios['interest_coverage'] != float('inf') else 'N/A',
            'ROA': round(ratios['roa'] * 100, 2),
            'Working_Capital_Ratio': round(ratios['wc_to_assets'], 3),
            'Asset_Turnover': round(ratios['sales_to_assets'], 2)
        }
        financial_data.append(company_analysis)

df = pd.DataFrame(financial_data)
print("\nCreated analysis DataFrame with", len(df), "companies")

df_sorted = df.sort_values('Risk_Score', ascending=False)
print("\n" + "="*60)
print("CREDIT RISK ANALYSIS RESULTS")
print("="*60)
print(df_sorted.to_string(index=False))

print(f"\n{'='*60}")
print("STATISTICAL SUMMARY")
print("="*60)
print(f"Average Z-Score: {df['Z_Score'].mean():.2f}")
print(f"Median Risk Score: {df['Risk_Score'].median():.1f}")
print(f"Companies in Safe Zone: {len(df[df['Risk_Category'] == 'Safe Zone'])}")
print(f"Companies in Grey Zone: {len(df[df['Risk_Category'] == 'Grey Zone'])}")
print(f"Companies in Distress Zone: {len(df[df['Risk_Category'] == 'Distress Zone'])}")

print("\nGenerating visualizations...")

plt.style.use('default')

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.hist(df['Risk_Score'], bins=8, alpha=0.7, color='skyblue', edgecolor='black')
ax1.set_title('Distribution of Credit Risk Scores', fontsize=14, fontweight='bold')
ax1.set_xlabel('Risk Score (Higher is Better)')
ax1.set_ylabel('Number of Companies')
ax1.grid(True, alpha=0.3)

bars = ax2.bar(df_sorted['Company'], df_sorted['Z_Score'],
               color=['green' if x > 2.99 else 'orange' if x > 1.8 else 'red'
                      for x in df_sorted['Z_Score']])
ax2.set_title('Altman Z-Score by Company', fontsize=14, fontweight='bold')
ax2.set_xlabel('Company')
ax2.set_ylabel('Z-Score')
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(y=2.99, color='green', linestyle='--', alpha=0.7, label='Safe Zone')
ax2.axhline(y=1.8, color='orange', linestyle='--', alpha=0.7, label='Grey Zone')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

risk_counts = df['Risk_Category'].value_counts()
colors = ['green', 'orange', 'red']
ax3.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
        colors=colors[:len(risk_counts)])
ax3.set_title('Credit Risk Category Distribution', fontsize=14, fontweight='bold')

scatter = ax4.scatter(df['Z_Score'], df['Risk_Score'],
                      c=df['Z_Score'], cmap='RdYlGn', s=100,
                      alpha=0.7, edgecolors='black')
ax4.set_title('Risk Score vs Altman Z-Score', fontsize=14, fontweight='bold')
ax4.set_xlabel('Altman Z-Score')
ax4.set_ylabel('Comprehensive Risk Score')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Z-Score Value')

for i, txt in enumerate(df['Company']):
    ax4.annotate(txt, (df['Z_Score'].iloc[i], df['Risk_Score'].iloc[i]),
                 xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print("INVESTMENT RECOMMENDATIONS")
print("="*60)

top_companies = df_sorted.head(3)
print("TOP 3 COMPANIES (Lowest Credit Risk):")
for idx, row in top_companies.iterrows():
    print(f"• {row['Company']}: Risk Score {row['Risk_Score']}, Z-Score {row['Z_Score']} ({row['Risk_Category']})")

print("\nHIGHEST RISK COMPANIES:")
risky_companies = df_sorted.tail(3)
for idx, row in risky_companies.iterrows():
    print(f"• {row['Company']}: Risk Score {row['Risk_Score']}, Z-Score {row['Z_Score']} ({row['Risk_Category']})")

print(f"\n{'='*60}")
print("KEY INSIGHTS")
print("="*60)
print("• Companies with Z-Score > 2.99 are in the 'Safe Zone' with low bankruptcy risk")
print("• Companies with Z-Score 1.8-2.99 are in the 'Grey Zone' requiring closer monitoring")
print("• Companies with Z-Score < 1.8 are in the 'Distress Zone' with high bankruptcy risk")
print("• Higher Risk Scores indicate better overall financial health")
print("• Consider debt-to-equity and interest coverage ratios for lending decisions")

print("\nAnalysis completed successfully! ")
