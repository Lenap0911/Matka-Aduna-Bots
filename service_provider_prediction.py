import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Load all relevant data
income_data = pd.read_csv('data/data/profession_income_by_year .csv')
count_data = pd.read_csv('data/data/profession_count_by_year.csv')
gdp_data = pd.read_csv('gpd_analyses_data/update_gdp.csv')

# Filter for service providers
sp_income = income_data[income_data['profession'] == 'service provider'].copy()
sp_count = count_data[count_data['profession'] == 'service provider'].copy()

# Merge service provider data
sp_data = sp_income.merge(sp_count, on='year', suffixes=('', '_count'))
sp_data = sp_data.merge(gdp_data, on='year')

# Calculate total population per year
total_pop = count_data.groupby('year')['count'].sum().reset_index()
total_pop.columns = ['year', 'total_population']
sp_data = sp_data.merge(total_pop, on='year')

# Calculate service provider percentage
sp_data['sp_percentage'] = (sp_data['count'] / sp_data['total_population']) * 100

# Get data for other professions that might correlate
professions = ['civil servant', 'craftsman', 'farmer', 'fisher']
for prof in professions:
    prof_income = income_data[income_data['profession'] == prof][['year', 'avg_income']]
    prof_income.columns = ['year', f'{prof}_income']
    sp_data = sp_data.merge(prof_income, on='year', how='left')
    
    prof_count = count_data[count_data['profession'] == prof][['year', 'count']]
    prof_count.columns = ['year', f'{prof}_count']
    sp_data = sp_data.merge(prof_count, on='year', how='left')

# Analyze correlations
print("="*80)
print("CORRELATION ANALYSIS: FACTORS INFLUENCING SERVICE PROVIDER INCOME")
print("="*80)

correlation_factors = {
    'Year': sp_data['year'].corr(sp_data['avg_income']),
    'Service Provider Count': sp_data['count'].corr(sp_data['avg_income']),
    'GDP': sp_data['gdp'].corr(sp_data['avg_income']),
    'Total Population': sp_data['total_population'].corr(sp_data['avg_income']),
    'SP Percentage': sp_data['sp_percentage'].corr(sp_data['avg_income']),
    'Civil Servant Income': sp_data['civil servant_income'].corr(sp_data['avg_income']),
    'Craftsman Income': sp_data['craftsman_income'].corr(sp_data['avg_income']),
    'Farmer Income': sp_data['farmer_income'].corr(sp_data['avg_income']),
    'Fisher Income': sp_data['fisher_income'].corr(sp_data['avg_income']),
}

# Sort by absolute correlation
sorted_corr = sorted(correlation_factors.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nCorrelation with Service Provider Income (sorted by strength):")
print("-"*80)
for factor, corr in sorted_corr:
    strength = "Very Strong" if abs(corr) > 0.8 else "Strong" if abs(corr) > 0.6 else "Moderate" if abs(corr) > 0.4 else "Weak"
    direction = "Positive" if corr > 0 else "Negative"
    print(f"{factor:30s}: {corr:+.4f} ({strength} {direction})")

# Prepare data for prediction (use years 0-100)
X = sp_data['year'].values.reshape(-1, 1)
y = sp_data['avg_income'].values

# Try different models
models = {}

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)
models['Linear'] = lr_model

# 2. Polynomial Regression (degree 2)
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)
poly2_model = LinearRegression()
poly2_model.fit(X_poly2, y)
models['Polynomial (2)'] = (poly2, poly2_model)

# 3. Polynomial Regression (degree 3)
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X)
poly3_model = LinearRegression()
poly3_model.fit(X_poly3, y)
models['Polynomial (3)'] = (poly3, poly3_model)

# Calculate R² scores
print("\n" + "="*80)
print("MODEL PERFORMANCE (R² Score):")
print("-"*80)
r2_linear = lr_model.score(X, y)
r2_poly2 = poly2_model.score(X_poly2, y)
r2_poly3 = poly3_model.score(X_poly3, y)

print(f"Linear Regression:        {r2_linear:.4f}")
print(f"Polynomial (degree 2):    {r2_poly2:.4f}")
print(f"Polynomial (degree 3):    {r2_poly3:.4f}")

# Choose best model
best_model_name = max([('Linear', r2_linear), ('Poly2', r2_poly2), ('Poly3', r2_poly3)], key=lambda x: x[1])[0]
print(f"\nBest Model: {best_model_name}")

# Predict next 5 years (101-105)
future_years = np.array([101, 102, 103, 104, 105]).reshape(-1, 1)

predictions = {
    'Linear': lr_model.predict(future_years),
    'Polynomial (2)': poly2_model.predict(poly2.transform(future_years)),
    'Polynomial (3)': poly3_model.predict(poly3.transform(future_years))
}

print("\n" + "="*80)
print("PREDICTIONS FOR NEXT 5 YEARS (Years 101-105)")
print("="*80)
print(f"\n{'Year':<8} {'Linear':<15} {'Poly (2)':<15} {'Poly (3)':<15}")
print("-"*80)
for i, year in enumerate([101, 102, 103, 104, 105]):
    print(f"{year:<8} ${predictions['Linear'][i]:<14,.2f} ${predictions['Polynomial (2)'][i]:<14,.2f} ${predictions['Polynomial (3)'][i]:<14,.2f}")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Historical income with predictions
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(sp_data['year'], sp_data['avg_income'], 'o-', label='Historical Data', color='#2E86AB', linewidth=2, markersize=6)

# Plot predictions
years_extended = np.arange(0, 106).reshape(-1, 1)
ax1.plot(years_extended, lr_model.predict(years_extended), '--', label='Linear Prediction', alpha=0.7, linewidth=2)
ax1.plot(years_extended, poly2_model.predict(poly2.transform(years_extended)), '--', label='Poly (2) Prediction', alpha=0.7, linewidth=2)
ax1.plot(years_extended, poly3_model.predict(poly3.transform(years_extended)), '--', label='Poly (3) Prediction', alpha=0.7, linewidth=2)

ax1.axvline(x=100, color='red', linestyle=':', alpha=0.5, label='Current Year')
ax1.fill_between([100, 105], ax1.get_ylim()[0], ax1.get_ylim()[1], alpha=0.1, color='green', label='Prediction Period')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Average Income', fontsize=12)
ax1.set_title('Service Provider Income: Historical & Predictions', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Income vs Service Provider Count
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(sp_data['count'], sp_data['avg_income'], c=sp_data['year'], cmap='viridis', s=100, alpha=0.6)
z = np.polyfit(sp_data['count'], sp_data['avg_income'], 1)
p = np.poly1d(z)
ax2.plot(sp_data['count'], p(sp_data['count']), "r--", alpha=0.8, linewidth=2)
ax2.set_xlabel('Service Provider Count', fontsize=11)
ax2.set_ylabel('Average Income', fontsize=11)
ax2.set_title(f'Income vs SP Count\n(Corr: {correlation_factors["Service Provider Count"]:+.3f})', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Income vs GDP
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(sp_data['gdp'], sp_data['avg_income'], c=sp_data['year'], cmap='plasma', s=100, alpha=0.6)
z = np.polyfit(sp_data['gdp'], sp_data['avg_income'], 1)
p = np.poly1d(z)
ax3.plot(sp_data['gdp'], p(sp_data['gdp']), "r--", alpha=0.8, linewidth=2)
ax3.set_xlabel('GDP', fontsize=11)
ax3.set_ylabel('Average Income', fontsize=11)
ax3.set_title(f'Income vs GDP\n(Corr: {correlation_factors["GDP"]:+.3f})', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Income vs Total Population
ax4 = fig.add_subplot(gs[1, 2])
ax4.scatter(sp_data['total_population'], sp_data['avg_income'], c=sp_data['year'], cmap='coolwarm', s=100, alpha=0.6)
z = np.polyfit(sp_data['total_population'], sp_data['avg_income'], 1)
p = np.poly1d(z)
ax4.plot(sp_data['total_population'], p(sp_data['total_population']), "r--", alpha=0.8, linewidth=2)
ax4.set_xlabel('Total Population', fontsize=11)
ax4.set_ylabel('Average Income', fontsize=11)
ax4.set_title(f'Income vs Population\n(Corr: {correlation_factors["Total Population"]:+.3f})', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Correlation heatmap (top factors)
ax5 = fig.add_subplot(gs[2, 0])
top_factors = sorted_corr[:8]
factor_names = [f[0] for f in top_factors]
factor_values = [f[1] for f in top_factors]
colors = ['green' if v > 0 else 'red' for v in factor_values]
ax5.barh(factor_names, factor_values, color=colors, alpha=0.7)
ax5.axvline(x=0, color='black', linewidth=0.8)
ax5.set_xlabel('Correlation Coefficient', fontsize=11)
ax5.set_title('Top Influencing Factors', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# Plot 6: Year-over-year change
ax6 = fig.add_subplot(gs[2, 1])
sp_data['income_change'] = sp_data['avg_income'].diff()
ax6.bar(sp_data['year'][1:], sp_data['income_change'][1:], color=['green' if x > 0 else 'red' for x in sp_data['income_change'][1:]], alpha=0.7)
ax6.axhline(y=0, color='black', linewidth=0.8)
ax6.set_xlabel('Year', fontsize=11)
ax6.set_ylabel('Income Change', fontsize=11)
ax6.set_title('Year-over-Year Income Change', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Prediction confidence intervals
ax7 = fig.add_subplot(gs[2, 2])
recent_years = sp_data[sp_data['year'] >= 90]
ax7.plot(recent_years['year'], recent_years['avg_income'], 'o-', label='Actual', color='#2E86AB', linewidth=2, markersize=8)
pred_years = np.array([101, 102, 103, 104, 105])
poly2_preds = poly2_model.predict(poly2.transform(pred_years.reshape(-1, 1)))
ax7.plot(pred_years, poly2_preds, 's-', label='Predicted (Poly 2)', color='#FB5607', linewidth=2, markersize=8)
ax7.axvline(x=100.5, color='red', linestyle=':', alpha=0.5)
ax7.fill_between([100.5, 105], ax7.get_ylim()[0], ax7.get_ylim()[1], alpha=0.1, color='green')
ax7.set_xlabel('Year', fontsize=11)
ax7.set_ylabel('Average Income', fontsize=11)
ax7.set_title('Recent Trend & 5-Year Forecast', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.suptitle('Service Provider Income Analysis & Prediction', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('service_provider_trend_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print(f"\n1. Historical Trend:")
print(f"   - Average annual growth rate: {((sp_data['avg_income'].iloc[-1] / sp_data['avg_income'].iloc[0]) ** (1/100) - 1) * 100:.2f}%")
print(f"   - Total change (Year 0 to 100): ${sp_data['avg_income'].iloc[-1] - sp_data['avg_income'].iloc[0]:.2f}")

print(f"\n2. Strongest Influencing Factors:")
for i, (factor, corr) in enumerate(sorted_corr[:3], 1):
    print(f"   {i}. {factor}: {corr:+.4f}")

print(f"\n3. Recommended Prediction (Polynomial degree 2):")
for i, year in enumerate([101, 102, 103, 104, 105]):
    change = predictions['Polynomial (2)'][i] - sp_data['avg_income'].iloc[-1]
    pct_change = (change / sp_data['avg_income'].iloc[-1]) * 100
    print(f"   Year {year}: ${predictions['Polynomial (2)'][i]:,.2f} ({pct_change:+.2f}% from Year 100)")

print("\n" + "="*80)
