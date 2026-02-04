import pandas as pd
import numpy as np

# Load all relevant data
print("Loading data...")
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

# Calculate service provider percentage and market share
sp_data['sp_percentage'] = (sp_data['count'] / sp_data['total_population']) * 100
sp_data['gdp_per_capita'] = sp_data['gdp'] / sp_data['total_population']

# Get data for other professions
professions = ['civil servant', 'craftsman', 'farmer', 'fisher', 'unemployed', 'homemaker']
for prof in professions:
    prof_income = income_data[income_data['profession'] == prof][['year', 'avg_income']]
    prof_income.columns = ['year', f'{prof}_income']
    sp_data = sp_data.merge(prof_income, on='year', how='left')
    
    prof_count = count_data[count_data['profession'] == prof][['year', 'count']]
    prof_count.columns = ['year', f'{prof}_count']
    sp_data = sp_data.merge(prof_count, on='year', how='left')

# Calculate economic indicators
sp_data['avg_other_income'] = sp_data[['civil servant_income', 'craftsman_income', 'farmer_income', 'fisher_income']].mean(axis=1)
sp_data['wealth_gap'] = sp_data['avg_income'] - sp_data['avg_other_income']
sp_data['employed_population'] = sp_data['total_population'] - sp_data.get('unemployed_count', 0)

# Calculate correlations
print("\n" + "="*90)
print("WHAT THE DATA TELLS YOU ABOUT SERVICE PROVIDER INCOME")
print("="*90)

# Basic statistics
print(f"\n1. BASIC PATTERNS (Years 0-100):")
print(f"   Starting income (Year 0):    ${sp_data['avg_income'].iloc[0]:,.2f}")
print(f"   Current income (Year 100):   ${sp_data['avg_income'].iloc[-1]:,.2f}")
print(f"   Total change:                ${sp_data['avg_income'].iloc[-1] - sp_data['avg_income'].iloc[0]:,.2f}")
print(f"   Average annual growth:       {((sp_data['avg_income'].iloc[-1] / sp_data['avg_income'].iloc[0]) ** (1/100) - 1) * 100:.2f}%")
print(f"   Peak income:                 ${sp_data['avg_income'].max():,.2f} (Year {sp_data.loc[sp_data['avg_income'].idxmax(), 'year']:.0f})")
print(f"   Lowest income:               ${sp_data['avg_income'].min():,.2f} (Year {sp_data.loc[sp_data['avg_income'].idxmin(), 'year']:.0f})")

# Volatility
income_changes = sp_data['avg_income'].diff()
print(f"\n2. VOLATILITY:")
print(f"   Average year-to-year change: ${income_changes.mean():,.2f}")
print(f"   Standard deviation:          ${sp_data['avg_income'].std():,.2f}")
print(f"   Largest increase:            ${income_changes.max():,.2f} (Year {sp_data.loc[income_changes.idxmax(), 'year']:.0f})")
print(f"   Largest decrease:            ${income_changes.min():,.2f} (Year {sp_data.loc[income_changes.idxmin(), 'year']:.0f})")

# Correlation analysis
print("\n" + "="*90)
print("VARIABLES THAT COULD HELP PREDICT SERVICE PROVIDER INCOME")
print("="*90)

correlation_factors = {
    'Year (Time Trend)': sp_data['year'].corr(sp_data['avg_income']),
    'GDP': sp_data['gdp'].corr(sp_data['avg_income']),
    'GDP per Capita': sp_data['gdp_per_capita'].corr(sp_data['avg_income']),
    'Total Population': sp_data['total_population'].corr(sp_data['avg_income']),
    'Service Provider Count': sp_data['count'].corr(sp_data['avg_income']),
    'SP Market Share (%)': sp_data['sp_percentage'].corr(sp_data['avg_income']),
    'Civil Servant Income': sp_data['civil servant_income'].corr(sp_data['avg_income']),
    'Craftsman Income': sp_data['craftsman_income'].corr(sp_data['avg_income']),
    'Farmer Income': sp_data['farmer_income'].corr(sp_data['avg_income']),
    'Fisher Income': sp_data['fisher_income'].corr(sp_data['avg_income']),
    'Average Other Profession Income': sp_data['avg_other_income'].corr(sp_data['avg_income']),
    'Farmer Count': sp_data['farmer_count'].corr(sp_data['avg_income']),
    'Craftsman Count': sp_data['craftsman_count'].corr(sp_data['avg_income']),
}

# Sort by absolute correlation
sorted_corr = sorted(correlation_factors.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nPREDICTIVE POWER RANKING (Correlation with SP Income):")
print("-"*90)
print(f"{'Rank':<6} {'Variable':<40} {'Correlation':<15} {'Strength':<15}")
print("-"*90)

for rank, (factor, corr) in enumerate(sorted_corr, 1):
    if abs(corr) > 0.8:
        strength = "Very Strong"
        symbol = "★★★★★"
    elif abs(corr) > 0.6:
        strength = "Strong"
        symbol = "★★★★"
    elif abs(corr) > 0.4:
        strength = "Moderate"
        symbol = "★★★"
    elif abs(corr) > 0.2:
        strength = "Weak"
        symbol = "★★"
    else:
        strength = "Very Weak"
        symbol = "★"
    
    direction = "↑" if corr > 0 else "↓"
    print(f"{rank:<6} {factor:<40} {corr:+.4f} {direction:<4}  {symbol} {strength}")

# Key insights
print("\n" + "="*90)
print("KEY INSIGHTS & INTERPRETATION")
print("="*90)

# Find top 3 correlations
top_3 = sorted_corr[:3]

print(f"\n📊 STRONGEST PREDICTORS:")
for i, (factor, corr) in enumerate(top_3, 1):
    direction = "increases" if corr > 0 else "decreases"
    print(f"\n{i}. {factor} (r = {corr:+.4f})")
    if abs(corr) > 0.8:
        print(f"   → VERY STRONG relationship: When {factor.lower()} {direction}, SP income {direction} proportionally")
        print(f"   → This is an EXCELLENT predictor - explains {(corr**2)*100:.1f}% of income variation")
    elif abs(corr) > 0.6:
        print(f"   → STRONG relationship: {factor} is a good predictor of SP income")
        print(f"   → Explains {(corr**2)*100:.1f}% of income variation")
    elif abs(corr) > 0.4:
        print(f"   → MODERATE relationship: {factor} has some predictive value")
        print(f"   → Explains {(corr**2)*100:.1f}% of income variation")

# Actionable insights
print("\n" + "="*90)
print("WHAT THIS MEANS FOR PREDICTION")
print("="*90)

print("\n✓ BEST VARIABLES TO USE FOR PREDICTION:")
strong_predictors = [f for f, c in sorted_corr if abs(c) > 0.6]
for pred in strong_predictors[:5]:
    print(f"   • {pred}")

print("\n✗ WEAK PREDICTORS (Less useful):")
weak_predictors = [f for f, c in sorted_corr if abs(c) < 0.3]
for pred in weak_predictors[-3:]:
    print(f"   • {pred}")

# Economic interpretation
print("\n" + "="*90)
print("ECONOMIC INTERPRETATION")
print("="*90)

if sp_data['gdp'].corr(sp_data['avg_income']) > 0.6:
    print("\n💰 GDP Relationship:")
    print("   Service provider income is STRONGLY tied to overall economic health (GDP).")
    print("   As the economy grows, service providers earn more - they benefit from economic prosperity.")

if sp_data['total_population'].corr(sp_data['avg_income']) > 0.4:
    print("\n👥 Population Effect:")
    print("   More people = more customers = higher service provider income.")
    print("   Service providers benefit from market expansion.")

if sp_data['count'].corr(sp_data['avg_income']) < -0.3:
    print("\n⚖️ Competition Effect:")
    print("   MORE service providers = LOWER individual income (negative correlation).")
    print("   This suggests market saturation - too many providers competing for same customers.")
elif sp_data['count'].corr(sp_data['avg_income']) > 0.3:
    print("\n📈 Market Growth:")
    print("   More service providers AND higher income suggests a GROWING market.")
    print("   The industry is expanding faster than worker supply.")

# Trend analysis
recent_20 = sp_data.tail(20)
recent_trend = recent_20['avg_income'].iloc[-1] - recent_20['avg_income'].iloc[0]

print("\n📈 RECENT TREND (Last 20 years):")
if recent_trend > 0:
    print(f"   ↗️  UPWARD trend: Income increased by ${recent_trend:,.2f}")
    print("   → Expect continued growth if conditions remain stable")
else:
    print(f"   ↘️  DOWNWARD trend: Income decreased by ${abs(recent_trend):,.2f}")
    print("   → Market may be saturating or facing challenges")

print("\n" + "="*90)
print("RECOMMENDATION FOR PREDICTION MODEL")
print("="*90)
print("\nTo predict service provider income for the next 5 years, use:")
print("\n1. PRIMARY PREDICTORS (include these):")
for i, (factor, corr) in enumerate(sorted_corr[:3], 1):
    print(f"   {i}. {factor} (r = {corr:+.4f})")

print("\n2. MODEL APPROACH:")
if sorted_corr[0][1] > 0.8:
    print("   → Use POLYNOMIAL REGRESSION (2nd or 3rd degree)")
    print("   → The very strong correlations suggest a clear mathematical relationship")
else:
    print("   → Use MULTIPLE LINEAR REGRESSION")
    print("   → Combine several moderate predictors for better accuracy")

print("\n3. DATA NEEDED FOR PREDICTION:")
print("   • Historical income data (years 0-100)")
print("   • Future estimates of top predictor variables for years 101-105")
print("   • If GDP is a top predictor, you'll need GDP forecasts")

print("\n" + "="*90)

# Save insights to file
sp_data.to_csv('service_provider_analysis_data.csv', index=False)
print("\n✓ Detailed data saved to: service_provider_analysis_data.csv")
print("="*90)
