[+      
 
 +]"""
Gini Coefficient Forecasting Model for Years 111-115
Uses dynamic regression with lagged Gini and macro predictors

Key Empirical Relationships Incorporated:
- GDP decline → Gini spike (distributional stress)
- Rapid GDP changes (up or down) → Gini rises
- Gini responds strongly to employment rate and income dispersion
- Expected stable range: 0.48-0.55 unless shock occurs
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: GDP FORECASTS INPUT (UPDATE WHEN AVAILABLE FROM YOUR FRIEND)
# =============================================================================
# TODO: Replace these placeholder values with actual GDP forecasts for years 111-115
GDP_FORECASTS = {
    111: None,  # <- Insert GDP forecast for year 111
    112: None,  # <- Insert GDP forecast for year 112
    113: None,  # <- Insert GDP forecast for year 113
    114: None,  # <- Insert GDP forecast for year 114
    115: None   # <- Insert GDP forecast for year 115
}

# =============================================================================
# STEP 2: LOAD HISTORICAL DATA
# =============================================================================
print("Loading historical data...")

# Load statistics data (has GDP and Gini)
stats_110 = pd.read_csv('data/data/statistics_matka_bots_year110.csv')
stats_105 = pd.read_csv('statistics_matka_bots_year105.csv')

# Load population data (has income distribution details)
pop_110 = pd.read_csv('data/data/population_matka_bots_year110.csv')

# Combine statistics data
stats_data = pd.concat([stats_105, stats_110], ignore_index=True)
stats_data = stats_data.sort_values('year').reset_index(drop=True)

# Calculate Gini coefficient for each year where it's missing
def calculate_gini(incomes):
    """Calculate Gini coefficient from income array"""
    incomes = np.array(incomes)
    incomes = incomes[incomes > 0]  # Remove zero incomes
    if len(incomes) == 0:
        return np.nan
    sorted_incomes = np.sort(incomes)
    n = len(sorted_incomes)
    cumsum = np.cumsum(sorted_incomes)
    return (2 * np.sum((np.arange(1, n + 1)) * sorted_incomes)) / (n * cumsum[-1]) - (n + 1) / n

# Fill in missing Gini values by calculating from population data
for idx, row in stats_data.iterrows():
    if pd.isna(row['gini']):
        year_pop = pop_110[pop_110['year'] == row['year']]
        if len(year_pop) > 0:
            calculated_gini = calculate_gini(year_pop['income'].values)
            stats_data.at[idx, 'gini'] = calculated_gini

# Convert gini to numeric (in case there are strings)
stats_data['gini'] = pd.to_numeric(stats_data['gini'], errors='coerce')

# =============================================================================
# STEP 3: CALCULATE MACRO PREDICTORS
# =============================================================================
print("Calculating predictors...")

# Calculate key variables
stats_data['gdp_growth'] = stats_data['gdp'].pct_change() * 100
stats_data['gdp_growth_abs'] = abs(stats_data['gdp_growth'])  # Rapid changes indicator
stats_data['gdp_declined'] = (stats_data['gdp_growth'] < 0).astype(int)  # Decline indicator

# Calculate income statistics by year
income_stats = []
for year in stats_data['year'].unique():
    year_pop = pop_110[pop_110['year'] == year]
    if len(year_pop) > 0:
        employed = year_pop[year_pop['profession'] != 'unemployed']
        income_stats.append({
            'year': year,
            'avg_income': year_pop['income'].mean(),
            'median_income': year_pop['income'].median(),
            'income_std': year_pop['income'].std(),
            'employment_rate': len(employed) / len(year_pop),
            'total_population': len(year_pop)
        })

income_stats_df = pd.DataFrame(income_stats)
stats_data = stats_data.merge(income_stats_df, on='year', how='left')

# Calculate lagged variables
stats_data['gini_lag1'] = stats_data['gini'].shift(1)
stats_data['gdp_lag1'] = stats_data['gdp'].shift(1)
stats_data['employment_lag1'] = stats_data['employment_rate'].shift(1)

# Income growth
stats_data['income_growth'] = stats_data['avg_income'].pct_change() * 100
stats_data['population_growth'] = stats_data['total_population'].pct_change() * 100

# Drop rows with NaN (first year has no lag)
model_data = stats_data.dropna().copy()

print(f"Historical data: {len(model_data)} years")
print(f"Gini range: {model_data['gini'].min():.3f} - {model_data['gini'].max():.3f}")

# =============================================================================
# STEP 4: TRAIN GINI FORECASTING MODEL
# =============================================================================
print("\nTraining Gini forecasting model...")

# Select predictors (based on economic theory + empirical observations)
# NOTE: Removed gini_lag1 to allow more volatility in predictions
predictor_cols = [
    'gdp_growth',          # GDP dynamics
    'gdp_growth_abs',      # Rapid change indicator
    'gdp_declined',        # Decline shock
    'employment_rate',     # Labor market
    'income_std',          # Income dispersion (KEY for inequality)
    'income_growth'        # Income dynamics
]

# Add interaction terms to capture non-linear effects
model_data['gdp_growth_sq'] = model_data['gdp_growth'] ** 2
model_data['gdp_decline_interaction'] = model_data['gdp_declined'] * model_data['gdp_growth_abs']

predictor_cols_extended = predictor_cols + ['gdp_growth_sq', 'gdp_decline_interaction']

X = model_data[predictor_cols_extended]
y = model_data['gini']

# Train model
model = LinearRegression()
model.fit(X, y)

# Model performance
train_predictions = model.predict(X)
train_rmse = np.sqrt(np.mean((y - train_predictions)**2))
train_r2 = model.score(X, y)
_extended, model.coef_):
    print(f"  {col:30ain_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")

# Show feature importance
print("\nFeature Coefficients:")
for col, coef in zip(predictor_cols, model.coef_):
    print(f"  {col:25s}: {coef:8.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# =============================================================================
# STEP 5: FORECAST GINI FOR YEARS 111-115
# =============================================================================
print("\n" + "="*60)
print("FORECASTING GINI FOR YEARS 111-115")
print("="*60)

# Get last known values (year 110)
last_year_data = model_data.iloc[-1]

# Check if GDP forecasts are provided
gdp_forecasts_available = all(v is not None for v in GDP_FORECASTS.values())

if not gdp_forecasts_available:
    print("\n❌ ERROR: GDP forecasts must be provided!")
    print("=" * 60)
    print("Cannot forecast Gini without GDP values.")
    print("Please update the GDP_FORECASTS dictionary at the top of this script:")
    print("")
    for year in range(111, 116):
        print(f"  {year}: [Your friend's GDP forecast]")
    print("")
    print("Then re-run this script.")
    print("=" * 60)
    exit()

print("✓ GDP forecasts provided. Proceeding with Gini forecasting...\n")

# Recursive forecasting
forecasts = []
current_gini = last_year_data['gini']
current_gdp = last_year_data['gdp']
current_employment = last_year_data['employment_rate']
current_avg_income = last_year_data['avg_income']
current_income_std = last_year_data['income_std']
current_population = last_year_data['total_population']

for year in range(111, 116):
    # Get GDP forecast for this year
    forecast_gdp = GDP_FORECASTS[year]
    
    # Calculate GDP dynamics
    gdp_growth = ((forecast_gdp - current_gdp) / current_gdp) * 100
    gdp_growth_abs = abs(gdp_growth)
    gdp_declined = 1 if gdp_growth < 0 else 0
    
    # Project other variables (simple trends or stability assumptions)
    # Employment tends to be stable unless major shock
    employment_trend = model_data['employment_rate'].tail(10).mean()
    forecast_employment = current_employment * 0.7 + employment_trend * 0.3
    
    # Income std tends to correlate with GDP growth
    income_std_trend = model_data['income_std'].tail(5).mean()
    forecast_income_std = cur with interaction terms
    gdp_growth_sq = gdp_growth ** 2
    gdp_decline_interaction = gdp_declined * gdp_growth_abs
    
    X_forecast = pd.DataFrame({
        'gdp_growth': [gdp_growth],
        'gdp_growth_abs': [gdp_growth_abs],
        'gdp_declined': [gdp_declined],
        'employment_rate': [forecast_employment],
        'income_std': [forecast_income_std],
        'income_growth': [income_growth],
        'gdp_growth_sq': [gdp_growth_sq],
        'gdp_decline_interaction': [gdp_decline_interaction
        'gdp_growth': [gdp_growth],
        'gdp_growth_abs': [gdp_growth_abs],
        'gdp_declined': [gdp_declined],
        'employment_rate': [forecast_employment],
        'income_std': [forecast_income_std],
        'income_growth': [income_growth]
    })
    
    # Predict Gini (base prediction from model)
    predicted_gini = model.predict(X_forecast)[0]
    
    # The GDP predictions already contain event information
    # Model should capture the relationship naturally, but amplify based on empirical patterns:
    
    # 1. GDP DECLINE → Major Gini spike (distributional stress)
    #    Historical pattern shows ~0.05-0.10 Gini increase on declines
    if gdp_declined:
        decline_magnitude = abs(gdp_growth)
        predicted_gini += 0.05 + (decline_magnitude * 0.015)
    
    # 2. RAPID GDP CHANGES (both directions) → Gini increases
    #    Any volatility increases inequality temporarily
    if gdp_growth_abs > 3:
        predicted_gini += (gdp_growth_abs - 3) * 0.012
    
    # 3. Bound within realistic range (wider to allow historical-level spikes)
    predicted_gini = np.clip(predicted_gini, 0.35, 0.70)
    
    # Store forecast
    forecasts.append({
        'year': year,
        'gini_forecast': predicted_gini,
        'gdp_forecast': forecast_gdp,
        'gdp_growth': gdp_growth,
        'employment_rate': forecast_employment,
        'avg_income': forecast_avg_income,
        'income_std': forecast_income_std,
        'gdp_declined': gdp_declined,
        'gdp_source': 'provided' if not gdp_forecasts_available else 'extrapolated'
    })
    
    # Update for next iteration (recursive forecasting)
    current_gini = predicted_gini
    current_gdp = forecast_gdp
    current_employment = forecast_employment
    current_avg_income = forecast_avg_income
    current_income_std = forecast_income_std
    
    print(f"Year {year}: Gini = {predicted_gini:.4f} | GDP = {forecast_gdp:,.0f} | Growth = {gdp_growth:+.2f}%")

forecast_df = pd.DataFrame(forecasts)

# =============================================================================
# STEP 6: SAVE RESULTS
# =============================================================================
print("\nSaving forecasts...")
forecast_df.to_csv('gini_forecasts_111_115.csv', index=False)
print("✓ Saved: gini_forecasts_111_115.csv")

# =============================================================================
# STEP 7: VISUALIZATION
# =============================================================================
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Historical GDP and Gini
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()

historical = model_data.tail(20)
ax1.plot(historical['year'], historical['gdp'], 'b-o', label='GDP', linewidth=2)
ax1_twin.plot(historical['year'], historical['gini'], 'r-s', label='Gini', linewidth=2)

ax1.plot(forecast_df['year'], forecast_df['gdp_forecast'], 'b--o', 
         alpha=0.7, label='GDP Forecast')
ax1_twin.plot(forecast_df['year'], forecast_df['gini_forecast'], 'r--s', 
              alpha=0.7, label='Gini Forecast')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('GDP', color='b', fontsize=12)
ax1.set_title('GDP and Gini Coefficient: Historical + Forecast', fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

ax1_twin.set_ylabel('Gini Coefficient', color='r', fontsize=12)
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1_twin.legend(loc='upper right')

# Plot 2: Gini Forecast Detail
ax2 = axes[0, 1]
all_years = pd.concat([
    model_data.tail(10)[['year', 'gini']].rename(columns={'gini': 'value'}),
    forecast_df[['year', 'gini_forecast']].rename(columns={'gini_forecast': 'value'})
])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

historical_years = model_data.tail(10)['year'].values
forecast_years = forecast_df['year'].values

ax2.plot(historical_years, model_data.tail(10)['gini'], 'go-', 
         linewidth=2, markersize=8, label='Historical Gini')
ax2.plot(forecast_years, forecast_df['gini_forecast'], 'ro--', 
         linewidth=2, markersize=8, label='Forecast Gini')
ax2.axhline(y=0.48, color='gray', linestyle=':', alpha=0.5, label='Stable Range')
ax2.axhline(y=0.55, color='gray', linestyle=':', alpha=0.5)
ax2.fill_between([forecast_years[0]-0.5, forecast_years[-1]+0.5], 0.48, 0.55, 
                  alpha=0.1, color='green')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Gini Coefficient', fontsize=12)
ax2.set_title('Gini Coefficient Forecast (Years 111-115)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: GDP Growth vs Gini Change
ax3 = axes[1, 0]
model_data['gini_change'] = model_data['gini'].diff()

ax3.scatter(model_data['gdp_growth'], model_data['gini_change'], 
           s=80, alpha=0.6, c='blue', edgecolors='black')

# Add forecast points
forecast_df['gini_change_forecast'] = forecast_df['gini_forecast'].diff()
ax3.scatter(forecast_df['gdp_growth'].iloc[1:], 
           forecast_df['gini_change_forecast'].iloc[1:],
           s=100, alpha=0.8, c='red', edgecolors='black', marker='s', 
           label='Forecasts')

ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
ax3.set_xlabel('GDP Growth (%)', fontsize=12)
ax3.set_ylabel('Gini Change', fontsize=12)
ax3.set_title('GDP Growth vs Gini Change\n(Empirical Relationship)', 
             fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Summary Table
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = "FORECAST SUMMARY\n" + "="*50 + "\n\n"
summary_text += f"Model Performance:\n"
summary_text += f"  R² Score: {train_r2:.4f}\n"
summary_text += f"  RMSE: {train_rmse:.4f}\n\n"

summary_text += "Forecasted Gini Coefficients:\n"
for _, row in forecast_df.iterrows():
    summary_text += f"  Year {int(row['year'])}: {row['gini_forecast']:.4f}\n"

summary_text += f"\nForecast Range: {forecast_df['gini_forecast'].min():.4f} - "
summary_text += f"{forecast_df['gini_forecast'].max():.4f}\n"
summary_text += f"Expected Stability: {'✓' if forecast_df['gini_forecast'].max() <= 0.55 else '⚠'}\n\n"

if not gdp_forecasts_available:
    summary_text += "⚠️ NOTE: Using extrapolated GDP values.\n"
    summary_text += "   Update GDP_FORECASTS for final predictions.\n"
else:
    summary_text += "✓ Using provided GDP forecasts\n"

ax4.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('gini_gdp_forecast_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: gini_gdp_forecast_analysis.png")

plt.show()

print("\n" + "="*60)
print("FORECAST COMPLETE!")
print("="*60)
if not gdp_forecasts_available:
    print("\n🔔 REMINDER: Update GDP_FORECASTS at the top of this script")
    print("   when your friend provides the GDP forecasts, then re-run.")
print("\nFiles created:")
print("  - gini_forecasts_111_115.csv")
print("  - gini_gdp_forecast_analysis.png")
