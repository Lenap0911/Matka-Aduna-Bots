"""
Gini Coefficient Forecasting Model
Forecasts Gini coefficient for years 111-115 using dynamic regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
print("Loading data...")
stats = pd.read_csv("data/data/statistics_matka_bots_year110.csv")
pop = pd.read_csv("data/data/population_matka_bots_year110.csv")

# Aggregate population data by year
print("Aggregating population data...")
pop_by_year = pop.groupby('year').agg({
    'income': ['sum', 'mean', 'median', 'std'],
    'sex': 'count',
    'profession': lambda x: (x != 'child').sum()  # employed count
}).reset_index()

pop_by_year.columns = ['year', 'total_income', 'avg_income', 'median_income', 
                        'income_std', 'total_population', 'employed_count']

# Calculate additional metrics
pop_by_year['employment_rate'] = pop_by_year['employed_count'] / pop_by_year['total_population']
pop_by_year['income_growth'] = pop_by_year['avg_income'].pct_change() * 100
pop_by_year['population_growth'] = pop_by_year['total_population'].pct_change() * 100
pop_by_year['gdp_growth'] = np.nan  # Will calculate after merge

# Merge with statistics data
df = pd.merge(stats, pop_by_year, on='year', how='left')

# Calculate GDP growth
df['gdp_growth'] = df['gdp'].pct_change() * 100

# Create lagged variables
df['gini_lag1'] = df['gini'].shift(1)
df['gdp_lag1'] = df['gdp'].shift(1)
df['avg_income_lag1'] = df['avg_income'].shift(1)
df['employment_rate_lag1'] = df['employment_rate'].shift(1)
df['income_std_lag1'] = df['income_std'].shift(1)

# Add GDP volatility (absolute change)
df['gdp_volatility'] = df['gdp_growth'].abs()

print(f"\nData loaded: {len(df)} years")
print(f"Gini data available for years: {df[df['gini'].notna()]['year'].min()} to {df[df['gini'].notna()]['year'].max()}")

# Prepare training data (drop rows with missing Gini or key predictors)
train_df = df[df['gini'].notna() & df['gini_lag1'].notna()].copy()

print(f"Training data: {len(train_df)} observations")

# Define features for the model
features = [
    'gdp',
    'income_growth',
    'employment_rate',
    'income_std',
    'gdp_growth',
    'gdp_volatility',
    'avg_income',
    'median_income',
    'gini_lag1'
]

# Remove any rows with NaN in features
train_clean = train_df.dropna(subset=features + ['gini'])
print(f"Clean training data: {len(train_clean)} observations")

# Prepare X and y
X_train = train_clean[features]
y_train = train_clean['gini']

# Fit the model
print("\n" + "="*60)
print("FITTING GINI FORECASTING MODEL")
print("="*60)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on training data
y_pred_train = model.predict(X_train)

# Model evaluation
r2 = r2_score(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)

print(f"\nModel Performance:")
print(f"  R² Score: {r2:.4f}")
print(f"  Mean Absolute Error: {mae:.4f}")

print(f"\nModel Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"  {feature:.<25} {coef:>10.6f}")
print(f"  {'Intercept':.<25} {model.intercept_:>10.6f}")

# Forecast next 5 years (years 111-115)
print("\n" + "="*60)
print("FORECASTING GINI FOR YEARS 111-115")
print("="*60)

# Get the last known values
last_year_data = df.iloc[-1].copy()
forecast_years = range(111, 116)
forecasts = []

# For forecasting, we need to predict the predictors first
# Use simple linear extrapolation for macro variables
recent_data = df.tail(10)  # Use last 10 years for trends

# Fit trends for predictors
def extrapolate_linear(series, n_ahead):
    """Simple linear extrapolation"""
    x = np.arange(len(series))
    y = series.values
    # Remove NaN
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return series.iloc[-1]  # Return last value if not enough data
    x_clean, y_clean = x[mask], y[mask]
    coef = np.polyfit(x_clean, y_clean, 1)
    return np.polyval(coef, len(series) + n_ahead - 1)

# Forecast each year recursively
for i, year in enumerate(forecast_years):
    print(f"\n--- Year {year} ---")
    
    # Extrapolate predictors
    if i == 0:
        # First forecast year - use actual last known values
        prev_gini = df[df['gini'].notna()]['gini'].iloc[-1]
        prev_year_full = df.iloc[-1]
        
        gdp_forecast = extrapolate_linear(recent_data['gdp'], i+1)
        avg_income_forecast = extrapolate_linear(recent_data['avg_income'], i+1)
        median_income_forecast = extrapolate_linear(recent_data['median_income'], i+1)
        income_std_forecast = extrapolate_linear(recent_data['income_std'], i+1)
        employment_rate_forecast = extrapolate_linear(recent_data['employment_rate'], i+1)
        
        # Growth rates based on forecast vs last known
        income_growth_forecast = ((avg_income_forecast - prev_year_full['avg_income']) / 
                                  prev_year_full['avg_income'] * 100)
        gdp_growth_forecast = ((gdp_forecast - prev_year_full['gdp']) / 
                               prev_year_full['gdp'] * 100)
    else:
        # Use previously forecasted values
        prev_forecast = forecasts[-1]
        prev_gini = prev_forecast['gini_forecast']
        
        gdp_forecast = extrapolate_linear(recent_data['gdp'], i+1)
        avg_income_forecast = extrapolate_linear(recent_data['avg_income'], i+1)
        median_income_forecast = extrapolate_linear(recent_data['median_income'], i+1)
        income_std_forecast = extrapolate_linear(recent_data['income_std'], i+1)
        employment_rate_forecast = extrapolate_linear(recent_data['employment_rate'], i+1)
        
        # Growth from previous forecast
        income_growth_forecast = ((avg_income_forecast - prev_forecast['avg_income']) / 
                                  prev_forecast['avg_income'] * 100)
        gdp_growth_forecast = ((gdp_forecast - prev_forecast['gdp']) / 
                               prev_forecast['gdp'] * 100)
    
    gdp_volatility_forecast = abs(gdp_growth_forecast)
    
    # Apply empirical rules about Gini behavior
    # Rule 1: GDP decline increases Gini
    if gdp_growth_forecast < -2:
        gini_shock = 0.02  # Add 2 points for significant decline
    elif gdp_growth_forecast < 0:
        gini_shock = 0.01  # Add 1 point for mild decline
    else:
        gini_shock = 0.0
    
    # Rule 2: High volatility increases Gini
    if gdp_volatility_forecast > 5:
        gini_shock += 0.01
    
    # Create feature vector
    X_forecast = np.array([[
        gdp_forecast,
        income_growth_forecast,
        employment_rate_forecast,
        income_std_forecast,
        gdp_growth_forecast,
        gdp_volatility_forecast,
        avg_income_forecast,
        median_income_forecast,
        prev_gini
    ]])
    
    # Predict Gini
    gini_forecast_raw = model.predict(X_forecast)[0]
    
    # Add empirical shock
    gini_forecast = gini_forecast_raw + gini_shock
    
    # Apply constraints: Gini should stay within 0.48-0.55 unless shock
    if abs(gini_shock) < 0.01:  # No major shock
        gini_forecast = np.clip(gini_forecast, 0.48, 0.55)
    else:  # Allow wider range during shocks
        gini_forecast = np.clip(gini_forecast, 0.45, 0.60)
    
    print(f"  GDP: ${gdp_forecast:,.0f} (growth: {gdp_growth_forecast:.2f}%)")
    print(f"  Avg Income: ${avg_income_forecast:,.0f} (growth: {income_growth_forecast:.2f}%)")
    print(f"  Employment Rate: {employment_rate_forecast:.2%}")
    print(f"  Income Std Dev: ${income_std_forecast:,.0f}")
    print(f"  Gini (raw): {gini_forecast_raw:.3f}")
    print(f"  Gini (adjusted): {gini_forecast:.3f}")
    
    forecasts.append({
        'year': year,
        'gdp': gdp_forecast,
        'gini_forecast': gini_forecast,
        'gini_raw': gini_forecast_raw,
        'avg_income': avg_income_forecast,
        'income_growth': income_growth_forecast,
        'employment_rate': employment_rate_forecast,
        'gdp_growth': gdp_growth_forecast,
        'income_std': income_std_forecast
    })

forecast_df = pd.DataFrame(forecasts)

# Save forecasts
forecast_df.to_csv('gini_forecasts_111_115.csv', index=False)
print(f"\n✓ Forecasts saved to 'gini_forecasts_111_115.csv'")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))

# Plot 1: GDP and Gini over time
ax1 = plt.subplot(3, 2, 1)
ax1_twin = ax1.twinx()

# Historical data
historical = df[df['gini'].notna()]
ax1.plot(historical['year'], historical['gdp'], 'b-', linewidth=2, label='Historical GDP')
ax1_twin.plot(historical['year'], historical['gini'], 'ro-', linewidth=2, label='Historical Gini')

# Forecasts
ax1.plot(forecast_df['year'], forecast_df['gdp'], 'b--', linewidth=2, alpha=0.7, label='Forecast GDP')
ax1_twin.plot(forecast_df['year'], forecast_df['gini_forecast'], 'rs--', linewidth=2, 
              alpha=0.7, label='Forecast Gini')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('GDP ($)', fontsize=12, color='b')
ax1_twin.set_ylabel('Gini Coefficient', fontsize=12, color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1.set_title('GDP and Gini Coefficient: Historical and Forecast', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Gini only with confidence band
ax2 = plt.subplot(3, 2, 2)
ax2.plot(historical['year'], historical['gini'], 'ro-', linewidth=2, label='Historical Gini')
ax2.plot(forecast_df['year'], forecast_df['gini_forecast'], 'rs--', linewidth=2, 
         alpha=0.7, label='Forecast Gini')
ax2.axhline(y=0.48, color='gray', linestyle=':', alpha=0.5, label='Expected Range')
ax2.axhline(y=0.55, color='gray', linestyle=':', alpha=0.5)
ax2.fill_between(forecast_df['year'], 0.48, 0.55, alpha=0.1, color='gray')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Gini Coefficient', fontsize=12)
ax2.set_title('Gini Coefficient Forecast with Expected Range', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Model Performance (Actual vs Predicted)
ax3 = plt.subplot(3, 2, 3)
ax3.scatter(y_train, y_pred_train, alpha=0.6, s=50)
ax3.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', linewidth=2, label='Perfect Fit')
ax3.set_xlabel('Actual Gini', fontsize=12)
ax3.set_ylabel('Predicted Gini', fontsize=12)
ax3.set_title(f'Model Fit (R² = {r2:.3f}, MAE = {mae:.3f})', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: GDP Growth vs Gini Change
ax4 = plt.subplot(3, 2, 4)
train_clean['gini_change'] = train_clean['gini'] - train_clean['gini_lag1']
colors = ['red' if x < 0 else 'green' for x in train_clean['gdp_growth']]
ax4.scatter(train_clean['gdp_growth'], train_clean['gini_change'], 
           c=colors, alpha=0.6, s=50)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax4.set_xlabel('GDP Growth (%)', fontsize=12)
ax4.set_ylabel('Gini Change', fontsize=12)
ax4.set_title('GDP Growth vs Gini Change (Red=Decline, Green=Growth)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Employment Rate vs Gini
ax5 = plt.subplot(3, 2, 5)
ax5.scatter(historical['employment_rate'], historical['gini'], alpha=0.6, s=50, c='purple')
ax5.set_xlabel('Employment Rate', fontsize=12)
ax5.set_ylabel('Gini Coefficient', fontsize=12)
ax5.set_title('Employment Rate vs Gini', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Feature Importance
ax6 = plt.subplot(3, 2, 6)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': np.abs(model.coef_)
}).sort_values('importance', ascending=True)

ax6.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
ax6.set_xlabel('Absolute Coefficient Value', fontsize=12)
ax6.set_title('Feature Importance (Absolute Coefficients)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('gini_gdp_forecast_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to 'gini_gdp_forecast_analysis.png'")

plt.show()

# Print summary table
print("\n" + "="*60)
print("FORECAST SUMMARY TABLE")
print("="*60)
print(forecast_df[['year', 'gdp', 'gini_forecast', 'gdp_growth', 
                    'income_growth', 'employment_rate']].to_string(index=False))
print("="*60)
