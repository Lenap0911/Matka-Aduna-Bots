# GDP Prediction Model - User Guide

## Overview
This simple predictive model forecasts GDP for years 101-105 based on 100 years of historical data. The model uses multiple forecasting methods and provides an ensemble prediction.

## Quick Start

### Run the model:
```bash
python gdp_prediction_model.py
```

### Expected Output:
- `gdp_predictions.csv` - Predictions for years 101-105
- `gdp_predictions.png` - Visualization of predictions

## Model Components

### 1. **Linear Regression Model**
- Simple trend line through historical data
- Best for: Stable, linear growth patterns
- Formula: `GDP = slope * year + intercept`

### 2. **Polynomial Regression Model**
- Captures non-linear trends (curves)
- Degree: 2 (quadratic) - can be adjusted
- Best for: Data with acceleration/deceleration patterns

### 3. **Moving Average Model**
- Based on average of last N years
- Window: 10 years - can be adjusted
- Best for: Short-term predictions, stable conditions

### 4. **Ensemble Prediction** ⭐ (RECOMMENDED)
- Average of all models
- More robust and balanced
- Reduces individual model bias

## How to Expand This Model

### Easy Expansions:

1. **Adjust Model Parameters:**
```python
model.train_polynomial_model(degree=3)  # Try different degrees
model.train_moving_average(window=20)   # Try different windows
```

2. **Add More Years:**
```python
predictions = model.predict_future(years_ahead=10)  # Predict 10 years
```

3. **Include Additional Features:**
```python
# You can modify the model to include:
- Population growth rate
- Average income per person
- Employment rates
- Profession distribution
```

### Advanced Expansions:

1. **Add ARIMA/Time Series Models:**
```python
from statsmodels.tsa.arima.model import ARIMA
# Add seasonal patterns, autocorrelation
```

2. **Add Machine Learning Models:**
```python
from sklearn.ensemble import RandomForestRegressor
# More complex pattern recognition
```

3. **Include Economic Indicators:**
```python
# Factor in:
- Birth/death rates
- Weather patterns (from your weather_forecast data)
- Profession changes over time
```

4. **Add Confidence Intervals:**
```python
# Show prediction ranges (e.g., 80%, 95% confidence)
```

5. **Scenario Analysis:**
```python
# Create optimistic/pessimistic scenarios
# What-if analysis for different conditions
```

## File Structure

```
gdp_prediction_model.py          # Main model file
gdp_predictions.csv              # Output: predictions
gdp_predictions.png              # Output: visualization
data/data/statistics_all_teams_year100.csv  # Input: historical data
```

## Understanding the Output

### Sample Predictions Table:
```
year  linear_prediction  polynomial_prediction  moving_average  ensemble_prediction
101   1,150,000         1,180,000              1,100,000       1,143,333
102   1,152,000         1,185,000              1,100,000       1,145,667
...
```

- **linear_prediction**: Simple trend continuation
- **polynomial_prediction**: Curved trend projection
- **moving_average**: Recent average
- **ensemble_prediction**: Average of all (USE THIS!)

## Tips for Improvement

1. **Validate Your Model:**
   - Test on years 90-100 using data from 0-89
   - Compare predictions vs actual values
   - Calculate prediction error (RMSE, MAE)

2. **Consider External Factors:**
   - Your data has profession, population, weather data
   - These could improve predictions significantly

3. **Monitor Model Performance:**
   - Track prediction accuracy over time
   - Adjust model weights if one performs better

4. **Regular Updates:**
   - Retrain model with new data each year
   - Update moving average window

## Next Steps

1. ✅ Run the basic model
2. 📊 Review predictions and visualization
3. 🔍 Add validation (test on historical data)
4. 🎯 Incorporate additional features (population, professions, etc.)
5. 🚀 Build more sophisticated models

## Need Help?

The code is well-commented and organized into a class structure. Each method has a clear purpose:
- `train_*()` methods: Train different models
- `predict_future()`: Generate predictions
- `visualize_predictions()`: Create charts
- `export_predictions()`: Save results

Modify any part to suit your needs!
