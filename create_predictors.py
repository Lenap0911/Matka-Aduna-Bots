"""
Create Predictors Dataset
==========================
Combines all available data into a comprehensive predictors CSV file
for multi-variable GDP prediction.
"""

import pandas as pd
import numpy as np

def create_predictors_dataset():
    """Create a comprehensive predictors dataset from all available data"""
    
    print("Loading data files...")
    
    # 1. Load main population data
    pop_df = pd.read_csv('data/data/population_all_teams_year100.csv')
    
    # 2. Load profession counts
    prof_count = pd.read_csv('data/data/profession_count_by_year.csv')
    
    # 3. Load statistics (has GDP)
    stats_df = pd.read_csv('data/data/statistics_all_teams_year100.csv')
    
    print("Creating aggregate features by year...")
    
    # Create features from population data
    yearly_features = []
    
    for year in range(101):  # Years 0-100
        year_data = pop_df[pop_df['year'] == year]
        
        features = {
            'year': year,
            
            # Population metrics
            'total_population': len(year_data),
            'male_count': len(year_data[year_data['sex'] == 'M']),
            'female_count': len(year_data[year_data['sex'] == 'F']),
            'male_ratio': len(year_data[year_data['sex'] == 'M']) / len(year_data) if len(year_data) > 0 else 0,
            
            # Income metrics
            'total_income': year_data['income'].sum(),
            'avg_income': year_data['income'].mean(),
            'median_income': year_data['income'].median(),
            'income_std': year_data['income'].std(),
            'max_income': year_data['income'].max(),
            'min_income': year_data['income'].min(),
            
            # Working population
            'employed_count': len(year_data[year_data['income'] > 0]),
            'unemployed_count': len(year_data[year_data['income'] == 0]),
            'employment_rate': len(year_data[year_data['income'] > 0]) / len(year_data) if len(year_data) > 0 else 0,
        }
        
        # Profession-specific counts
        year_prof = prof_count[prof_count['year'] == year]
        for _, row in year_prof.iterrows():
            prof_name = row['profession'].replace(' ', '_')
            features[f'{prof_name}_count'] = row['count']
        
        # Profession-specific income averages
        for profession in year_data['profession'].unique():
            prof_data = year_data[year_data['profession'] == profession]
            prof_name = profession.replace(' ', '_')
            features[f'{prof_name}_avg_income'] = prof_data['income'].mean()
            features[f'{prof_name}_total_income'] = prof_data['income'].sum()
        
        yearly_features.append(features)
    
    # Create DataFrame
    predictors_df = pd.DataFrame(yearly_features)
    
    # Fill NaN values with 0 (for professions that don't exist in certain years)
    predictors_df = predictors_df.fillna(0)
    
    # Add GDP as target variable
    predictors_df = predictors_df.merge(stats_df, on='year', how='left')
    
    # Create lagged features (previous year values)
    lag_columns = ['total_population', 'avg_income', 'employment_rate', 'total_income']
    for col in lag_columns:
        if col in predictors_df.columns:
            predictors_df[f'{col}_lag1'] = predictors_df[col].shift(1)
            predictors_df[f'{col}_lag2'] = predictors_df[col].shift(2)
    
    # Create growth rate features
    if 'total_population' in predictors_df.columns:
        predictors_df['population_growth'] = predictors_df['total_population'].pct_change() * 100
    if 'avg_income' in predictors_df.columns:
        predictors_df['income_growth'] = predictors_df['avg_income'].pct_change() * 100
    
    # Fill NaN from lagged features with 0
    predictors_df = predictors_df.fillna(0)
    
    # Save to CSV
    output_path = 'predictors.csv'
    predictors_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Predictors dataset created: {output_path}")
    print(f"   Shape: {predictors_df.shape}")
    print(f"   Years: {predictors_df['year'].min()} to {predictors_df['year'].max()}")
    print(f"   Features: {len(predictors_df.columns)} columns")
    print(f"\nColumn names:")
    for i, col in enumerate(predictors_df.columns, 1):
        print(f"   {i}. {col}")
    
    return predictors_df

if __name__ == "__main__":
    df = create_predictors_dataset()
    print("\n✅ Preview of first 3 rows:")
    print(df.head(3))
