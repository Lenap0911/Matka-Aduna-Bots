import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/kaisahanni/Downloads/population_all_teams_year100 (3).csv')

# Filter for farmer and fisher only
farmer_fisher_df = df[df['profession'].isin(['farmer', 'fisher'])]

# Calculate average income by year for each profession
income_by_year = farmer_fisher_df.groupby(['year', 'profession'])['income'].mean().reset_index()

# Separate data for each profession
farmer_data = income_by_year[income_by_year['profession'] == 'farmer'].sort_values('year')
fisher_data = income_by_year[income_by_year['profession'] == 'fisher'].sort_values('year')

# Create enhanced plot with prominent markers for each year
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Plot 1: Both professions on same graph with clear year markers
ax1.plot(farmer_data['year'], farmer_data['income'], 
         marker='o', markersize=5, linewidth=2, label='Farmer', color='green', alpha=0.7)
ax1.plot(fisher_data['year'], fisher_data['income'], 
         marker='s', markersize=5, linewidth=2, label='Fisher', color='blue', alpha=0.7)

# Add every 10th year label
for i, year in enumerate(farmer_data['year']):
    if year % 10 == 0:
        ax1.axvline(x=year, color='gray', linestyle='--', alpha=0.2, linewidth=0.5)
        ax1.text(year, ax1.get_ylim()[1] * 0.95, str(int(year)), 
                ha='center', fontsize=8, alpha=0.5)

ax1.set_xlabel('Year', fontsize=13)
ax1.set_ylabel('Average Income', fontsize=13)
ax1.set_title('Farmer vs Fisher Average Income - All 101 Years (Year 0-100)', fontsize=15, fontweight='bold')
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Separate subplots for each profession
ax2_left = ax2
ax2_right = ax2.twinx()

# Farmer on left axis
line1 = ax2_left.plot(farmer_data['year'], farmer_data['income'], 
         marker='o', markersize=6, linewidth=2.5, label='Farmer', color='green', alpha=0.8)
ax2_left.set_xlabel('Year', fontsize=13)
ax2_left.set_ylabel('Farmer Income', fontsize=13, color='green')
ax2_left.tick_params(axis='y', labelcolor='green')

# Fisher on right axis
line2 = ax2_right.plot(fisher_data['year'], fisher_data['income'], 
         marker='s', markersize=6, linewidth=2.5, label='Fisher', color='blue', alpha=0.8)
ax2_right.set_ylabel('Fisher Income', fontsize=13, color='blue')
ax2_right.tick_params(axis='y', labelcolor='blue')

# Add grid
ax2_left.grid(True, alpha=0.3)

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2_left.legend(lines, labels, fontsize=12, loc='upper left')

ax2.set_title('Farmer and Fisher Income - Dual Axis View', fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/kaisahanni/Downloads/farmer_fisher_detailed_years.png', dpi=300, bbox_inches='tight')
print("Detailed graph saved to: /Users/kaisahanni/Downloads/farmer_fisher_detailed_years.png")
plt.close()

# Create a detailed table showing every single year
print("\n" + "="*100)
print("COMPLETE YEAR-BY-YEAR BREAKDOWN (101 YEARS: YEAR 0 TO YEAR 100)")
print("="*100)
print(f"{'Year':<6} | {'Farmer Income':>15} | {'Fisher Income':>15} | {'Difference':>15} | {'Fisher/Farmer':>12}")
print("-"*100)

for _, farmer_row in farmer_data.iterrows():
    year = int(farmer_row['year'])
    farmer_income = farmer_row['income']
    fisher_row = fisher_data[fisher_data['year'] == year]
    fisher_income = fisher_row['income'].values[0] if len(fisher_row) > 0 else 0
    diff = fisher_income - farmer_income
    ratio = fisher_income / farmer_income if farmer_income != 0 else 0
    
    print(f"{year:<6} | {farmer_income:>15.2f} | {fisher_income:>15.2f} | {diff:>+15.2f} | {ratio:>12.2f}x")

print("="*100)

# Statistical summary for each decade
print("\n" + "="*100)
print("DECADE-BY-DECADE SUMMARY")
print("="*100)

for decade_start in range(0, 101, 10):
    decade_end = min(decade_start + 9, 100)
    
    farmer_decade = farmer_data[(farmer_data['year'] >= decade_start) & (farmer_data['year'] <= decade_end)]
    fisher_decade = fisher_data[(fisher_data['year'] >= decade_start) & (fisher_data['year'] <= decade_end)]
    
    print(f"\nYears {decade_start}-{decade_end}:")
    print(f"  Farmer - Avg: {farmer_decade['income'].mean():>10.2f}, Min: {farmer_decade['income'].min():>10.2f}, Max: {farmer_decade['income'].max():>10.2f}")
    print(f"  Fisher - Avg: {fisher_decade['income'].mean():>10.2f}, Min: {fisher_decade['income'].min():>10.2f}, Max: {fisher_decade['income'].max():>10.2f}")
    print(f"  Gap    - Avg: {fisher_decade['income'].mean() - farmer_decade['income'].mean():>+10.2f}")

print("\n" + "="*100)

# Create detailed CSV with additional columns
detailed_df = pd.DataFrame({
    'Year': farmer_data['year'].values,
    'Farmer_Income': farmer_data['income'].values,
    'Fisher_Income': fisher_data['income'].values,
    'Income_Difference': fisher_data['income'].values - farmer_data['income'].values,
    'Fisher_to_Farmer_Ratio': fisher_data['income'].values / farmer_data['income'].values,
    'Farmer_Change_from_Previous': [0] + list(np.diff(farmer_data['income'].values)),
    'Fisher_Change_from_Previous': [0] + list(np.diff(fisher_data['income'].values))
})

detailed_df.to_csv('/Users/kaisahanni/Downloads/farmer_fisher_all_101_years.csv', index=False)
print("\nComplete data for all 101 years exported to: /Users/kaisahanni/Downloads/farmer_fisher_all_101_years.csv")
print("This file includes: Income for each year, differences, ratios, and year-over-year changes")
