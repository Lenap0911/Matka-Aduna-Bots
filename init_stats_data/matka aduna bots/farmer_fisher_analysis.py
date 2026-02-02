import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('/Users/kaisahanni/Downloads/population_all_teams_year100 (3).csv')

# Filter for farmer and fisher only
farmer_fisher_df = df[df['profession'].isin(['farmer', 'fisher'])]

# Calculate average income by year for each profession
income_by_year = farmer_fisher_df.groupby(['year', 'profession'])['income'].mean().reset_index()

# Separate data for each profession
farmer_data = income_by_year[income_by_year['profession'] == 'farmer']
fisher_data = income_by_year[income_by_year['profession'] == 'fisher']

# Create the plot
plt.figure(figsize=(14, 8))
plt.plot(farmer_data['year'], farmer_data['income'], 
         marker='o', markersize=4, linewidth=2, label='Farmer', color='green', alpha=0.8)
plt.plot(fisher_data['year'], fisher_data['income'], 
         marker='s', markersize=4, linewidth=2, label='Fisher', color='blue', alpha=0.8)

plt.xlabel('Year', fontsize=13)
plt.ylabel('Average Income', fontsize=13)
plt.title('Farmer vs Fisher Average Income Across Years', fontsize=15, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('/Users/kaisahanni/Downloads/farmer_fisher_income.png', dpi=300, bbox_inches='tight')
print("Graph saved to: /Users/kaisahanni/Downloads/farmer_fisher_income.png")
plt.close()

# Print detailed statistics
print("\n" + "="*80)
print("FARMER AND FISHER AVERAGE INCOME BY YEAR")
print("="*80)
print(f"{'Year':<8} {'Farmer Income':<20} {'Fisher Income':<20} {'Difference':<15}")
print("-"*80)

for year in sorted(df['year'].unique()):
    farmer_income = farmer_data[farmer_data['year'] == year]['income'].values
    fisher_income = fisher_data[fisher_data['year'] == year]['income'].values
    
    farmer_val = farmer_income[0] if len(farmer_income) > 0 else 0
    fisher_val = fisher_income[0] if len(fisher_income) > 0 else 0
    diff = fisher_val - farmer_val
    
    print(f"{year:<8} {farmer_val:<20.2f} {fisher_val:<20.2f} {diff:<+15.2f}")

print("="*80)

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nFARMER:")
print(f"  Average income (all years): {farmer_data['income'].mean():.2f}")
print(f"  Minimum income: {farmer_data['income'].min():.2f} (Year {farmer_data.loc[farmer_data['income'].idxmin(), 'year']:.0f})")
print(f"  Maximum income: {farmer_data['income'].max():.2f} (Year {farmer_data.loc[farmer_data['income'].idxmax(), 'year']:.0f})")
print(f"  Standard deviation: {farmer_data['income'].std():.2f}")
print(f"  Starting income (Year 0): {farmer_data[farmer_data['year'] == 0]['income'].values[0]:.2f}")
print(f"  Ending income (Year 100): {farmer_data[farmer_data['year'] == 100]['income'].values[0]:.2f}")
change = ((farmer_data[farmer_data['year'] == 100]['income'].values[0] - 
           farmer_data[farmer_data['year'] == 0]['income'].values[0]) / 
          farmer_data[farmer_data['year'] == 0]['income'].values[0] * 100)
print(f"  Overall change: {change:+.1f}%")

print("\nFISHER:")
print(f"  Average income (all years): {fisher_data['income'].mean():.2f}")
print(f"  Minimum income: {fisher_data['income'].min():.2f} (Year {fisher_data.loc[fisher_data['income'].idxmin(), 'year']:.0f})")
print(f"  Maximum income: {fisher_data['income'].max():.2f} (Year {fisher_data.loc[fisher_data['income'].idxmax(), 'year']:.0f})")
print(f"  Standard deviation: {fisher_data['income'].std():.2f}")
print(f"  Starting income (Year 0): {fisher_data[fisher_data['year'] == 0]['income'].values[0]:.2f}")
print(f"  Ending income (Year 100): {fisher_data[fisher_data['year'] == 100]['income'].values[0]:.2f}")
change = ((fisher_data[fisher_data['year'] == 100]['income'].values[0] - 
           fisher_data[fisher_data['year'] == 0]['income'].values[0]) / 
          fisher_data[fisher_data['year'] == 0]['income'].values[0] * 100)
print(f"  Overall change: {change:+.1f}%")

print("\n" + "="*80)

# Export to CSV
combined_data = farmer_data.merge(fisher_data, on='year', suffixes=('_farmer', '_fisher'))
combined_data['difference'] = combined_data['income_fisher'] - combined_data['income_farmer']
combined_data = combined_data[['year', 'income_farmer', 'income_fisher', 'difference']]
combined_data.columns = ['Year', 'Farmer Income', 'Fisher Income', 'Difference (Fisher - Farmer)']
combined_data.to_csv('/Users/kaisahanni/Downloads/farmer_fisher_comparison.csv', index=False)
print("\nDetailed data exported to: /Users/kaisahanni/Downloads/farmer_fisher_comparison.csv")
