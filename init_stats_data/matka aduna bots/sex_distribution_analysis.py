import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('/Users/kaisahanni/Downloads/population_all_teams_year100 (3).csv')

# Count by sex and year
sex_count_by_year = df.groupby(['year', 'sex']).size().reset_index(name='count')

# Pivot for easier analysis
sex_pivot = sex_count_by_year.pivot(index='year', columns='sex', values='count').fillna(0)

# Calculate totals
sex_pivot['Total'] = sex_pivot.sum(axis=1)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Line graph for all sexes
ax1 = axes[0, 0]
for sex in sex_pivot.columns[:-1]:  # Exclude 'Total'
    ax1.plot(sex_pivot.index, sex_pivot[sex], marker='o', markersize=4, 
             linewidth=2, label=f'{sex}', alpha=0.8)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Population Count by Sex Across Years', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Stacked area chart
ax2 = axes[0, 1]
ax2.stackplot(sex_pivot.index, 
              *[sex_pivot[col] for col in sex_pivot.columns[:-1]], 
              labels=sex_pivot.columns[:-1], alpha=0.8)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Total Count', fontsize=12)
ax2.set_title('Stacked Population by Sex', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: Gender ratio over time (if M and F exist)
ax3 = axes[1, 0]
if 'M' in sex_pivot.columns and 'F' in sex_pivot.columns:
    ratio = sex_pivot['F'] / sex_pivot['M']
    ax3.plot(sex_pivot.index, ratio, marker='o', markersize=4, 
             linewidth=2, color='purple', alpha=0.8)
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Equal ratio')
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Female to Male Ratio', fontsize=12)
    ax3.set_title('Female to Male Ratio Over Time', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)

# Plot 4: Total population over time
ax4 = axes[1, 1]
ax4.plot(sex_pivot.index, sex_pivot['Total'], marker='o', markersize=4, 
         linewidth=2, color='black', alpha=0.8)
ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('Total Population', fontsize=12)
ax4.set_title('Total Population Across Years', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/kaisahanni/Downloads/sex_distribution_by_year.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: /Users/kaisahanni/Downloads/sex_distribution_by_year.png")
plt.close()

# Print complete year-by-year breakdown
print("\n" + "="*100)
print("MALE AND FEMALE COUNTS PER YEAR (ALL 101 YEARS)")
print("="*100)

# Get unique sexes
sexes = sorted(df['sex'].unique())
header = f"{'Year':<6}"
for sex in sexes:
    header += f" | {sex:>8}"
header += f" | {'Total':>8}"
print(header)
print("-"*100)

for year in sorted(sex_pivot.index):
    line = f"{int(year):<6}"
    for sex in sexes:
        count = sex_pivot.loc[year, sex] if sex in sex_pivot.columns else 0
        line += f" | {int(count):>8}"
    line += f" | {int(sex_pivot.loc[year, 'Total']):>8}"
    print(line)

print("="*100)

# Summary statistics
print("\n" + "="*100)
print("SUMMARY STATISTICS")
print("="*100)

for sex in sexes:
    if sex in sex_pivot.columns:
        print(f"\n{sex}:")
        print(f"  Average per year: {sex_pivot[sex].mean():.2f}")
        print(f"  Minimum: {sex_pivot[sex].min():.0f} (Year {sex_pivot[sex].idxmin()})")
        print(f"  Maximum: {sex_pivot[sex].max():.0f} (Year {sex_pivot[sex].idxmax()})")
        print(f"  Starting count (Year 0): {sex_pivot.loc[0, sex]:.0f}")
        print(f"  Ending count (Year 100): {sex_pivot.loc[100, sex]:.0f}")
        change = sex_pivot.loc[100, sex] - sex_pivot.loc[0, sex]
        pct_change = (change / sex_pivot.loc[0, sex] * 100) if sex_pivot.loc[0, sex] > 0 else 0
        print(f"  Change: {change:+.0f} ({pct_change:+.1f}%)")

print(f"\nTOTAL POPULATION:")
print(f"  Average per year: {sex_pivot['Total'].mean():.2f}")
print(f"  Minimum: {sex_pivot['Total'].min():.0f} (Year {sex_pivot['Total'].idxmin()})")
print(f"  Maximum: {sex_pivot['Total'].max():.0f} (Year {sex_pivot['Total'].idxmax()})")
print(f"  Starting total (Year 0): {sex_pivot.loc[0, 'Total']:.0f}")
print(f"  Ending total (Year 100): {sex_pivot.loc[100, 'Total']:.0f}")
change = sex_pivot.loc[100, 'Total'] - sex_pivot.loc[0, 'Total']
pct_change = (change / sex_pivot.loc[0, 'Total'] * 100) if sex_pivot.loc[0, 'Total'] > 0 else 0
print(f"  Change: {change:+.0f} ({pct_change:+.1f}%)")

# Gender ratio statistics (if applicable)
if 'M' in sex_pivot.columns and 'F' in sex_pivot.columns:
    print(f"\nGENDER RATIO (Female/Male):")
    ratio = sex_pivot['F'] / sex_pivot['M']
    print(f"  Average ratio: {ratio.mean():.3f}")
    print(f"  Minimum ratio: {ratio.min():.3f} (Year {ratio.idxmin()})")
    print(f"  Maximum ratio: {ratio.max():.3f} (Year {ratio.idxmax()})")

print("\n" + "="*100)

# Export to CSV
sex_pivot_export = sex_pivot.reset_index()
sex_pivot_export.to_csv('/Users/kaisahanni/Downloads/sex_counts_by_year.csv', index=False)
print("\nDetailed data exported to: /Users/kaisahanni/Downloads/sex_counts_by_year.csv")

# Decade summary
print("\n" + "="*100)
print("DECADE-BY-DECADE SUMMARY")
print("="*100)

for decade_start in range(0, 101, 10):
    decade_end = min(decade_start + 9, 100)
    decade_data = sex_pivot[(sex_pivot.index >= decade_start) & (sex_pivot.index <= decade_end)]
    
    print(f"\nYears {decade_start}-{decade_end}:")
    for sex in sexes:
        if sex in sex_pivot.columns:
            print(f"  {sex}: Avg = {decade_data[sex].mean():>6.1f}, Min = {decade_data[sex].min():>6.0f}, Max = {decade_data[sex].max():>6.0f}")
    print(f"  Total: Avg = {decade_data['Total'].mean():>6.1f}, Min = {decade_data['Total'].min():>6.0f}, Max = {decade_data['Total'].max():>6.0f}")

print("\n" + "="*100)
