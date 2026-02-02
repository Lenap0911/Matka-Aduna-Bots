import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/Users/kaisahanni/Downloads/population_all_teams_year100 (3).csv')

# Calculate statistics by profession
profession_stats = df.groupby('profession').agg({
    'income': ['count', 'mean', 'std', 'min', 'max']
}).round(2)

# Flatten column names
profession_stats.columns = ['Count', 'Average Income', 'Std Dev (Variability)', 'Min Income', 'Max Income']
profession_stats = profession_stats.reset_index()

# Sort by count descending
profession_stats = profession_stats.sort_values('Count', ascending=False)

# Display the results
print("=" * 100)
print("PROFESSION ANALYSIS")
print("=" * 100)
print(profession_stats.to_string(index=False))
print("=" * 100)

# Summary statistics
print(f"\nTotal number of professions: {len(profession_stats)}")
print(f"Total records: {df.shape[0]:,}")
print(f"\nProfession with highest count: {profession_stats.iloc[0]['profession']} ({profession_stats.iloc[0]['Count']:,.0f})")
print(f"Profession with highest average income: {profession_stats.loc[profession_stats['Average Income'].idxmax(), 'profession']} ({profession_stats['Average Income'].max():,.2f})")
print(f"Profession with highest variability: {profession_stats.loc[profession_stats['Std Dev (Variability)'].idxmax(), 'profession']} (σ = {profession_stats['Std Dev (Variability)'].max():,.2f})")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Count by profession
ax1 = axes[0, 0]
profession_stats_plot = profession_stats.sort_values('Count', ascending=True)
ax1.barh(profession_stats_plot['profession'], profession_stats_plot['Count'])
ax1.set_xlabel('Count', fontsize=11)
ax1.set_title('Count of Individuals by Profession', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. Average income by profession
ax2 = axes[0, 1]
profession_stats_income = profession_stats[profession_stats['Average Income'] > 0].sort_values('Average Income', ascending=True)
ax2.barh(profession_stats_income['profession'], profession_stats_income['Average Income'], color='green')
ax2.set_xlabel('Average Income', fontsize=11)
ax2.set_title('Average Income by Profession', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Variability (Std Dev) by profession
ax3 = axes[1, 0]
profession_stats_std = profession_stats[profession_stats['Std Dev (Variability)'] > 0].sort_values('Std Dev (Variability)', ascending=True)
ax3.barh(profession_stats_std['profession'], profession_stats_std['Std Dev (Variability)'], color='orange')
ax3.set_xlabel('Standard Deviation of Income', fontsize=11)
ax3.set_title('Income Variability by Profession', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Average Income vs Variability scatter plot
ax4 = axes[1, 1]
profession_stats_scatter = profession_stats[profession_stats['Average Income'] > 0]
scatter = ax4.scatter(profession_stats_scatter['Average Income'], 
                     profession_stats_scatter['Std Dev (Variability)'],
                     s=profession_stats_scatter['Count']/100,
                     alpha=0.6,
                     c=range(len(profession_stats_scatter)),
                     cmap='viridis')
ax4.set_xlabel('Average Income', fontsize=11)
ax4.set_ylabel('Standard Deviation (Variability)', fontsize=11)
ax4.set_title('Average Income vs Variability (bubble size = count)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add profession labels to scatter plot
for idx, row in profession_stats_scatter.iterrows():
    ax4.annotate(row['profession'], 
                (row['Average Income'], row['Std Dev (Variability)']),
                fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('/Users/kaisahanni/Downloads/profession_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: /Users/kaisahanni/Downloads/profession_analysis.png")
plt.close()

# Export detailed stats to CSV
profession_stats.to_csv('/Users/kaisahanni/Downloads/profession_statistics.csv', index=False)
print(f"Detailed statistics saved to: /Users/kaisahanni/Downloads/profession_statistics.csv")
