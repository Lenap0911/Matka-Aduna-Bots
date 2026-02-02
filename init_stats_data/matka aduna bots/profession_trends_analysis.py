import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('/Users/kaisahanni/Downloads/population_all_teams_year100 (3).csv')

# Calculate profession counts by year
profession_count_by_year = df.groupby(['year', 'profession']).size().reset_index(name='count')

# Calculate average income by profession and year
profession_income_by_year = df.groupby(['year', 'profession'])['income'].mean().reset_index()
profession_income_by_year.columns = ['year', 'profession', 'avg_income']

# Get list of professions (excluding child for income plot since they have 0 income)
all_professions = df['profession'].unique()
income_professions = [p for p in all_professions if p != 'child']

# Create visualizations
fig = plt.figure(figsize=(18, 12))

# 1. Profession count over years (all professions)
ax1 = plt.subplot(2, 2, 1)
for profession in all_professions:
    data = profession_count_by_year[profession_count_by_year['profession'] == profession]
    ax1.plot(data['year'], data['count'], marker='o', markersize=2, linewidth=1.5, label=profession)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Profession Count Across Years', fontsize=12, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Average income by profession over years (excluding children)
ax2 = plt.subplot(2, 2, 2)
for profession in income_professions:
    data = profession_income_by_year[profession_income_by_year['profession'] == profession]
    ax2.plot(data['year'], data['avg_income'], marker='o', markersize=2, linewidth=1.5, label=profession)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Average Income', fontsize=11)
ax2.set_title('Average Income by Profession Across Years', fontsize=12, fontweight='bold')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Stacked area chart for profession counts
ax3 = plt.subplot(2, 2, 3)
pivot_count = profession_count_by_year.pivot(index='year', columns='profession', values='count').fillna(0)
ax3.stackplot(pivot_count.index, *[pivot_count[col] for col in pivot_count.columns], 
              labels=pivot_count.columns, alpha=0.8)
ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Total Count', fontsize=11)
ax3.set_title('Stacked Profession Distribution Over Years', fontsize=12, fontweight='bold')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Heatmap of profession counts
ax4 = plt.subplot(2, 2, 4)
# Sample every 10 years for readability
pivot_sample = pivot_count.iloc[::10, :]
sns.heatmap(pivot_sample.T, cmap='YlOrRd', annot=False, fmt='g', cbar_kws={'label': 'Count'}, ax=ax4)
ax4.set_xlabel('Year', fontsize=11)
ax4.set_ylabel('Profession', fontsize=11)
ax4.set_title('Profession Count Heatmap (Every 10 Years)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/kaisahanni/Downloads/profession_trends.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: /Users/kaisahanni/Downloads/profession_trends.png")
plt.close()

# Print summary statistics
print("\n" + "="*100)
print("PROFESSION COUNT AND INCOME ACROSS YEARS - SUMMARY")
print("="*100)

print("\n--- PROFESSION COUNT BY YEAR (First 5 and Last 5 Years) ---")
years = sorted(df['year'].unique())
for year in list(years[:5]) + list(years[-5:]):
    year_data = profession_count_by_year[profession_count_by_year['year'] == year]
    print(f"\nYear {year}:")
    for _, row in year_data.iterrows():
        print(f"  {row['profession']:20s}: {row['count']:5.0f}")

print("\n--- AVERAGE INCOME BY PROFESSION AND YEAR (First 5 and Last 5 Years) ---")
for year in list(years[:5]) + list(years[-5:]):
    year_data = profession_income_by_year[profession_income_by_year['year'] == year]
    print(f"\nYear {year}:")
    for _, row in year_data.iterrows():
        if row['profession'] != 'child':  # Skip children for income
            print(f"  {row['profession']:20s}: {row['avg_income']:10.2f}")

# Export detailed data to CSV
profession_count_by_year.to_csv('/Users/kaisahanni/Downloads/profession_count_by_year.csv', index=False)
profession_income_by_year.to_csv('/Users/kaisahanni/Downloads/profession_income_by_year.csv', index=False)

print("\n" + "="*100)
print("Detailed data exported to:")
print("  - /Users/kaisahanni/Downloads/profession_count_by_year.csv")
print("  - /Users/kaisahanni/Downloads/profession_income_by_year.csv")
print("="*100)

# Calculate growth rates
print("\n--- PROFESSION GROWTH OVER TIME ---")
start_year = years[0]
end_year = years[-1]
for profession in all_professions:
    start_count = profession_count_by_year[(profession_count_by_year['year'] == start_year) & 
                                          (profession_count_by_year['profession'] == profession)]['count'].values
    end_count = profession_count_by_year[(profession_count_by_year['year'] == end_year) & 
                                        (profession_count_by_year['profession'] == profession)]['count'].values
    if len(start_count) > 0 and len(end_count) > 0:
        start_val = start_count[0]
        end_val = end_count[0]
        growth = ((end_val - start_val) / start_val * 100) if start_val > 0 else 0
        print(f"{profession:20s}: {start_val:5.0f} → {end_val:5.0f} ({growth:+6.1f}%)")
