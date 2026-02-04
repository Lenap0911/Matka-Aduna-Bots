import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
df = pd.read_csv('data/data/profession_income_by_year .csv')

# Filter for years 91-95
data = df[(df['year'] >= 91) & (df['year'] <= 95)]

# Get unique professions (excluding child as they have 0 income)
professions = data[data['profession'] != 'child']['profession'].unique()

# Create the visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Income by Profession (Years 91-95)', fontsize=16, fontweight='bold')

# Plot 1: Line plot for all professions
ax1 = axes[0, 0]
for profession in professions:
    prof_data = data[data['profession'] == profession]
    ax1.plot(prof_data['year'], prof_data['avg_income'], marker='o', label=profession, linewidth=2, markersize=8)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Average Income', fontsize=11)
ax1.set_title('Income Trends by Profession', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(91, 96))

# Plot 2: Stacked area chart
ax2 = axes[0, 1]
pivot_data = data[data['profession'] != 'child'].pivot(index='year', columns='profession', values='avg_income')
pivot_data.plot(kind='area', stacked=False, ax=ax2, alpha=0.6)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Average Income', fontsize=11)
ax2.set_title('Income Distribution Over Time', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(91, 96))

# Plot 3: Bar chart comparing first year (91) vs last year (95)
ax3 = axes[1, 0]
year_91 = data[data['year'] == 91].set_index('profession')['avg_income']
year_95 = data[data['year'] == 95].set_index('profession')['avg_income']

x = np.arange(len(professions))
width = 0.35

bars1 = ax3.bar(x - width/2, [year_91.get(p, 0) for p in professions], width, 
                label='Year 91', alpha=0.8, color='#3A86FF')
bars2 = ax3.bar(x + width/2, [year_95.get(p, 0) for p in professions], width,
                label='Year 95', alpha=0.8, color='#FB5607')

ax3.set_xlabel('Profession', fontsize=11)
ax3.set_ylabel('Average Income', fontsize=11)
ax3.set_title('Income Comparison: Year 91 vs Year 95', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(professions, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Heatmap
ax4 = axes[1, 1]
pivot_data = data[data['profession'] != 'child'].pivot(index='profession', columns='year', values='avg_income')
im = ax4.imshow(pivot_data.values, aspect='auto', cmap='YlOrRd')
ax4.set_xticks(range(len(pivot_data.columns)))
ax4.set_xticklabels(pivot_data.columns)
ax4.set_yticks(range(len(pivot_data.index)))
ax4.set_yticklabels(pivot_data.index)
ax4.set_xlabel('Year', fontsize=11)
ax4.set_ylabel('Profession', fontsize=11)
ax4.set_title('Income Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax4, label='Average Income')

plt.tight_layout()
plt.savefig('income_by_profession_91_95.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("INCOME BY PROFESSION SUMMARY (YEARS 91-95)")
print("="*70)

for profession in sorted(professions):
    prof_data = data[data['profession'] == profession]
    avg_income = prof_data['avg_income'].mean()
    min_income = prof_data['avg_income'].min()
    max_income = prof_data['avg_income'].max()
    
    year_91_income = prof_data[prof_data['year'] == 91]['avg_income'].values
    year_95_income = prof_data[prof_data['year'] == 95]['avg_income'].values
    
    if len(year_91_income) > 0 and len(year_95_income) > 0:
        change = year_95_income[0] - year_91_income[0]
        percent_change = (change / year_91_income[0]) * 100 if year_91_income[0] > 0 else 0
        
        print(f"\n{profession.upper()}:")
        print(f"  Average (91-95): ${avg_income:.2f}")
        print(f"  Range: ${min_income:.2f} - ${max_income:.2f}")
        print(f"  Change (91→95): ${change:.2f} ({percent_change:+.2f}%)")

print("\n" + "="*70)

# Overall comparison table for year 95
print("\nINCOME RANKING (YEAR 95):")
print("-"*70)
year_95_data = data[data['year'] == 95].sort_values('avg_income', ascending=False)
for idx, row in year_95_data.iterrows():
    if row['profession'] != 'child':
        print(f"{row['profession']:20s}: ${row['avg_income']:,.2f}")

print("\n" + "="*70)
print("\nDETAILED YEAR-BY-YEAR BREAKDOWN:")
print("="*70)
for year in range(91, 96):
    print(f"\nYEAR {year}:")
    print("-"*70)
    year_data = data[data['year'] == year].sort_values('avg_income', ascending=False)
    for idx, row in year_data.iterrows():
        if row['profession'] != 'child':
            print(f"  {row['profession']:20s}: ${row['avg_income']:,.2f}")
print("="*70)
