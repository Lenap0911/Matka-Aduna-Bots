import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Read the population data
df = pd.read_csv('population_matka_bots_year105.csv')

# Filter for years 96-105
df_filtered = df[df['year'] >= 96].copy()

# Calculate average happiness by year and profession
happiness_by_year_profession = df_filtered.groupby(['year', 'profession'])['happiness'].mean().reset_index()

# Create a figure with multiple subplots
fig = plt.figure(figsize=(18, 12))

# 1. Line plot - Happiness trends over time by profession
ax1 = plt.subplot(2, 2, 1)
professions = happiness_by_year_profession['profession'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(professions)))

for i, profession in enumerate(sorted(professions)):
    data = happiness_by_year_profession[happiness_by_year_profession['profession'] == profession]
    ax1.plot(data['year'], data['happiness'], marker='o', label=profession, 
             linewidth=2, markersize=6, color=colors[i])

ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Happiness', fontsize=12, fontweight='bold')
ax1.set_title('Happiness Trends by Profession (Years 96-105)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(96, 106))

# 2. Bar chart - Average happiness by profession
ax2 = plt.subplot(2, 2, 2)
avg_happiness = df_filtered.groupby('profession')['happiness'].mean().sort_values(ascending=False)
bars = ax2.barh(range(len(avg_happiness)), avg_happiness.values, color=plt.cm.viridis(np.linspace(0.2, 0.9, len(avg_happiness))))
ax2.set_yticks(range(len(avg_happiness)))
ax2.set_yticklabels(avg_happiness.index)
ax2.set_xlabel('Average Happiness', fontsize=12, fontweight='bold')
ax2.set_title('Average Happiness by Profession (Years 96-105)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, avg_happiness.values)):
    ax2.text(value + 0.5, i, f'{value:.2f}', va='center', fontsize=9)

# 3. Heatmap - Happiness by profession and year
ax3 = plt.subplot(2, 2, 3)
pivot_data = happiness_by_year_profession.pivot(index='profession', columns='year', values='happiness')
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=100, 
            cbar_kws={'label': 'Happiness'}, ax=ax3, linewidths=0.5)
ax3.set_title('Happiness Heatmap: Profession vs Year', fontsize=14, fontweight='bold')
ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
ax3.set_ylabel('Profession', fontsize=12, fontweight='bold')

# 4. Box plot - Happiness distribution by profession
ax4 = plt.subplot(2, 2, 4)
profession_order = df_filtered.groupby('profession')['happiness'].mean().sort_values(ascending=False).index
box_data = [df_filtered[df_filtered['profession'] == prof]['happiness'].values for prof in profession_order]
bp = ax4.boxplot(box_data, labels=profession_order, vert=False, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2))
ax4.set_xlabel('Happiness', fontsize=12, fontweight='bold')
ax4.set_title('Happiness Distribution by Profession (Years 96-105)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('happiness_visualizations_years_96_105.png', dpi=300, bbox_inches='tight')
print("Visualization saved as: happiness_visualizations_years_96_105.png")
plt.show()

# Additional: Individual profession trends
fig2, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, profession in enumerate(sorted(professions)):
    data = happiness_by_year_profession[happiness_by_year_profession['profession'] == profession]
    axes[i].plot(data['year'], data['happiness'], marker='o', linewidth=2.5, 
                 markersize=8, color=colors[i])
    axes[i].set_title(f'{profession.upper()}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Year', fontsize=10)
    axes[i].set_ylabel('Happiness', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xticks(range(96, 106))
    axes[i].set_xticklabels(range(96, 106), rotation=45)
    
    # Add trend line
    z = np.polyfit(data['year'], data['happiness'], 1)
    p = np.poly1d(z)
    axes[i].plot(data['year'], p(data['year']), "--", alpha=0.5, color='red', linewidth=1.5)

plt.tight_layout()
plt.savefig('happiness_individual_professions_years_96_105.png', dpi=300, bbox_inches='tight')
print("Individual profession trends saved as: happiness_individual_professions_years_96_105.png")
plt.show()

print("\nSummary Statistics:")
print("=" * 60)
for profession in sorted(professions):
    prof_data = df_filtered[df_filtered['profession'] == profession]['happiness']
    print(f"\n{profession.upper()}:")
    print(f"  Mean: {prof_data.mean():.2f}")
    print(f"  Median: {prof_data.median():.2f}")
    print(f"  Std Dev: {prof_data.std():.2f}")
    print(f"  Min: {prof_data.min():.2f}")
    print(f"  Max: {prof_data.max():.2f}")
