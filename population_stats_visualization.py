import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load data
pop_data = pd.read_csv('data/data/population_matka_bots_year100.csv')
stats_data = pd.read_csv('data/data/statistics_all_teams_year100.csv')

# Use all years
pop_first_10 = pop_data
stats_first_10 = stats_data

# Create figure with subplots
fig = plt.figure(figsize=(18, 14))

# 1. Average Net Worth by Profession Over Time
ax1 = plt.subplot(3, 3, 1)
net_worth_by_prof = pop_first_10.groupby(['year', 'profession'])['net_worth'].mean().reset_index()
professions = net_worth_by_prof['profession'].unique()
for prof in professions:
    prof_data = net_worth_by_prof[net_worth_by_prof['profession'] == prof]
    ax1.plot(prof_data['year'], prof_data['net_worth'], marker='o', label=prof, linewidth=2)
ax1.set_xlabel('Year', fontsize=10)
ax1.set_ylabel('Average Net Worth', fontsize=10)
ax1.set_title('Average Net Worth by Profession (All Years)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)

# 2. Average Income by Profession Over Time
ax2 = plt.subplot(3, 3, 2)
income_by_prof = pop_first_10.groupby(['year', 'profession'])['income'].mean().reset_index()
for prof in professions:
    prof_data = income_by_prof[income_by_prof['profession'] == prof]
    ax2.plot(prof_data['year'], prof_data['income'], marker='o', label=prof, linewidth=2)
ax2.set_xlabel('Year', fontsize=10)
ax2.set_ylabel('Average Income', fontsize=10)
ax2.set_title('Average Income by Profession (All Years)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='best')
ax2.grid(True, alpha=0.3)

# 3. GDP Over Time
ax3 = plt.subplot(3, 3, 3)
ax3.plot(stats_first_10['year'], stats_first_10['gdp'], marker='o', color='green', linewidth=2, markersize=8)
ax3.fill_between(stats_first_10['year'], stats_first_10['gdp'], alpha=0.3, color='green')
ax3.set_xlabel('Year', fontsize=10)
ax3.set_ylabel('GDP', fontsize=10)
ax3.set_title('GDP Over Time (All Years)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.ticklabel_format(style='plain', axis='y')

# 4. Population by Profession (Total Count)
ax4 = plt.subplot(3, 3, 4)
prof_counts = pop_first_10['profession'].value_counts().sort_values(ascending=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(prof_counts)))
ax4.barh(prof_counts.index, prof_counts.values, color=colors)
ax4.set_xlabel('Total Count', fontsize=10)
ax4.set_title('Population by Profession (All Years)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 5. Average Net Worth by Profession (Bar Chart)
ax5 = plt.subplot(3, 3, 5)
avg_net_worth = pop_first_10.groupby('profession')['net_worth'].mean().sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(avg_net_worth)))
ax5.barh(avg_net_worth.index, avg_net_worth.values, color=colors)
ax5.set_xlabel('Average Net Worth', fontsize=10)
ax5.set_title('Average Net Worth by Profession (All Years)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# 6. Average Income by Profession (Bar Chart)
ax6 = plt.subplot(3, 3, 6)
avg_income = pop_first_10.groupby('profession')['income'].mean().sort_values(ascending=True)
colors = plt.cm.plasma(np.linspace(0, 1, len(avg_income)))
ax6.barh(avg_income.index, avg_income.values, color=colors)
ax6.set_xlabel('Average Income', fontsize=10)
ax6.set_title('Average Income by Profession (All Years)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# 7. Gender Distribution
ax7 = plt.subplot(3, 3, 7)
gender_counts = pop_first_10['sex'].value_counts()
colors_gender = ['#ff9999', '#66b3ff']
wedges, texts, autotexts = ax7.pie(gender_counts.values, labels=gender_counts.index, 
                                     autopct='%1.1f%%', startangle=90, colors=colors_gender)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
ax7.set_title('Gender Distribution (All Years)', fontsize=12, fontweight='bold')

# 8. Total Population Over Time
ax8 = plt.subplot(3, 3, 8)
pop_by_year = pop_first_10.groupby('year').size()
ax8.plot(pop_by_year.index, pop_by_year.values, marker='o', color='purple', linewidth=2, markersize=8)
ax8.fill_between(pop_by_year.index, pop_by_year.values, alpha=0.3, color='purple')
ax8.set_xlabel('Year', fontsize=10)
ax8.set_ylabel('Population', fontsize=10)
ax8.set_title('Total Population Over Time (All Years)', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 9. Income vs Net Worth Scatter (Excluding Children)
ax9 = plt.subplot(3, 3, 9)
pop_no_child = pop_first_10[pop_first_10['profession'] != 'child']
professions_no_child = pop_no_child['profession'].unique()
colors_scatter = plt.cm.tab10(np.linspace(0, 1, len(professions_no_child)))
for i, prof in enumerate(professions_no_child):
    prof_data = pop_no_child[pop_no_child['profession'] == prof]
    ax9.scatter(prof_data['income'], prof_data['net_worth'], alpha=0.6, 
               label=prof, s=50, c=[colors_scatter[i]])
ax9.set_xlabel('Income', fontsize=10)
ax9.set_ylabel('Net Worth', fontsize=10)
ax9.set_title('Income vs Net Worth by Profession (All Years)', fontsize=12, fontweight='bold')
ax9.legend(fontsize=8, loc='best')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('population_stats_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'population_stats_visualization.png'")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS (All Years)")
print("="*60)

print("\n1. Average Net Worth by Profession:")
print("-" * 50)
avg_nw = pop_first_10.groupby('profession')['net_worth'].mean().sort_values(ascending=False)
for prof, value in avg_nw.items():
    print(f"  {prof:20s}: ${value:,.2f}")

print("\n2. Average Income by Profession:")
print("-" * 50)
avg_inc = pop_first_10.groupby('profession')['income'].mean().sort_values(ascending=False)
for prof, value in avg_inc.items():
    print(f"  {prof:20s}: ${value:,.2f}")

print("\n3. Total Population Count by Profession:")
print("-" * 50)
prof_cnt = pop_first_10['profession'].value_counts().sort_values(ascending=False)
for prof, count in prof_cnt.items():
    print(f"  {prof:20s}: {count:,}")

print("\n4. Average GDP by Year:")
print("-" * 50)
for year, gdp in stats_first_10[['year', 'gdp']].values:
    print(f"  Year {int(year):2d}: ${gdp:,.2f}")

print("\n5. Gender Distribution:")
print("-" * 50)
gender_dist = pop_first_10['sex'].value_counts()
total = len(pop_first_10)
for gender, count in gender_dist.items():
    print(f"  {gender}: {count:,} ({count/total*100:.1f}%)")

print("\n" + "="*60)
