import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
profession_counts = pd.read_csv('../../data/data/profession_count_by_year.csv')
profession_income = pd.read_csv('../../data/data/profession_income_by_year .csv')

# Filter for first 10 years (0-9)
counts_first_10 = profession_counts[profession_counts['year'] <= 9]
income_first_10 = profession_income[profession_income['year'] <= 9]

# Calculate average income by profession across years 0-9
avg_income_by_profession = income_first_10.groupby('profession')['avg_income'].mean().sort_values(ascending=False)

# Calculate total counts by profession across years 0-9
total_counts_by_profession = counts_first_10.groupby('profession')['count'].sum().sort_values(ascending=False)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Profession Analysis: First 10 Years (Year 0-9)', fontsize=16, fontweight='bold')

# 1. Average Income by Profession (Bar Chart)
ax1 = axes[0, 0]
colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, len(avg_income_by_profession)))
avg_income_by_profession.plot(kind='bar', ax=ax1, color=colors1)
ax1.set_title('Average Income by Profession', fontsize=14, fontweight='bold')
ax1.set_xlabel('Profession', fontsize=12)
ax1.set_ylabel('Average Income ($)', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)
# Add value labels on bars
for i, v in enumerate(avg_income_by_profession):
    ax1.text(i, v, f'${v:.0f}', ha='center', va='bottom', fontsize=9)

# 2. Total Counts by Profession (Bar Chart)
ax2 = axes[0, 1]
colors2 = plt.cm.plasma(np.linspace(0.3, 0.9, len(total_counts_by_profession)))
total_counts_by_profession.plot(kind='bar', ax=ax2, color=colors2)
ax2.set_title('Total Profession Counts', fontsize=14, fontweight='bold')
ax2.set_xlabel('Profession', fontsize=12)
ax2.set_ylabel('Total Count', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)
# Add value labels on bars
for i, v in enumerate(total_counts_by_profession):
    ax2.text(i, v, f'{v}', ha='center', va='bottom', fontsize=9)

# 3. Income Trends Over Time (Line Chart)
ax3 = axes[1, 0]
# Get top 5 income-earning professions (excluding child)
top_professions = avg_income_by_profession[avg_income_by_profession.index != 'child'].head(5).index
for profession in top_professions:
    prof_data = income_first_10[income_first_10['profession'] == profession]
    ax3.plot(prof_data['year'], prof_data['avg_income'], marker='o', label=profession, linewidth=2)
ax3.set_title('Income Trends Over Time (Top 5 Professions)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Year', fontsize=12)
ax3.set_ylabel('Average Income ($)', fontsize=12)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, 10))

# 4. Profession Distribution (Pie Chart)
ax4 = axes[1, 1]
# Exclude children for better visualization
counts_no_children = total_counts_by_profession[total_counts_by_profession.index != 'child']
colors4 = plt.cm.Set3(np.linspace(0, 1, len(counts_no_children)))
ax4.pie(counts_no_children, labels=counts_no_children.index, autopct='%1.1f%%',
        startangle=90, colors=colors4)
ax4.set_title('Profession Distribution (Excluding Children)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('first_10_years_analysis.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'first_10_years_analysis.png'")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS: FIRST 10 YEARS (Year 0-9)")
print("="*60)
print("\nAverage Income by Profession:")
print("-" * 40)
for profession, income in avg_income_by_profession.items():
    print(f"{profession:20s}: ${income:,.2f}")

print("\n\nTotal Counts by Profession:")
print("-" * 40)
for profession, count in total_counts_by_profession.items():
    print(f"{profession:20s}: {count:,}")
print("="*60)
