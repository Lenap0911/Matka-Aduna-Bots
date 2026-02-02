import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('/Users/kaisahanni/Downloads/population_all_teams_year100 (3).csv')

# ============================================================================
# PART 1: CHILDREN GENDER COUNT OVER YEARS
# ============================================================================

# Filter for children only
children_df = df[df['profession'] == 'child']

# Count children by sex and year
children_sex_count = children_df.groupby(['year', 'sex']).size().reset_index(name='count')
children_pivot = children_sex_count.pivot(index='year', columns='sex', values='count').fillna(0)
children_pivot['Total'] = children_pivot.sum(axis=1)

# ============================================================================
# PART 2: PROFESSION GENDER COUNTS ACROSS YEARS
# ============================================================================

# Count by profession, sex, and year
profession_sex_count = df.groupby(['year', 'profession', 'sex']).size().reset_index(name='count')

# Get unique professions and sexes
professions = sorted(df['profession'].unique())
sexes = sorted(df['sex'].unique())

# Create pivot for each profession
profession_sex_pivots = {}
for profession in professions:
    prof_data = profession_sex_count[profession_sex_count['profession'] == profession]
    pivot = prof_data.pivot(index='year', columns='sex', values='count').fillna(0)
    profession_sex_pivots[profession] = pivot

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Create comprehensive visualizations
fig = plt.figure(figsize=(18, 14))

# Plot 1: Children gender count over years
ax1 = plt.subplot(3, 3, 1)
for sex in children_pivot.columns[:-1]:
    ax1.plot(children_pivot.index, children_pivot[sex], marker='o', markersize=3, 
             linewidth=2, label=f'{sex}', alpha=0.8)
ax1.set_xlabel('Year', fontsize=10)
ax1.set_ylabel('Count', fontsize=10)
ax1.set_title('Children: Gender Count Over Years', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Children stacked area
ax2 = plt.subplot(3, 3, 2)
ax2.stackplot(children_pivot.index, 
              *[children_pivot[col] for col in children_pivot.columns[:-1]], 
              labels=children_pivot.columns[:-1], alpha=0.8)
ax2.set_xlabel('Year', fontsize=10)
ax2.set_ylabel('Count', fontsize=10)
ax2.set_title('Children: Stacked Gender Distribution', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Total children over time
ax3 = plt.subplot(3, 3, 3)
ax3.plot(children_pivot.index, children_pivot['Total'], marker='o', markersize=3, 
         linewidth=2, color='purple', alpha=0.8)
ax3.set_xlabel('Year', fontsize=10)
ax3.set_ylabel('Total Count', fontsize=10)
ax3.set_title('Total Children Over Years', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plots 4-9: Gender distribution for selected professions
plot_idx = 4
selected_professions = [p for p in professions if p != 'child'][:6]  # Top 6 non-child professions
for profession in selected_professions:
    ax = plt.subplot(3, 3, plot_idx)
    pivot = profession_sex_pivots[profession]
    for sex in pivot.columns:
        ax.plot(pivot.index, pivot[sex], marker='o', markersize=2, 
                linewidth=1.5, label=f'{sex}', alpha=0.8)
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(f'{profession.capitalize()}: Gender Over Years', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plot_idx += 1

plt.tight_layout()
plt.savefig('/Users/kaisahanni/Downloads/gender_profession_trends.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: /Users/kaisahanni/Downloads/gender_profession_trends.png")
plt.close()

# ============================================================================
# DETAILED OUTPUT
# ============================================================================

print("\n" + "="*100)
print("PART 1: CHILDREN GENDER COUNT OVER YEARS (ALL 101 YEARS)")
print("="*100)

child_sexes = sorted(children_pivot.columns[:-1])
header = f"{'Year':<6}"
for sex in child_sexes:
    header += f" | {sex:>8}"
header += f" | {'Total':>8}"
print(header)
print("-"*100)

for year in sorted(children_pivot.index):
    line = f"{int(year):<6}"
    for sex in child_sexes:
        count = children_pivot.loc[year, sex] if sex in children_pivot.columns else 0
        line += f" | {int(count):>8}"
    line += f" | {int(children_pivot.loc[year, 'Total']):>8}"
    print(line)

print("="*100)

# Children statistics
print("\n" + "="*100)
print("CHILDREN STATISTICS BY GENDER")
print("="*100)

for sex in child_sexes:
    if sex in children_pivot.columns:
        print(f"\n{sex}:")
        print(f"  Average per year: {children_pivot[sex].mean():.2f}")
        print(f"  Minimum: {children_pivot[sex].min():.0f} (Year {children_pivot[sex].idxmin()})")
        print(f"  Maximum: {children_pivot[sex].max():.0f} (Year {children_pivot[sex].idxmax()})")
        print(f"  Starting count (Year 0): {children_pivot.loc[0, sex]:.0f}")
        print(f"  Ending count (Year 100): {children_pivot.loc[100, sex]:.0f}")
        change = children_pivot.loc[100, sex] - children_pivot.loc[0, sex]
        pct_change = (change / children_pivot.loc[0, sex] * 100) if children_pivot.loc[0, sex] > 0 else 0
        print(f"  Change: {change:+.0f} ({pct_change:+.1f}%)")

print(f"\nTOTAL CHILDREN:")
print(f"  Average per year: {children_pivot['Total'].mean():.2f}")
print(f"  Minimum: {children_pivot['Total'].min():.0f} (Year {children_pivot['Total'].idxmin()})")
print(f"  Maximum: {children_pivot['Total'].max():.0f} (Year {children_pivot['Total'].idxmax()})")
print(f"  Starting total (Year 0): {children_pivot.loc[0, 'Total']:.0f}")
print(f"  Ending total (Year 100): {children_pivot.loc[100, 'Total']:.0f}")
change = children_pivot.loc[100, 'Total'] - children_pivot.loc[0, 'Total']
pct_change = (change / children_pivot.loc[0, 'Total'] * 100) if children_pivot.loc[0, 'Total'] > 0 else 0
print(f"  Change: {change:+.0f} ({pct_change:+.1f}%)")

print("\n" + "="*100)

# ============================================================================
# PART 2: PROFESSION GENDER COUNTS ACROSS YEARS
# ============================================================================

print("\n" + "="*100)
print("PART 2: PROFESSION GENDER COUNTS ACROSS YEARS")
print("="*100)

for profession in professions:
    print(f"\n{'-'*100}")
    print(f"PROFESSION: {profession.upper()}")
    print(f"{'-'*100}")
    
    pivot = profession_sex_pivots[profession]
    
    # Print year-by-year data (showing every 10th year for brevity)
    print(f"\n{'Year':<6}", end='')
    for sex in sexes:
        print(f" | {sex:>8}", end='')
    print(f" | {'Total':>8}")
    print("-"*100)
    
    for year in sorted(pivot.index):
        if year % 10 == 0 or year == 100:  # Show every 10 years plus year 100
            total = sum(pivot.loc[year, sex] if sex in pivot.columns else 0 for sex in sexes)
            print(f"{int(year):<6}", end='')
            for sex in sexes:
                count = pivot.loc[year, sex] if sex in pivot.columns else 0
                print(f" | {int(count):>8}", end='')
            print(f" | {int(total):>8}")
    
    # Statistics for this profession
    print(f"\nStatistics for {profession}:")
    for sex in sexes:
        if sex in pivot.columns and pivot[sex].sum() > 0:
            print(f"  {sex}: Avg = {pivot[sex].mean():.1f}, Min = {pivot[sex].min():.0f}, Max = {pivot[sex].max():.0f}, Total across all years = {pivot[sex].sum():.0f}")

print("\n" + "="*100)

# ============================================================================
# SUMMARY TABLE: AVERAGE GENDER DISTRIBUTION BY PROFESSION
# ============================================================================

print("\n" + "="*100)
print("AVERAGE GENDER DISTRIBUTION BY PROFESSION (ACROSS ALL YEARS)")
print("="*100)

summary_data = []
for profession in professions:
    pivot = profession_sex_pivots[profession]
    row = {'Profession': profession}
    for sex in sexes:
        if sex in pivot.columns:
            row[sex] = pivot[sex].mean()
        else:
            row[sex] = 0
    row['Total'] = sum(row[s] for s in sexes)
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Total', ascending=False)

print(f"\n{summary_df.to_string(index=False, float_format=lambda x: f'{x:.1f}')}")

print("\n" + "="*100)

# ============================================================================
# EXPORT TO CSV FILES
# ============================================================================

# Export children data
children_pivot_export = children_pivot.reset_index()
children_pivot_export.to_csv('/Users/kaisahanni/Downloads/children_gender_by_year.csv', index=False)
print("\nChildren gender data exported to: /Users/kaisahanni/Downloads/children_gender_by_year.csv")

# Export profession-sex data for all years
profession_sex_count.to_csv('/Users/kaisahanni/Downloads/profession_gender_counts_all_years.csv', index=False)
print("Profession gender data exported to: /Users/kaisahanni/Downloads/profession_gender_counts_all_years.csv")

# Export summary table
summary_df.to_csv('/Users/kaisahanni/Downloads/profession_gender_summary.csv', index=False)
print("Profession gender summary exported to: /Users/kaisahanni/Downloads/profession_gender_summary.csv")

print("\n" + "="*100)
