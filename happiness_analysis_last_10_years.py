import pandas as pd
import numpy as np

# Read the population data
df = pd.read_csv('population_matka_bots_year105.csv')

# Filter for the last 10 years (years 96-105)
last_10_years = df[df['year'] >= 96]

# Calculate average happiness per profession for the last 10 years
avg_happiness_by_profession = last_10_years.groupby('profession')['happiness'].mean().sort_values(ascending=False)

print("Average Happiness by Profession (Years 96-105)")
print("=" * 60)
print(f"\n{'Profession':<25} {'Avg Happiness':>15}")
print("-" * 60)

for profession, happiness in avg_happiness_by_profession.items():
    print(f"{profession:<25} {happiness:>15.2f}")

print("\n" + "=" * 60)
print(f"Overall Average: {last_10_years['happiness'].mean():.2f}")

# Save to CSV
output_df = pd.DataFrame({
    'profession': avg_happiness_by_profession.index,
    'average_happiness': avg_happiness_by_profession.values
})
output_df.to_csv('avg_happiness_by_profession_years_96_105.csv', index=False)
print("\nResults saved to: avg_happiness_by_profession_years_96_105.csv")
