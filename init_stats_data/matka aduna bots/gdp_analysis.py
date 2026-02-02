import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('/Users/kaisahanni/Downloads/population_all_teams_year100 (3).csv')

# Calculate GDP (total income) for each year
gdp_by_year = df.groupby('year')['income'].sum().reset_index()
gdp_by_year.columns = ['Year', 'GDP']

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(gdp_by_year['Year'], gdp_by_year['GDP'], linewidth=2, marker='o', markersize=3)
plt.xlabel('Year', fontsize=12)
plt.ylabel('GDP (Total Income)', fontsize=12)
plt.title('GDP Across Years', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('/Users/kaisahanni/Downloads/gdp_graph.png', dpi=300, bbox_inches='tight')
print(f"Graph saved to: /Users/kaisahanni/Downloads/gdp_graph.png")

# Display the plot
plt.show()

# Print summary statistics
print(f"\nGDP Summary:")
print(f"Years covered: {gdp_by_year['Year'].min()} to {gdp_by_year['Year'].max()}")
print(f"Starting GDP: {gdp_by_year['GDP'].iloc[0]:,.2f}")
print(f"Ending GDP: {gdp_by_year['GDP'].iloc[-1]:,.2f}")
print(f"Maximum GDP: {gdp_by_year['GDP'].max():,.2f} (Year {gdp_by_year.loc[gdp_by_year['GDP'].idxmax(), 'Year']:.0f})")
print(f"Minimum GDP: {gdp_by_year['GDP'].min():,.2f} (Year {gdp_by_year.loc[gdp_by_year['GDP'].idxmin(), 'Year']:.0f})")
