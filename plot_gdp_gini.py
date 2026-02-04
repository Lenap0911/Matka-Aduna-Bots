import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
data = pd.read_csv('data/data/statistics_matka_bots_year110.csv')

# Create figure with two y-axes
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot GDP on left y-axis
color1 = '#2E86AB'
ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
ax1.set_ylabel('GDP', color=color1, fontsize=12, fontweight='bold')
line1 = ax1.plot(data['year'], data['gdp'], color=color1, linewidth=2.5, 
                 marker='o', markersize=4, label='GDP', alpha=0.8)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3, linestyle='--')

# Create second y-axis for Gini
ax2 = ax1.twinx()
color2 = '#A23B72'
ax2.set_ylabel('Gini Coefficient', color=color2, fontsize=12, fontweight='bold')
line2 = ax2.plot(data['year'], data['gini'], color=color2, linewidth=2.5,
                 marker='s', markersize=4, label='Gini', alpha=0.8)
ax2.tick_params(axis='y', labelcolor=color2)

# Set Gini y-axis limits for better visualization
ax2.set_ylim(0.45, 0.65)

# Title and layout
plt.title('GDP and Gini Coefficient Over Time\nMatka Aduna Bots (Years 0-110)', 
          fontsize=14, fontweight='bold', pad=20)

# Add legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', framealpha=0.9)

# Adjust layout
fig.tight_layout()

# Save and show
plt.savefig('gdp_gini_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'gdp_gini_plot.png'")
plt.show()
