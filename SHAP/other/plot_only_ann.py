import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Example DataFrame (replace this with your actual merged DataFrame)
# data = {
#     "Column1": [10, 20, 30, 40, 50, 60, 70],
#     "Column2": [15, 25, 35, 45, 55, 65, 75],
#     "Column3": [20, 30, 40, 50, 60, 70, 80],
#     "Model": ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE", "ModelF", "ModelG"]
# }
# df = pd.DataFrame(data)
csv_file_path = r"F:\Codes\joint attention\Nano-particle\New_training_SHAP\other\merged_output_v2.0.csv"

category_1 = ['Curve_21-30', 'Volume', 'N_bonds', 'Curve_1-10', 'Curve_11-20', 'Formation_E', 'Total_E', 'Curve_31-40']
category_2 = [ 'S_111', 'Avg_total', 'FCC', 'Avg_bonds', 'Avg_surf', 'q6q6_avg_surf', 'Curve_41-50', 'HCP','Std_bonds', 'S_100']
category_3 = ['Curve_51-60', 'tau', 'T', 'q6q6_avg_total', 'angle_avg', 'S_110', 'Avg_bulk', 'time', 'q6q6_avg_bulk', 'DECA', 'angle_std', 'S_311', 'Max_bonds', 'Curve_61-70', 'Min_bonds', 'Curve_71-80', 'ICOS']

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file_path)
df = df.iloc[[0]] #selectmodels ; comment if using all models
# sorted_df = df.sort_values(by='Age', ascending=False)
# Separate columns for plotting
columns_to_plot = df.columns[:-1]  # All columns except 'Model'
models = df["Model"]    

# Prepare the data for grouped bar plot
x = np.arange(len(columns_to_plot))  # Positions for each group (column)
width = 0.48  # Width of each bar
colors = plt.cm.tab10.colors[:len(models)]  # Get distinct colors

# Create the plot
plt.figure(figsize=(14, 7))
# Assign a color to each feature based on its category
category_colors = {}
for feature in columns_to_plot:
    if feature in category_1:
        category_colors[feature] = "#1f77b4"  # blue
    elif feature in category_2:
        category_colors[feature] = "#2ca02c"  # green
    elif feature in category_3:
        category_colors[feature] = "#d62728"  # red
    else:
        category_colors[feature] = "#7f7f7f"  # gray for uncategorized

bar_colors = [category_colors[feature] for feature in columns_to_plot]
for i, model in enumerate(models):
    y = df.iloc[i, :-1]  # Get values for this model across all columns
    plt.bar(x + i * width, y, width, label=model, color=bar_colors)

# Customize the plot
ii=12
plt.xticks(x + width * (len(models) - 1) / 2, columns_to_plot, rotation=90,fontsize=ii, fontweight='bold')
plt.yticks(fontsize=ii, fontweight='bold')
plt.ylabel("Average |SHAP| Value", fontsize=14, fontweight='bold')
plt.xlabel("Feature",fontsize=14, fontweight='bold')
# plt.title("Feature-wise average SHAP value distribution (ordered by descending order of ANN)")
# plt.legend(title="Models", loc="upper left", bbox_to_anchor=(1, 1))
# Add legend for the three categories (colors)

category_legend = [
    Patch(facecolor="#1f77b4", label="Group 1"),
    Patch(facecolor="#2ca02c", label="Group 2"),
    Patch(facecolor="#d62728", label="Group 3")
    # Patch(facecolor="#7f7f7f", label="Uncategorized")
]
plt.legend(handles=category_legend, title="Feature Group", loc="upper right")

# Adjust layout for readability
plt.tight_layout()
#save the plot as a PNG file
plt.savefig('ann_shap_plot1.png', dpi=600, bbox_inches='tight')
plt.show()
a=1
