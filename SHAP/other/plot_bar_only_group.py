import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example DataFrame (replace this with your actual merged DataFrame)
# data = {
#     "Column1": [10, 20, 30, 40, 50, 60, 70],
#     "Column2": [15, 25, 35, 45, 55, 65, 75],
#     "Column3": [20, 30, 40, 50, 60, 70, 80],
#     "Model": ["ModelA", "ModelB", "ModelC", "ModelD", "ModelE", "ModelF", "ModelG"]
# }
# df = pd.DataFrame(data)
csv_file_path = r"F:\Codes\joint attention\Nano-particle\New_training_SHAP\other\merged_output_v2.0.csv"

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file_path)
# df = df.iloc[[0]] #selectmodels ; comment if using all models
# sorted_df = df.sort_values(by='Age', ascending=False)
# Separate columns for plotting
columns_to_plot = df.columns[:-1]  # All columns except 'Model'
models = df["Model"]

models = models[5:]

# Prepare the data for grouped bar plot
x = np.arange(len(columns_to_plot))  # Positions for each group (column)
width = 0.12  # Width of each bar
colors = plt.cm.tab10.colors[:len(models)]  # Get distinct colors

# Create the plot
plt.figure(figsize=(14, 7))

for i, model in enumerate(models):
    y = df.iloc[i, :-1]  # Get values for this model across all columns
    plt.bar(x + i * width, y, width, label=model, color=colors[i])

# Customize the plot
plt.xticks(x + width * (len(models) - 1) / 2, columns_to_plot, rotation=90)
plt.ylabel("Average |SHAP| Value")
plt.xlabel("Feature")
# plt.title("Feature-wise average SHAP value distribution (ordered by descending order of ANN)")
# plt.legend(title="Models", loc="upper left", bbox_to_anchor=(1, 1))
plt.legend(title="Models", loc="upper right")

# Adjust layout for readability
plt.tight_layout()
plt.savefig(r"C:\Users\sanka\Desktop\Dr H\bar_plot_RF_group.png", dpi=300,bbox_inches='tight')
# plt.show()
a=1