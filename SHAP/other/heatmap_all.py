import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Example: Load your data (replace with your actual DataFrame)
df = pd.read_csv('merged_output_v2.0.csv')  # or construct it manually

# df= df.iloc[1:5,:]
df =df.iloc[5:,:]
df.set_index("Model", inplace=True)

# Set the figure size
plt.figure(figsize=(12, 10))

# Create heatmap
# df = df.applymap(lambda x: None if x <= 0 else x)  # Avoid log(0) or log(negative)
# df = df.applymap(lambda x: np.log10(x) if x is not None else np.nan)
sns.heatmap(df, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.tick_params(axis='both', which='major', labelsize=ii, labelcolor='black', labelrotation=0)
ii=12
plt.xticks(fontsize=ii, fontweight='bold')
plt.yticks(fontsize=ii, fontweight='bold')
# plt.xticks(fontsize=ii)
# plt.yticks(fontsize=ii)
# Add labels and title
plt.xlabel("Features", fontsize=14, fontweight='bold')
plt.ylabel("Models", fontsize=14, fontweight='bold')
# plt.xlabel("Features")
# plt.ylabel("Models")
# plt.title("Average SHAP Value Heatmap")

plt.tight_layout()
plt.savefig(r'C:\Users\sanka\Desktop\Dr H\heatmap_RF_group.png', dpi=600, bbox_inches='tight')
plt.show()
# Save the heatmap as an image file


