import pandas as pd
import glob

# List all CSV files (update the path if necessary)
file_paths = glob.glob(r'F:\Codes\joint attention\Nano-particle\New_training_SHAP\*.csv')

# Initialize an empty list to hold dataframes
dataframes = []

# Read each CSV file and align the columns
for file in file_paths:
    df = pd.read_csv(file)
    dataframes.append(df)

# Combine all dataframes, aligning columns by their names
merged_df = pd.concat(dataframes, ignore_index=True)
mean_df = merged_df.iloc[::2,:]
# Save the combined data to a new CSV file
mean_df.to_csv('merged_output.csv', index=False)

print("Merged CSV created: merged_output.csv")
