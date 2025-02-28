from pathlib import Path
parent_folder = Path("/mnt/d/ubuntubackup/test/reconstructed/mosh_results/SOMA_manual_labeled")

import os
# List to store the paths of all .c3d files
all_error= [str(c3d_file) for c3d_file in parent_folder.rglob("*.pkl")]


import os
import pickle
import pandas as pd

# Define the directory containing the .pkl files
pkl_directory = all_error
output_csv_path = "combined_errors.csv"

# Initialize a list to store data for the CSV
csv_data = []

# Loop through all .pkl files in the directory
print(all_error)
for file_name in pkl_directory:

        
    # Load the .pkl file
    with open(file_name, "rb") as pkl_file:
        errors_dict = pickle.load(pkl_file)
    
    row = {"File Name": os.path.dirname(file_name).split("/")[-1]}
        
    # Add marker errors to the row
    for marker, (x, y, z) in errors_dict.items():
        row[f"{marker}_X"] = x
        row[f"{marker}_Y"] = y
        row[f"{marker}_Z"] = z
        
        # Keep track of column names dynamically
        
    
    # Append the row to the data
    csv_data.append(row)
    # Compute the average errors across all markers
   
    
    # Append the data for this file
   

# Create a DataFrame from the collected data
df = pd.DataFrame(csv_data)

# Save the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

print(f"Combined errors saved to {output_csv_path}")
