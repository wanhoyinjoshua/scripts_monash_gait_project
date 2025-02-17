import ezc3d
import numpy as np
import os
from pathlib import Path

#this script will take in a c3d file and position the c3d mocap data in the origin.
#ad it will output the updated c3d in the specified directory
parent_folder = Path("/mnt/d/Research_data_central/Processed_c3d/stroke_dataset")
subfolders = [folder for folder in parent_folder.iterdir() if folder.is_dir()]
print(subfolders)
print("shit")
# List to store the paths of all .c3d files
import ezc3d
import numpy as np

def merge(animfile, cal_file):
    # Load C3D files
    anim_c3d = ezc3d.c3d(animfile)
    cal_c3d = ezc3d.c3d(cal_file)

    # Extract marker data
    all_anim_markers = anim_c3d["data"]["points"]  # Shape: (4, n_markers, n_frames)
    all_cal_markers = cal_c3d["data"]["points"]    # Shape: (4, n_markers, n_frames)
    print(all_anim_markers.shape)
    print(all_cal_markers.shape)
    
    # Check marker consistency
    n_markers_anim = all_anim_markers.shape[1]
    n_markers_cal = all_cal_markers.shape[1]

    if n_markers_anim != n_markers_cal:
        raise ValueError("The number of markers in the two files does not match.")

    # Combine marker data along the frame dimension
    combined_markers = np.concatenate((all_anim_markers, all_cal_markers), axis=2)
    #print(combined_markers.shape())
    print(combined_markers.shape)
    # Copy animation labels
    all_anim_labels = anim_c3d["parameters"]["POINT"]["LABELS"]["value"]

    # Create a new C3D file
    new_c3d = ezc3d.c3d()

    # Add combined marker data
    new_c3d["data"]["points"] = combined_markers

    # Update marker labels and related parameters
    new_c3d["parameters"]["POINT"]["LABELS"]["value"] = all_anim_labels
    new_c3d["parameters"]["POINT"]["USED"]["value"] = [n_markers_anim]  # Update marker count
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * n_markers_anim
    new_c3d["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]  # Ensure consistent units
    new_c3d["parameters"]["POINT"]["RATE"]["value"] = anim_c3d["parameters"]["POINT"]["RATE"]["value"]

    # Copy other metadata if needed
    new_c3d["parameters"]["ANALOG"]["USED"]["value"] = [0]
    new_c3d["parameters"]["ANALOG"]["RATE"]["value"] = [0]

    # Save the new C3D file
    new_c3d.write(f"new{animfile}")

    print(f"Combined C3D file saved to {animfile}")



for subject_folder in subfolders:
    c3d_files = []
    cal_files=[]

    input_paths = [str(path) for path in parent_folder.rglob("*.c3d")]
    for file in input_paths:
        if "Cal" in file:  # Use rglob to search recursively for .c3d files
            cal_files.append(file)
        else:
            c3d_files.append(file)

    for c3d in c3d_files:
        merge(c3d,cal_files[0])

    







