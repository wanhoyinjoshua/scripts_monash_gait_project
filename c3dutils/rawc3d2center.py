import ezc3d
import numpy as np
import os
from pathlib import Path
import traceback

#this script will take in a c3d file and position the c3d mocap data in the origin.
#ad it will output the updated c3d in the specified directory
parent_folder = Path("/mnt/d/Research_data_central/Raw_c3d/")

# List to store the paths of all .c3d files
c3d_files = []

error_files=['/mnt/d/Research_data_central/Raw_c3d/stroke_dataset/TVC20/BWA6.c3d', '/mnt/d/Research_data_central/Raw_c3d/stroke_dataset/TVC21/BWA6.c3d', '/mnt/d/Research_data_central/Raw_c3d/stroke_dataset/TVC52/BWA7.c3d', '/mnt/d/Research_data_central/Raw_c3d/stroke_dataset/TVC53/BWA3.c3d', '/mnt/d/Research_data_central/Raw_c3d/stroke_dataset/TVC53/BWA4.c3d', '/mnt/d/Research_data_central/Raw_c3d/stroke_dataset/TVC53/BWA5.c3d', '/mnt/d/Research_data_central/Raw_c3d/stroke_dataset/TVC60/BWA01.c3d']

# Traverse the parent folder and its subfolders
for file in parent_folder.rglob("*.c3d"):  # Use rglob to search recursively for .c3d files
    c3d_files.append(file)



#inputpath = "/mnt/d/Research_data_central/Raw_c3d/stroke_dataset/TVC03/BWA6.c3d"
#outputpath = "/mnt/d/Research_data_central/Processed_c3d/stroke_dataset/TVC03/Processed_BWA6.c3d"


input_paths = [str(path) for path in c3d_files]

outputpaths=[]

for actual_path in input_paths:
    outputpath = actual_path.replace("Raw_c3d", "Processed_c3d")
    output_file = Path(outputpath)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    outputpaths.append(outputpath)

pathdict=[(x, y) for x, y in zip(input_paths, outputpaths)]

print(pathdict)

errorlog=[]

def replace_nan_with_closest(points):
    # Iterate over each frame
    for frame in range(points.shape[2]):  # Loop over frames
        frame_points = points[:, :, frame]
        
        # Loop over each marker
        for marker in range(frame_points.shape[1]):
            # Find where the marker has NaN values
            nan_indices = np.isnan(frame_points[:, marker])
            
            # Replace NaN values with closest valid values
            for i in range(frame_points.shape[0]):  # Loop over coordinates (x, y, z)
                if nan_indices[i]:
                    # Find the closest non-NaN value (forward fill or backward fill)
                    prev_value = None
                    next_value = None
                    
                    # Check previous frames for valid values
                    for back_frame in range(frame-1, -1, -1):
                        if not np.isnan(points[i, marker, back_frame]):
                            prev_value = points[i, marker, back_frame]
                            break
                    
                    # Check next frames for valid values
                    for forward_frame in range(frame+1, points.shape[2]):
                        if not np.isnan(points[i, marker, forward_frame]):
                            next_value = points[i, marker, forward_frame]
                            break
                    
                    # If both previous and next values exist, use an average of them
                    if prev_value is not None and next_value is not None:
                        points[i, marker, frame] = (prev_value + next_value) / 2
                    elif prev_value is not None:
                        points[i, marker, frame] = prev_value
                    elif next_value is not None:
                        points[i, marker, frame] = next_value

    return points


def run_once(inputpath):
    c3d = ezc3d.c3d(inputpath[0])


    # Extract marker data and labels
    all_markers = c3d["data"]["points"]  # Shape: (4, n_markers, n_frames)
    all_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    print(all_labels)
    #LTHI=INDEX 23 
    #LTIB =INDEX 25
    #RTHI =29 
    #RTIB =31 
  


    # Check if there are at least 34 markers
    if all_markers.shape[1] < 34:
        raise ValueError("The C3D file contains fewer than 34 markers!")

    # Extract the first 34 markers and their labels
    subset_markers = all_markers[:, :34, :]  # Keep only the first 34 markers
    subset_labels = all_labels[:34]  # First 34 marker labels
    marker_indices = [22, 24, 28, 30]
    lthi_index = subset_labels.index("LTHI")
    LTIB_index= subset_labels.index("LTIB")
    rthi_index = subset_labels.index("RTHI")
    RTIB_index= subset_labels.index("RTIB")
    print(lthi_index)
    first_marker_x=subset_markers[1, 0, 0]
    print(first_marker_x)
    print(subset_markers[1, rthi_index, 0])
    wand_length=60
    if first_marker_x - subset_markers[1, rthi_index, 0] <0 :

        subset_markers[1, rthi_index, :] -= wand_length  # Adjust x-coordinates
        subset_markers[1, RTIB_index, :] -= wand_length  # Adjust x-coordinates

        subset_markers[1, lthi_index, :] += wand_length  # Adjust x-coordinates
        subset_markers[1, LTIB_index, :] += wand_length  # Adjust x-coordinates
    else:
        subset_markers[1, rthi_index, :] += wand_length  # Adjust x-coordinates
        subset_markers[1, RTIB_index, :] += wand_length  # Adjust x-coordinates

        subset_markers[1, lthi_index, :] -= wand_length  # Adjust x-coordinates
        subset_markers[1, LTIB_index, :] -= wand_length  # Adjust x-coordinates


    #take first frame
    def calculatecentroid(firstframee):
        centroid = np.mean(firstframee[:3, :], axis=1)
        return centroid 


    x_first_frame = all_markers[0, 0, 0]

    # Extract the x-coordinate of the first marker from the last frame
    x_last_frame = all_markers[0, 0, -1]
    T = np.eye(4)

    if x_first_frame - x_last_frame <0:
        #define transformation matrix
        

        #it is is -ve rotate the other way.
        T[0, 0] = 0  # cos(90°) = 0
        T[0, 1] = 1  # sin(90°) = 1
        T[1, 0] = -1  # -sin(90°) = -1
        T[1, 1] = 0  # cos(90°) = 0
    else:
        T[0, 0] = 0  # cos(90°) = 0
        T[0, 1] = -1  # -sin(90°) = -1
        T[1, 0] = 1   # sin(90°) = 1
        T[1, 1] = 0   # cos(90°) = 0




    #how do I make sure they are facing all the same way?


    #apply transformation matrix on all points. 


    #calculate centroid 
    #calulate transformtion necessary from centroid to origin 
    #c=apply transformation to all points
    #ensure orientation correct, rotate if necessary.
    points=subset_markers
    points=replace_nan_with_closest(points)
    num_frames = points.shape[2]
    for frame in range(num_frames):
        
        frame_points = points[:, :, frame]
        valid_indices = ~np.isnan(frame_points).any(axis=0)

        #homogeneous_points = np.vstack((frame_points[:, valid_indices], np.ones(valid_indices.sum())))
    
        #print(homogeneous_points)
        transformed_points = T @ frame_points
        
        points[:, valid_indices, frame] = transformed_points[:4, :]

    # Update the C3D data with the transformed points

    #now need to move rotated points to origin 

    centroid= calculatecentroid(points[:, :, 0])
    T = np.eye(4)
    T[0, 3] = centroid[0]*-1 # Set the x translation
    T[1,3]=centroid[1]*-1
    num_frames = points.shape[2]
    for frame in range(num_frames):
        frame_points = points[:, :, frame]
        valid_indices = ~np.isnan(frame_points).any(axis=0)
        #homogeneous_points = np.vstack((frame_points[:, valid_indices], np.ones(valid_indices.sum())))
    
        #print(homogeneous_points)
        transformed_points = T @ frame_points
        

        points[:, valid_indices, frame] = transformed_points[:4, :]

    # Update the C3D data with the transformed points



    # Update the C3D file structure
    new_c3d = ezc3d.c3d()

    # Add marker data
    new_c3d["data"]["points"] = points

    # Update labels
    new_c3d["parameters"]["POINT"]["LABELS"]["value"] = subset_labels

    # Adjust other parameters based on the new marker count
    new_c3d["parameters"]["POINT"]["USED"]["value"] = [34]  # Update marker count
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * 34  # Placeholder descriptions

    # Copy metadata from the original file if needed
    new_c3d["parameters"]["ANALOG"]["USED"]["value"] = [0]  # No analog channels
    new_c3d["parameters"]["ANALOG"]["RATE"]["value"] = [0]  # Analog rate
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * 34  # Empty descriptions
    new_c3d["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]  # Units
    new_c3d["parameters"]["POINT"]["RATE"]["value"] = c3d["parameters"]["POINT"]["RATE"]["value"]
    # Save the updated C3D file
    new_c3d.write(inputpath[1])
    print(f"completed export to c3d to {inputpath[1]}")
    
    return


def run_cal(inputpath):
    c3d = ezc3d.c3d(inputpath[0])


    # Extract marker data and labels
    all_markers = c3d["data"]["points"]  # Shape: (4, n_markers, n_frames)
    all_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]

    # Check if there are at least 34 markers
    if all_markers.shape[1] < 34:
        raise ValueError("The C3D file contains fewer than 34 markers!")

    # Extract the first 34 markers and their labels
    subset_markers = all_markers[:, 76:110, :]  # Keep only the first 34 markers
    subset_labels = all_labels[76:110]  # First 34 marker labels
    temp_labels=[]
    for label in subset_labels:
        temp_labels.append(label.split(":")[1])
    subset_labels=temp_labels

    #take first frame
    def calculatecentroid(firstframee):
        centroid = np.mean(firstframee[:3, :], axis=1)
        return centroid 


    x_first_frame = all_markers[0, 0, 0]

    # Extract the x-coordinate of the first marker from the last frame
    x_last_frame = all_markers[0, 0, -1]
    T = np.eye(4)

    



    T[0, 0] = 0  # cos(90°) = 0
    T[0, 1] = 1  # sin(90°) = 1
    T[1, 0] = -1  # -sin(90°) = -1
    T[1, 1] = 0  # cos(90°) = 0




    #how do I make sure they are facing all the same way?


    #apply transformation matrix on all points. 


    #calculate centroid 
    #calulate transformtion necessary from centroid to origin 
    #c=apply transformation to all points
    #ensure orientation correct, rotate if necessary.
    points=subset_markers
    num_frames = points.shape[2]
    for frame in range(num_frames):
        frame_points = points[:, :, frame]
        valid_indices = ~np.isnan(frame_points).any(axis=0)
        #homogeneous_points = np.vstack((frame_points[:, valid_indices], np.ones(valid_indices.sum())))
    
        #print(homogeneous_points)
        transformed_points = T @ frame_points
        
        points[:, valid_indices, frame] = transformed_points[:4, :]

    # Update the C3D data with the transformed points

    #now need to move rotated points to origin 

    centroid= calculatecentroid(points[:, :, 0])
    T = np.eye(4)
    T[0, 3] = centroid[0]*-1 # Set the x translation
    T[1,3]=centroid[1]*-1
    num_frames = points.shape[2]
    for frame in range(num_frames):
        frame_points = points[:, :, frame]
        valid_indices = ~np.isnan(frame_points).any(axis=0)
        #homogeneous_points = np.vstack((frame_points[:, valid_indices], np.ones(valid_indices.sum())))
    
        #print(homogeneous_points)
        transformed_points = T @ frame_points

        points[:, valid_indices, frame] = transformed_points[:4, :]

    # Update the C3D data with the transformed points



    # Update the C3D file structure
    new_c3d = ezc3d.c3d()

    # Add marker data
    new_c3d["data"]["points"] = points

    # Update labels
    new_c3d["parameters"]["POINT"]["LABELS"]["value"] = subset_labels

    # Adjust other parameters based on the new marker count
    new_c3d["parameters"]["POINT"]["USED"]["value"] = [34]  # Update marker count
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * 34  # Placeholder descriptions

    # Copy metadata from the original file if needed
    new_c3d["parameters"]["ANALOG"]["USED"]["value"] = [0]  # No analog channels
    new_c3d["parameters"]["ANALOG"]["RATE"]["value"] = [0]  # Analog rate
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * 34  # Empty descriptions
    new_c3d["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]  # Units
    new_c3d["parameters"]["POINT"]["RATE"]["value"] = c3d["parameters"]["POINT"]["RATE"]["value"]
    # Save the updated C3D file
    new_c3d.write(inputpath[1])
    print(f"completed export to c3d to {inputpath[1]}")
    
    return



for path in pathdict:

    try:
        if "Cal" in path[0]:
            print("cal")
        else:
            run_once(path)
    except Exception as e:
        errorlog.append(path[0])
        print(e)
        


print(errorlog)