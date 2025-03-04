import ezc3d
import numpy as np
import os
from pathlib import Path
import traceback
import pandas as pd
import json
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

def find_index(subset_labels):

    LFHD = subset_labels.index("LFHD")
    RFHD = subset_labels.index("RFHD")
    LBHD = subset_labels.index("LBHD")
    RBHD = subset_labels.index("RBHD")
    C7 = subset_labels.index("C7")
    T10 = subset_labels.index("T10")
    CLAV = subset_labels.index("CLAV")
    STRN = subset_labels.index("STRN")
    LSHO = subset_labels.index("LSHO")
    LELB = subset_labels.index("LELB")
    LWRA = subset_labels.index("LWRA")
    LWRB = subset_labels.index("LWRB")
    LFIN = subset_labels.index("LFIN")
    RSHO = subset_labels.index("RSHO")
    RELB = subset_labels.index("RELB")
    RWRA = subset_labels.index("RWRA")
    RWRB = subset_labels.index("RWRB")
    RFIN = subset_labels.index("RFIN")
    LASI = subset_labels.index("LASI")
    RASI = subset_labels.index("RASI")
    LPSI = subset_labels.index("LPSI")
    RPSI = subset_labels.index("RPSI")
    LTHI = subset_labels.index("LTHI")
    LKNE = subset_labels.index("LKNE")
    LTIB = subset_labels.index("LTIB")
    LANK = subset_labels.index("LANK")
    LHEE = subset_labels.index("LHEE")
    LTOE = subset_labels.index("LTOE")
    RTHI = subset_labels.index("RTHI")
    RKNE = subset_labels.index("RKNE")
    RTIB = subset_labels.index("RTIB")
    RANK = subset_labels.index("RANK")
    RHEE = subset_labels.index("RHEE")
    RTOE = subset_labels.index("RTOE")
    indices = [
    LFHD, RFHD, LBHD, RBHD, C7, T10, CLAV, STRN, LSHO, LELB,
    LWRA, LWRB, LFIN, RSHO, RELB, RWRA, RWRB, RFIN, LASI, RASI,
    LPSI, RPSI, LTHI, LKNE, LTIB, LANK, LHEE, LTOE, RTHI, RKNE,
    RTIB, RANK, RHEE, RTOE
    ]
   
    return indices

def interpolate_frames(frame1, frame2, steps=10):
    """
    Interpolate linearly between frame1 and frame2 over a specified number of steps.
    Returns an array of interpolated frames.
    """
    # Ensure frames have the same shape, which they should
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes must match for interpolation, but got {frame1.shape} and {frame2.shape}")
    
    # Interpolation factor calculation
    interpolated_frames = []
    for step in range(steps):
        alpha = step / (steps - 1)  # Use steps-1 to get correct interpolation range
        interpolated_frame = (1 - alpha) * frame1 + alpha * frame2
        interpolated_frames.append(interpolated_frame)
    
    return np.array(interpolated_frames)

def join_c3ds(path1,path2,analogs1,analogs2):
    print(path1.shape)
    print(path2.shape)
    print (path1[:,:,-1].shape)
    print(path2[:,:,0].shape)
    smooth_frames = interpolate_frames(path1[:,:,-1], path2[:,:,0], 20)
    print(smooth_frames.transpose(1, 2, 0).shape)

    combined_points = np.concatenate((path1, smooth_frames.transpose(1, 2, 0), path2), axis=2)  # Concatenate marker data
   
    combined_analogs = np.concatenate((analogs1, analogs2), axis=2)
    return [combined_points,combined_analogs]

def findHeelindex(subset_labels):
    LHEE = subset_labels.index("LHEE")
    return LHEE

def center_normalise(inputpath,isExperiment,isCal,heel_height):
    c3d = ezc3d.c3d(inputpath)
    print(c3d)


    # Extract marker data and labels
    all_markers = c3d["data"]["points"]  # Shape: (4, n_markers, n_frames)
    all_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    analogs= c3d["data"]["analogs"]
    
    #LTHI=INDEX 23 
    #LTIB =INDEX 25
    #RTHI =29 
    #RTIB =31 
  


    # Check if there are at least 34 markers
    if all_markers.shape[1] < 34:
        raise ValueError("The C3D file contains fewer than 34 markers!")

    # Extract the first 34 markers and their labels
    subset_markers=[]
    subset_labels=[]

    if isCal:
        def toString(label):
            return label.split(":")[-1]

        subsetlabels=list(map(toString,all_labels))
        custom_indices=find_index(subsetlabels)
        subset_markers = all_markers[:, custom_indices, :]  # Keep only the first 34 markers
        subset_labels = [subsetlabels[i] for i in custom_indices] # First 34 marker labels
        
       
    else:
        subset_markers = all_markers[:, :34, :]  # Keep only the first 34 markers
        subset_labels = all_labels[:34]  # First 34 marker labels


   
    marker_indices = [22, 24, 28, 30]
    lthi_index = subset_labels.index("LTHI")
    LTIB_index= subset_labels.index("LTIB")
    rthi_index = subset_labels.index("RTHI")
    RTIB_index= subset_labels.index("RTIB")

    first_marker_x=subset_markers[1, 0, 0]

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

    pointforwards = False
    T = np.eye(4)
    if isCal:
        heel=all_markers[1, 27, 0]
        toe=all_markers[1, 27, 0]
        
        if heel>toe:
            pass
            
        else:
            T[0, 0] = 0   # cos(90°) = 0
            T[0, 1] = 1   # -sin(90°) = 1
            T[1, 0] = -1  # sin(90°) = -1
            T[1, 1] = 0   # cos(90°) = 0
    else:
        x_first_frame = all_markers[0, 0, 0]

        # Extract the x-coordinate of the first marker from the last frame
        x_last_frame = all_markers[0, 0, -1]
    



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

    
    def calculatecentroid(firstframee):
        centroid = np.mean(firstframee[:3, :], axis=1)
        return centroid 


    


    if isExperiment == True:
        valid_marker_indices = [i for i in range(subset_markers.shape[1]) if i not in marker_indices]

        # Use the mask to filter the array
        subset_markers = subset_markers[:, valid_marker_indices, :]
        
       
        subset_labels = [item for idx, item in enumerate(subset_labels) if idx not in marker_indices]
        len(f"{subset_labels}_label")
        len(subset_markers)
  

    #take first frame
    


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
    if isCal==False:
        current_heel=np.min(subset_markers[2, findHeelindex(subset_labels), :])
        print(current_heel)
        
        cal_heel=heel_height
        distance= current_heel-cal_heel
        print(cal_heel)
        print(distance*-1)
        T[2, 3] = distance*-1


    num_frames = points.shape[2]
    for frame in range(num_frames):
        frame_points = points[:, :, frame]
        valid_indices = ~np.isnan(frame_points).any(axis=0)
        #homogeneous_points = np.vstack((frame_points[:, valid_indices], np.ones(valid_indices.sum())))
    
        #print(homogeneous_points)
        transformed_points = T @ frame_points
        

        points[:, valid_indices, frame] = transformed_points[:4, :]
    return [points,subset_labels,analogs,subset_markers[2, findHeelindex(subset_labels), 0]]
    


def run_once(inputpath,isExperiment):
    #search from input_files for cal folder for the current data trial
    c3d = ezc3d.c3d(inputpath[0])
    trial_no= inputpath[0].split("/")[-2]
    trial=""
    for e in input_paths:
      
        if trial_no in e and "Cal" in e:
            trial=e
        else:
            pass
    print(trial)


    

    trialdata=center_normalise(trial,isExperiment,True,0)
    #i need to get the z coordinated of the heel and output it here 
    print("shitnigga")
    trial_points=trialdata[0]
    trial_labels=trialdata[1]
    trial_analogs=trialdata[2]
    print(trial_labels)
    #here I need to process the c3d trial c3d file. 

    print(inputpath[0])
    data= center_normalise(inputpath[0],isExperiment,False,trialdata[3])
  

    points=data[0]
    subset_labels=data[1]
    analog=data[2]

    # Update the C3D data with the transformed points

    newdata=join_c3ds(trial_points,points,trial_analogs,analog)
    print(newdata)
    # Update the C3D file structure
    new_c3d = ezc3d.c3d()

    # Add marker data
    new_c3d["data"]["points"] = newdata[0]

    # Update labels
    new_c3d["parameters"]["POINT"]["LABELS"]["value"] = subset_labels
    #new_c3d["data"]["analogs"]=newdata[1]
    # Adjust other parameters based on the new marker count
    new_c3d["parameters"]["POINT"]["USED"]["value"] = [len(points)]  # Update marker count
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * len(subset_labels)  # Placeholder descriptions

    # Copy metadata from the original file if needed
    new_c3d["parameters"]["ANALOG"]["USED"]["value"] = [0]  # No analog channels
    new_c3d["parameters"]["ANALOG"]["RATE"]["value"] = [0]  # Analog rate
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * len(subset_labels)  # Empty descriptions
    new_c3d["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]  # Units
    new_c3d["parameters"]["POINT"]["RATE"]["value"] = c3d["parameters"]["POINT"]["RATE"]["value"]
    # Save the updated C3D file
    new_c3d.write(inputpath[1])
    print(f"completed export to c3d to {inputpath[1]}")
    trialname=inputpath[1].split("/")[-1].split(".")[0]
    subjectname=inputpath[1].split("/")[-2]

    newpath=f"/mnt/d/ubuntubackup/test/support_files/evaluation_mocaps/original/SOMA_manual_labeled/{subjectname}/{trialname}.c3d"
    os.makedirs(f"/mnt/d/ubuntubackup/test/support_files/evaluation_mocaps/original/SOMA_manual_labeled/{subjectname}", exist_ok=True)
    new_c3d.write(newpath)
    df=pd.read_csv("subjects_char.csv")
    gender = df.loc[df["ID"] == subjectname, "Gender (M/F)"].values[0]
    if gender =="M":
        data = {"gender": "male"}

        # File path
        file_path = f"/mnt/d/ubuntubackup/test/support_files/evaluation_mocaps/original/SOMA_manual_labeled/{subjectname}/settings.json"

        # Write to a JSON file
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    
    else:
        data = {"gender": "female"}

        # File path
        file_path = f"/mnt/d/ubuntubackup/test/support_files/evaluation_mocaps/original/SOMA_manual_labeled/{subjectname}/settings.json"


        # Write to a JSON file
        with open(file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

    print(f"completed export to c3d to {newpath}")

    
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
    #also now export it to a different location
    trialname=inputpath.split("/")[-1].split(".")[0]
    subjectname=inputpath.split("/")[-2]

    newpath=f"/mnt/d/ubuntubackup/test/support_files/evaluation_mocaps/original/SOMA_manual_labeled/{subjectname}_{trialname}/{trialname}.c3d"
    new_c3d.write(newpath)
    print(f"completed export to c3d to {newpath}")

    
    return



for path in pathdict:

    try:
        if "Cal" in path[0]:
            print("cal")
        else:
            run_once(path,True)
    except Exception as e:
        print(e)
        errorlog.append(path[0])
        traceback.print_exc()
        
        




#then I need to get from all files --> naming conventions boom 
