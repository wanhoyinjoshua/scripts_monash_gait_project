import ezc3d
import numpy as np
import os
from pathlib import Path
import traceback
import os.path as osp
import pickle
import cv2

from human_body_prior.tools.omni_tools import copy2cpu as c2c
import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel


from psbody.mesh import Mesh
from human_body_prior.tools.rotation_tools import rotate_points_xyz

from moshpp.mosh_head import MoSh
from loguru import logger

marker_dict={
        "LFHD": 707,
        "RFHD": 2198,
        "LBHD": 2026,
        "RBHD": 3066,
        "C7": 3354,
        "T10": 5623,
        "CLAV": 5618,
        "STRN": 5531,
        "LSHO": 4481,
        "LELB": 4302,
        "LIWR": 4726,
        "LOWR": 4722,
        "LFIN": 4888,
        "RSHO": 6627,
        "RELB": 7040,
        "RIWR": 7462,
        "ROWR": 7458,
        "RFIN": 7624,
        "LFWT": 3486,
        "RFWT": 6248,
        "LBWT": 5697,
        "RBWT": 8391,
        "LTHI": 3591,
        "LKNE": 3683,
        "LTIB": 3724,
        "LANK": 5882,
        "LHEE": 8846,
        "LTOE": 5895,
        "RTHI": 6352,
        "RKNE": 6444,
        "RTIB": 6485,
        "RANK": 8576,
        "RHEE": 8634,
        "RTOE": 8589,
        
        
      }

exp_marker_dict={
        "LFHD": 707,
        "RFHD": 2198,
        "LBHD": 2026,
        "RBHD": 3066,
        "C7": 3354,
        "T10": 5623,
        "CLAV": 5618,
        "STRN": 5531,
        "LSHO": 4481,
        "LELB": 4302,
        "LIWR": 4726,
        "LOWR": 4722,
        "LFIN": 4888,
        "RSHO": 6627,
        "RELB": 7040,
        "RIWR": 7462,
        "ROWR": 7458,
        "RFIN": 7624,
        "LFWT": 3486,
        "RFWT": 6248,
        "LBWT": 5697,
        "RBWT": 8391,
       
        "LKNE": 3683,
       
        "LANK": 5882,
        "LHEE": 8846,
        "LTOE": 5895,
       
        "RKNE": 6444,
       
        "RANK": 8576,
        "RHEE": 8634,
        "RTOE": 8589,
        
        
      }
def compare(reconstruted_path,real_path):

    return 

def convert_to_mesh_once(stageii_input_file,matched_original_path,isExperiment):
    
    #cfg = prepare_render_cfg(**cfg)

    logger.info(f'Preparing mesh files for: {stageii_input_file}')
    
    outputpath = stageii_input_file.replace("running_just_mosh", "reconstructed")
    print(outputpath)
    outputpath = outputpath.replace(".pkl", ".c3d")
    print(outputpath)
    output_file = Path(outputpath)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f'dirs.mesh_out_dir: {output_file}')

    datas = {}
    selected_frames = None
    time_length = None

   
    mosh_id = '/'.join(stageii_input_file.replace('.pkl', '').split('/')[-2:])

    datas[mosh_id] = {}

    stageii_npz_fname=stageii_input_file.replace('.pkl','.npz')

    mosh_result = MoSh.load_as_amass_npz(stageii_input_file,stageii_npz_fname=stageii_npz_fname, include_markers=True)

    # logger.info(mosh_result.keys())

    num_betas = len(mosh_result['betas']) if 'betas' in mosh_result else 10
    
    surface_model_type = mosh_result['surface_model_type']
    gender = mosh_result['gender']
   

    
    # Todo add object model here
    

    # selected_frames = range(0, 10, step_size)
    if selected_frames is None:
        time_length = len(mosh_result['trans'])
        selected_frames = range(0, time_length)

    assert time_length == len(mosh_result['trans']), \
        ValueError(
            f'All mosh sequences should have same length. {mosh_stageii_pkl_fname} '
            f'has {len(mosh_result["trans"])} != {time_length}')

    datas[mosh_id]['markers'] = mosh_result['markers'][selected_frames]
    datas[mosh_id]['labels'] = mosh_result['labels']
    # todo: add the ability to have a control on marker colors here

    datas[mosh_id]['num_markers'] = mosh_result['markers'].shape[1]

    if 'betas' in mosh_result:
        mosh_result['betas'] = np.repeat(mosh_result['betas'][None], repeats=time_length, axis=0)

    body_keys = ['betas', 'trans', 'pose_body', 'root_orient', 'pose_hand']

    if 'v_template' in mosh_result:
        mosh_result['v_template'] = np.repeat(mosh_result['v_template'][None], repeats=time_length, axis=0)
        body_keys += ['v_template']




    
    #I suspect what I have to do now is to manually map the markers from the standard gait plug in to 
    #the smplx model, which might be difficult....
    first_frame_rot = cv2.Rodrigues(mosh_result['root_orient'][0].copy())[0]
    datas[mosh_id]['theta_z_mosh'] = np.rad2deg(np.arctan2(first_frame_rot[1, 0], first_frame_rot[0, 0]))
    surface_model_fname = osp.join("/mnt/d/ubuntubackup/test/support_files", surface_model_type, gender, 'model.npz')
    num_expressions = len(mosh_result['expression']) if 'expression' in mosh_result else None
    sm = BodyModel(bm_fname=surface_model_fname,
                       num_betas=num_betas,
                       num_expressions=num_expressions,
                       num_dmpls=None,
                       dmpl_fname=None)
    surface_parms = {k: torch.Tensor(v[selected_frames]) for k, v in mosh_result.items() if k in body_keys}

    datas[mosh_id]['mosh_bverts'] = c2c(sm(**surface_parms).v)
    
    dataarray=[]
    for t, fId in enumerate(selected_frames):
        body_mesh = None
        marker_mesh = None

        for mosh_id, data in datas.items():



            
            cur_body_verts = rotate_points_xyz(data['mosh_bverts'][t][None],
                                               np.array([0, 0, 0]).reshape(-1, 3))
            cur_body_verts = rotate_points_xyz(cur_body_verts, np.array([0, 0, 0]).reshape(-1, 3))[0]
           
            #only select the 34 points as
            indices=list(exp_marker_dict.values())
            filteredb_vertices= cur_body_verts[indices]
          


            
            #cur_marker_verts = rotate_points_xyz(data['markers'][t][None],
                                                    #np.array([0, 0, 0]).reshape(-1, 3))
            #-data['theta_z_mosh']
            #cur_marker_verts = rotate_points_xyz(cur_marker_verts, np.array([0, 0, 0]).reshape(-1, 3))[0]
            
            
            dataarray.append(filteredb_vertices*1000)
    
  
    if np.any(np.isnan(dataarray)):
        print("Data contains NaNs.")
    else:
        print("No NaNs in data.")
    markers = np.array(dataarray).transpose(2, 1, 0)  # Reshape to (3, number_of_markers, number_of_frames)
 
   

    dict_used={}
    if isExperiment:
        dict_used=exp_marker_dict
    else:
        dict_used=marker_dict


   

            
    new_c3d = ezc3d.c3d()

    print(markers.shape)

    # Add marker data
    new_c3d["data"]["points"] = markers
    #new_c3d['data']['meta_points']['residuals'] = residuals.transpose([2, 1, 0])

    # Update labels
    print(list(dict_used.keys()))
    new_c3d["parameters"]["POINT"]["LABELS"]["value"] = list(dict_used.keys())

    # Adjust other parameters based on the new marker count
    new_c3d["parameters"]["POINT"]["USED"]["value"] = [len(list(dict_used.keys()))]  # Update marker count
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * len(list(dict_used.keys()))  # Placeholder descriptions

    # Copy metadata from the original file if needed
    new_c3d["parameters"]["ANALOG"]["USED"]["value"] = [0]  # No analog channels
    new_c3d["parameters"]["ANALOG"]["RATE"]["value"] = [0]  # Analog rate
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * len(list(dict_used.keys()))  # Empty descriptions
    new_c3d["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]  # Units
    new_c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
    # Save the updated C3D file
    
 
    new_c3d.write(outputpath)
    print(f"completed export to c3d to {outputpath}")

    print(matched_original_path)
    original_c3d = ezc3d.c3d(matched_original_path)
    

    absolute_differences = np.abs(original_c3d["data"]["points"][:3, :, :] - new_c3d["data"]["points"][:3, :, :])

    # Average the errors for each marker across all frames
    average_errors_per_marker = np.mean(absolute_differences, axis=2)  # Shape: (3, n_markers)

    # Display the average errors for x, y, z per marker
    for i, (x_err, y_err, z_err) in enumerate(average_errors_per_marker.T):
        print(f"Marker {list(dict_used.keys())[i]}: Avg Error -> X: {x_err:.4f}, Y: {y_err:.4f}, Z: {z_err:.4f}")

    
    errors_dict = {
    list(dict_used.keys())[i]: tuple(average_errors_per_marker[:, i])
    for i in range(len(list(dict_used.keys())))
    }

    output_directory_path = os.path.dirname(outputpath)
    with open(f"{output_directory_path}/errors.pkl", "wb") as pkl_file:
        pickle.dump(errors_dict, pkl_file)

    
   
       

    return




parent_folder = Path("/mnt/d/ubuntubackup/test/running_just_mosh/mosh_results/SOMA_manual_labeled")

original_data_folder= Path("/mnt/d/ubuntubackup/test/support_files/evaluation_mocaps/original/SOMA_manual_labeled")
# List to store the paths of all .c3d files
all_original_c3ds= [str(c3d_file) for c3d_file in original_data_folder.rglob("*.c3d")]
print(all_original_c3ds)
pkl = []
#what I can do ti to get all original c3d into a listl then I filter them based on trial name

# Traverse the parent folder and its subfolders
for file in parent_folder.rglob("*.pkl"):  # Use rglob to search recursively for .c3d files

    
    if "stageii" in str(file):  # Check if 'stageii' is in the file path
        pkl.append(str(file))

print(pkl)
for pk in pkl:
    print(pk)
    trialname=pk.split("/")[-2]
    matched_original_path=[x for x in all_original_c3ds if trialname in x]
    print(matched_original_path)
    convert_to_mesh_once(pk,matched_original_path[0],True)