import ezc3d
import numpy as np
import os
from pathlib import Path
import traceback
import os.path as osp

import cv2


import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel


from psbody.mesh import Mesh
from human_body_prior.tools.rotation_tools import rotate_points_xyz

from moshpp.mosh_head import MoSh
from loguru import logger

def convert_to_mesh_once(stageii_input_file):
    
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

    mosh_result = MoSh.load_as_amass_npz(stageii_input_file, include_markers=True)

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




    

    first_frame_rot = cv2.Rodrigues(mosh_result['root_orient'][0].copy())[0]
    datas[mosh_id]['theta_z_mosh'] = np.rad2deg(np.arctan2(first_frame_rot[1, 0], first_frame_rot[0, 0]))


    
    dataarray=[]
    for t, fId in enumerate(selected_frames):
        body_mesh = None
        marker_mesh = None

        for mosh_id, data in datas.items():



            


           
            
            cur_marker_verts = rotate_points_xyz(data['markers'][t][None],
                                                    np.array([0, 0, 0]).reshape(-1, 3))
            #-data['theta_z_mosh']
            cur_marker_verts = rotate_points_xyz(cur_marker_verts, np.array([0, 0, 0]).reshape(-1, 3))[0]
            
            
            dataarray.append(cur_marker_verts*1000)
    
  
    if np.any(np.isnan(dataarray)):
        print("Data contains NaNs.")
    else:
        print("No NaNs in data.")
    markers = np.array(dataarray).transpose(2, 1, 0)  # Reshape to (3, number_of_markers, number_of_frames)
 
   


   
   

            
    new_c3d = ezc3d.c3d()

    # Add marker data
    new_c3d["data"]["points"] = markers
    #new_c3d['data']['meta_points']['residuals'] = residuals.transpose([2, 1, 0])

    # Update labels
    new_c3d["parameters"]["POINT"]["LABELS"]["value"] = mosh_result['labels']

    # Adjust other parameters based on the new marker count
    new_c3d["parameters"]["POINT"]["USED"]["value"] = [34]  # Update marker count
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * 34  # Placeholder descriptions

    # Copy metadata from the original file if needed
    new_c3d["parameters"]["ANALOG"]["USED"]["value"] = [0]  # No analog channels
    new_c3d["parameters"]["ANALOG"]["RATE"]["value"] = [0]  # Analog rate
    new_c3d["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = [""] * 34  # Empty descriptions
    new_c3d["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]  # Units
    new_c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
    # Save the updated C3D file
    print(outputpath)
 
    new_c3d.write(outputpath)
    print(f"completed export to c3d to {outputpath}")

    
   
       

    return




parent_folder = Path("/mnt/d/ubuntubackup/test/running_just_mosh/mosh_results/SOMA_manual_labeled")

# List to store the paths of all .c3d files
pkl = []


# Traverse the parent folder and its subfolders
for file in parent_folder.rglob("*.pkl"):  # Use rglob to search recursively for .c3d files

    
    if "stageii" in str(file):  # Check if 'stageii' is in the file path
        pkl.append(str(file))

print(pkl)
for pk in pkl:
    print(pk)
    convert_to_mesh_once(pk)