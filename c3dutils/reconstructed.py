import ezc3d
import numpy as np
import os
from pathlib import Path
import traceback
import os.path as osp

import cv2

from human_body_prior.tools.omni_tools import copy2cpu as c2c
import torch
from colour import Color
from human_body_prior.body_model.body_model import BodyModel


from psbody.mesh import Mesh
from human_body_prior.tools.rotation_tools import rotate_points_xyz

from moshpp.mosh_head import MoSh
from loguru import logger

marker_dict={"C7": 3832,
        "CLAV": 5533,
        "LANK": 5882,
        "LBHD": 2026,
        "LBWT": 5697,
        "LELB": 4302,
        "LFHD": 707,
        "LFIN": 4788,
        "LFWT": 3486,
        "LHEE": 8846,
        "LIWR": 4726,
        "LKNE": 3682,
        "LOWR": 4722,
        "LSHO": 4481,
        "LTHI": 4088,
        "LTIB": 3745,
        "LTOE": 5787,
        "RANK": 8576,
        "RBHD": 3066,
        "RBWT": 8391,
        "RELB": 7040,
        "RFHD": 2198,
        "RFIN": 7524,
        "RFWT": 6248,
        "RHEE": 8634,
        "RIWR": 7462,
        "RKNE": 6443,
        "ROWR": 7458,
        "RSHO": 6627,
        "RTHI": 6832,
        "RTIB": 6503,
        "RTOE": 8481,
        "STRN": 5531,
        "T10": 5623
      }
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
            indices=list(marker_dict.values())
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
 
   


   
   

            
    new_c3d = ezc3d.c3d()

    # Add marker data
    new_c3d["data"]["points"] = markers
    #new_c3d['data']['meta_points']['residuals'] = residuals.transpose([2, 1, 0])

    # Update labels
    new_c3d["parameters"]["POINT"]["LABELS"]["value"] = list(marker_dict.keys())

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