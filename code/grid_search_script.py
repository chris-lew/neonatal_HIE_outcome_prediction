# Script for running grid search

# Imports
# Resnets/densenets are sometimes included in run_dicts
import os
import pandas as pd
from pathlib import Path
from monai.networks.nets import resnet18, resnet34, resnet50
from monai.networks.nets.densenet import DenseNet121
from pprint import pprint, pformat
from datetime import datetime

from training_utils import grid_search
from transforms import *

# Some warnings that cloud output
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

training_set = pd.read_csv('../data/train_set.csv').values.flatten().tolist()
val_set = pd.read_csv('../data/val_set.csv').values.flatten().tolist()
training_and_val_sets = training_set + val_set

sequence_path_base_str = '../ext_storage/mri_data/preprocessed/'

sequence_combinations = [
    [sequence_path_base_str + 'DTI_eddy_MD_wm'],
    [sequence_path_base_str + 'T1_wm'],
    [sequence_path_base_str + 'T2_wm'],
    [sequence_path_base_str + 'DTI_eddy_trace_wm'],
    [sequence_path_base_str + 'DTI_eddy_FA_wm'],
    [
        sequence_path_base_str + 'T1_wm',
        sequence_path_base_str + 'T2_wm',
        sequence_path_base_str + 'DTI_eddy_MD_wm',
    ],
    [
        sequence_path_base_str + 'T1_wm',
        sequence_path_base_str + 'T2_wm',
        sequence_path_base_str + 'DTI_eddy_MD_wm',
        sequence_path_base_str + 'DTI_eddy_trace_wm',
    ],
    [
        sequence_path_base_str + 'T1_wm',
        sequence_path_base_str + 'T2_wm',
        sequence_path_base_str + 'DTI_eddy_MD_wm',
        sequence_path_base_str + 'DTI_eddy_trace_wm',
        sequence_path_base_str + 'DTI_eddy_FA_wm'
    ],
]

grid_search_parameters = {
    'learning_rate': [3e-5, 3e-4, 3e-3],
    'total_epochs': [50, 100, 200],
    'batch_size': [2, 4, 8]
}

# Training different basic CNN on various channels/combinations
for sequences in sequence_combinations:
    stable_run_parameters = {
        'img_dirs': sequences,
        'transforms': TRAIN_TRANSFORMS_MIRROR_PROB,
        'basic_block_depth': 4,
        'num_workers': 16 # Increased for higher performance computing
    }

    grid_search(
        training_and_val_sets, 
        grid_search_parameters, 
        stable_run_parameters
    )

# Again, with tabular data
for sequences in sequence_combinations:
    stable_run_parameters = {
        'img_dirs': sequences,
        'additional_feature_cols': [
            'sex', 'txtassign', 
            'inf_gestage_zscore', 'total_brain_injury_volume_zscore', 
            'inf_gestage_minmax', 'total_brain_injury_volume_minmax'
        ],
        'transforms': TRAIN_TRANSFORMS_MIRROR_PROB,
        'basic_block_depth': 4,
        'num_workers': 16 # Increased for higher performance computing
    }

    grid_search(
        training_and_val_sets, 
        grid_search_parameters, 
        stable_run_parameters
    )

# Again, using densenet
# Again, with tabular data
for sequences in sequence_combinations:
    stable_run_parameters = {
        'img_dirs': sequences,
        'transforms': TRAIN_TRANSFORMS_MIRROR_PROB,
        'net_architecture': 'densenet',
        'densenet_class': DenseNet121,
        'num_workers': 16 # Increased for higher performance computing
    }

    grid_search(
        training_and_val_sets, 
        grid_search_parameters, 
        stable_run_parameters
    )

# Finally, LR
stable_run_parameters = {
    'additional_feature_cols': [
        'sex', 'txtassign', 
        'inf_gestage_zscore', 'total_brain_injury_volume_zscore', 
        'inf_gestage_minmax', 'total_brain_injury_volume_minmax'
    ],
    'net_architecture': 'logistic_regression',

    'num_workers': 16 # Increased for higher performance computing
}

grid_search(
    training_and_val_sets, 
    grid_search_parameters, 
    stable_run_parameters
)