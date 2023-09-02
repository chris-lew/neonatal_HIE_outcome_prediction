# Script for running test_runs functions repeatedly

# Imports
# Resnets are sometimes included in run_dicts
import os
from pathlib import Path
from monai.networks.nets import resnet18, resnet34, resnet50
from pprint import pprint, pformat
from datetime import datetime

from test_runs import train_run
from transforms import *

# Some warnings that cloud output
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


# Dictionary that contains all parameters to be used in each model test
# Default dict if value is unspecified
default_run_dict = {
    # Images to use
    'img_dirs': ['../ext_storage/mri_data/preprocessed/DTI_eddy_MD_wm'],

    # Hyperparams
    'total_epochs': 100,
    'batch_size': 4,
    'dropout': 0.3,
    'learning_rate': 3e-4,

    # Add params
    'additional_feature_cols': None, # ['sex', 'txtassign', 'inf_gestage_zscore', 'total_brain_injury_volume_zscore']
    'transforms': None, # transforms.py contains various sets of transforms from rising
    'fine_tune_unfreeze': None, # What % to unfreeze all weights in model to allow for fine tuning

    # Model: basic, efficientnet, resnet, ensemble, logistic_regression
    'net_architecture': 'basic',

    # Model params
    'pretrained_path': None, # Note: to use pretrained models in ensemble this MUST be set to True
    'basic_block_depth': 3, # For basic CNN model
    'resnet_class': None, # For resnet
    'efficientnet_model_name': 'efficientnet-b0',
    'ensemble_model_params': None, # See below for example

    # Other
    'save_base_dir': '../ext_storage/saved_models_storage',
    'label_csv_columns': {
        'subject_col': 'studyid',
        'label_col': 'primary_all'
    }
}

    # Example:
    # 'ensemble_model_params: {
    #     'cnn_model_list': [
    #         # Basic
    #         # ADC
    #         {
    #             'pretrained_path': ADC_pretrained_path, # Pretrained paths are optional
    #             'net_architecture': 'basic',
    #             'basic_block_depth': 4,
    #             'dropout': 0.3,
    #         },
    #         # T1
    #         {
    #             'pretrained_path': T1_pretrained_path,
    #             'net_architecture': 'basic',
    #             'basic_block_depth': 4,
    #             'dropout': 0.3,
    #         },
    #         ... Can include as many as desired with varying architecture and can repeat channels
    #     ],
    #     'cnn_image_index_list': [0, 1], # For each corresponding model in above list
    # 
    #     'lr_model_list': [
    #         {
    #             'pretrained_path': 'LR_pretrained_path,
    #             'net_architecture': 'logistic_regression',
    #         },
    #     ],
    # }


# Run dicts; not included will be populated from default
run_list = [

    ## Example of a fully loaded CNN

    # {
    #     'img_dirs': [
    #         '../ext_storage/mri_data/preprocessed/DTI_eddy_MD_wm',
    #         '../ext_storage/mri_data/preprocessed/T1_wm',
    #         '../ext_storage/mri_data/preprocessed/T2_wm'
    #     ],
    #     'total_epochs': 500,
    #     'additional_feature_cols': ['sex', 'txtassign', 'inf_gestage_zscore', 'total_brain_injury_volume_zscore'],
    #     'transforms': TRAIN_TRANSFORMS_MIRROR_PROB, 
    #     'basic_block_depth': 4
    # },

    # Can include additional dicts
]
    

# Run logs are saved in same area as models
if 'save_base_dir' in run_list[0].keys():
    run_log_dir = Path(run_list[0]['save_base_dir']) / 'lightning_logs'
else:
    run_log_dir = Path(default_run_dict['save_base_dir']) / 'lightning_logs'

if not os.path.exists(run_log_dir):
    os.mkdir(run_log_dir)
run_log_file = run_log_dir / 'run_log.txt' 

# Open file and write basic info
f = open(run_log_file, 'a')
f.write('\n------------------------------------------------------------------------\n')
f.write(f'Set of tests beginning on {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
f.write('Default test run params:\n')
f.write(pformat(default_run_dict))
f.write('\n\n')

# Now iterate through runs
for run in run_list:

    # Make sure all keys are correct
    incorrect_keys = [x for x in run.keys() if x not in default_run_dict.keys()]
    assert len(incorrect_keys) == 0, 'Run dict contains incorrect keys'

    # More logging for each run
    start_time = datetime.now()

    f.write('###################################################################\n')
    f.write(f'Model test start: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    f.write('Test run info:\n')
    f.write(pformat(run))
    f.write('\n')

    pprint(run)

    # Fill out missing keys from default
    missing_keys = [x for x in default_run_dict.keys() if x not in run.keys()]
    for missing_key in missing_keys:
        run[missing_key] = default_run_dict[missing_key]   

    # Get area to save runs
    save_base_dir = Path(run['save_base_dir']) / 'lightning_logs'
    # Figure out how many runs have already occurred so it lines up with tensorboard
    current_saved_models = os.listdir(save_base_dir)
    f.write(f'TEST RUN version_{len(current_saved_models)}\n')
    print(f'TEST RUN version_{len(current_saved_models)}\n')

    # Train
    train_run(
        img_dirs = run['img_dirs'],
        total_epochs = run['total_epochs'],
        batch_size = run['batch_size'],
        dropout = run['dropout'],
        learning_rate = run['learning_rate'],
        net_architecture = run['net_architecture'],
        additional_feature_cols = run['additional_feature_cols'],
        transforms = run['transforms'],
        resnet_class = run['resnet_class'],
        pretrained_path = run['pretrained_path'],
        basic_block_depth = run['basic_block_depth'],
        efficientnet_model_name = run['efficientnet_model_name'],
        ensemble_model_params = run['ensemble_model_params'],
        save_base_dir=run['save_base_dir'],
        fine_tune_unfreeze=run['fine_tune_unfreeze'],
        label_csv_columns=run['label_csv_columns']
    )

    # More logging
    f.write(f'Model test end: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
    time_elapsed = datetime.now() - start_time 
    f.write('Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))

    print()
    print('########################################################################')
    print()