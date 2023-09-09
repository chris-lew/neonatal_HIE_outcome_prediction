# Script for running test_runs functions repeatedly

# Imports
# Resnets are sometimes included in run_dicts
import os
from pathlib import Path
from monai.networks.nets import resnet18, resnet34, resnet50
from pprint import pprint, pformat
from datetime import datetime

from train_and_evaluate import cross_validation
from transforms import *
from training_parameters import default_training_parameters

# Some warnings that cloud output
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

FOLD_COUNT = 5

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

sequence_combinations = [
    ['../ext_storage/mri_data/preprocessed/DTI_eddy_MD_wm'],
    ['../ext_storage/mri_data/preprocessed/T1_wm'],
    ['../ext_storage/mri_data/preprocessed/T2_wm'],
    ['../ext_storage/mri_data/preprocessed/DTI_eddy_trace_wm'],
    ['../ext_storage/mri_data/preprocessed/DTI_eddy_FA_wm'],
    [
        '../ext_storage/mri_data/preprocessed/T1_wm',
        '../ext_storage/mri_data/preprocessed/T2_wm',
        '../ext_storage/mri_data/preprocessed/DTI_eddy_MD_wm',
    ],
    [
        '../ext_storage/mri_data/preprocessed/T1_wm',
        '../ext_storage/mri_data/preprocessed/T2_wm',
        '../ext_storage/mri_data/preprocessed/DTI_eddy_MD_wm'
        '../ext_storage/mri_data/preprocessed/DTI_eddy_FA_wm',
    ]
]

epochs = [50, 100, 200]

for sequences in sequence_combinations:
    for epoch in epochs:
        run_parameters = {
            'img_dirs': sequences,
            'total_epochs': epoch,
            'basic_block_depth': 4,
            'transforms': TRAIN_TRANSFORMS_MIRROR_PROB
        }

        run_list.append(run_parameters)

# Run logs are saved in same area as models
if 'save_base_dir' in run_list[0].keys():
    run_log_dir = Path(run_list[0]['save_base_dir']) / 'lightning_logs'
else:
    run_log_dir = Path(default_training_parameters['save_base_dir']) / 'lightning_logs'

if not os.path.exists(run_log_dir):
    os.mkdir(run_log_dir)
run_log_file = run_log_dir / 'cross_validation_run_log.txt' 

# Open file and write basic info
f = open(run_log_file, 'a')
f.write('\n------------------------------------------------------------------------\n')
f.write(f'Set of cross validation tests beginning on {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
f.write('Default test run params:\n')
f.write(pformat(default_training_parameters))
f.write('\n\n')

# Now iterate through runs
for run in run_list:

    # Make sure all keys are correct
    incorrect_keys = [x for x in run.keys() if x not in default_training_parameters.keys()]
    assert len(incorrect_keys) == 0, 'Run dict contains incorrect keys'

    # More logging for each run
    start_time = datetime.now()

    f.write('####################################################################################\n')
    f.write(f'Model cross validation start: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    f.write('Cross validation run info:\n')
    f.write(pformat(run))
    f.write('\n')

    pprint(run)

    # Fill out missing keys from default
    missing_keys = [x for x in default_training_parameters.keys() if x not in run.keys()]
    for missing_key in missing_keys:
        run[missing_key] = default_training_parameters[missing_key]   

    # Get area to save runs
    save_base_dir = Path(run['save_base_dir']) / 'lightning_logs'
    # Figure out how many runs have already occurred so it lines up with tensorboard
    current_saved_models = os.listdir(save_base_dir)
    f.write(f'CROSS VALIDATION RUN version_{len(current_saved_models)}\n')
    print(f'CROSS VALIDATION RUN version_{len(current_saved_models)}\n')

    # Train
    val_final_aurocs, val_best_aurocs, epoch_of_val_best_aurocs = cross_validation(
        fold_count = FOLD_COUNT,
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
    f.write(f'Model cross validation end: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
    time_elapsed = datetime.now() - start_time 
    f.write('Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))

    f.write('Mean Final Val AUROC: {} \n Mean Best Val AUROC: {}'.format(
        np.mean(val_final_aurocs),
        np.mean(val_best_aurocs)
    ))

    f.write('Final Val AUROCs\n {}\n Best Val AUROCs\n {}\n Epoch of Best Val AUROCs\n {}'.format(
        val_final_aurocs, val_best_aurocs, epoch_of_val_best_aurocs
    ))

    print('\n####################################################################################\n')

f.close()