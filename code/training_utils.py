from pathlib import Path
from pprint import pprint, pformat
import random
from datetime import datetime
import os
import pandas as pd
import numpy as np
import csv
from training_parameters import default_training_parameters
import itertools

from train_and_evaluate import train_model, cross_validation

def train_models_by_params(
    training_set,
    run_list,
    default_training_parameters = default_training_parameters,
    validation_set = None,
    training_type = 'standard', # other option is cross_validation  
    fold_count = 5 # For cross_validation
):
    """
    Trains models by training an individual model or running CV. 
    To see default parameters, see training_parameters.py

    Parameters
    ----------
    training_set: list of str
        List of patients to use
    run_list: list of dict
        List of run parameters
    default_training_parameters: dict
        Default run parameters, see training_parameters.py
    validation_set: list of str
        If standard mode, a validation set can be included
    training_type: str 
        standard or cross_validation 
    fold_count: int
        Number of folds to use for CV
    

    Returns
    -------
    None

    """

    assert training_type in ['standard', 'cross_validation'], 'training_type must be standard or cross_validation'
    assert len(run_list) > 0, 'run_list must contain at least one run or empty dict'

    if 'save_base_dir' in run_list[0].keys():
        run_log_dir = Path(run_list[0]['save_base_dir']) / 'lightning_logs'
    else:
        run_log_dir = Path(default_training_parameters['save_base_dir']) / 'lightning_logs'

    if not os.path.exists(run_log_dir):
        os.mkdir(run_log_dir)

    run_log_file_path = run_log_dir / f'{training_type}_run_log.txt' 
    csv_run_log_file_path = run_log_dir / f'{training_type}_run_log.csv' 

    # Open file and write basic info 
    with open(run_log_file_path, 'a') as run_log_file:
        run_log_file.write(f"""------------------------------------------------------------------------
Set of {training_type} tests beginning on {datetime.now().strftime("%Y-%m-%d %H:%M")}
Default test run params:
{pformat(default_training_parameters)} \n\n""")

    csv_header = [
        'brief_run_description', 'AUC_val_final', 'AUC_val_best', 'AUC_data', 'version_number'
        'img_dirs', 'net_architecture', 'additional_feature_cols',
        'transforms', 'basic_block_depth', 'total_epochs', 'batch_size', 'learning_rate',
        'ensemble_model_params', 'resnet_class', 'densenet_class', 'pretrained_path', 
        'efficientnet_model_name', 'fine_tune_unfreeze', 'dropout',
        'time_start', 'time_end', 'duration'
    ]

    with open(csv_run_log_file_path, 'a') as csv_run_log_file:
        csvwriter = csv.writer(csv_run_log_file)

        # Write header
        csvwriter.writerow(csv_header)
    
        # Write default parameters
        csv_row = ['DEFAULT_PARAMETERS', '', '', '', '']

        for parameter in csv_header[5:-3]: # Skip first 4 because they are accounted for; last 3 will be added later
            csv_row.append(default_training_parameters[parameter])

        csvwriter.writerow(csv_row)
        csvwriter.writerow(['', '']) # Blank row for spacing

    # Now iterate through runs
    for run in run_list:

        # Make sure all keys are correct
        incorrect_keys = [x for x in run.keys() if x not in default_training_parameters.keys()]
        assert len(incorrect_keys) == 0, 'Run dict contains incorrect keys'

        start_time = datetime.now()
        pprint(run)
        original_run_info = run.copy()

        # Fill out missing keys from default
        missing_keys = [x for x in default_training_parameters.keys() if x not in run.keys()]
        for missing_key in missing_keys:
            run[missing_key] = default_training_parameters[missing_key]   

        # Get area to save runs
        save_base_dir = Path(run['save_base_dir']) / 'lightning_logs'
        # Figure out how many runs have already occurred so it lines up with tensorboard
        version_number = len(os.listdir(save_base_dir))

        # Logging
        with open(run_log_file_path, 'a') as run_log_file:
            run_log_file.write(f"""####################################################################################
Model {training_type} start: {datetime.now().strftime("%Y-%m-%d %H:%M")}
{training_type} run info:
{pformat(original_run_info)}

{training_type} RUN version_{version_number}""")

        print(f'{training_type} RUN version_{version_number}\n')

        if training_type == 'standard':
            val_final_auroc, val_best_auroc, epoch_of_val_best_auroc = train_model(
                train_subject_list = training_set,
                val_subject_list = validation_set,
                img_dirs = run['img_dirs'],
                total_epochs = run['total_epochs'],
                batch_size = run['batch_size'],
                dropout = run['dropout'],
                learning_rate = run['learning_rate'],
                net_architecture = run['net_architecture'],
                additional_feature_cols = run['additional_feature_cols'],
                transforms = run['transforms'],
                resnet_class = run['resnet_class'],
                densenet_class = run['densenet_class'],
                pretrained_path = run['pretrained_path'],
                basic_block_depth = run['basic_block_depth'],
                efficientnet_model_name = run['efficientnet_model_name'],
                ensemble_model_params = run['ensemble_model_params'],
                save_base_dir=run['save_base_dir'],
                fine_tune_unfreeze=run['fine_tune_unfreeze'],
                label_csv_columns=run['label_csv_columns'],
                num_workers=run['num_workers']
            )

            end_time = datetime.now()
            time_elapsed = end_time - start_time 

            # More logging
            with open(run_log_file_path, 'a') as run_log_file:
                val_summary_str = f"""Final Val AUROCs
{val_final_auroc}
Best Val AUROCs
{val_best_auroc}
Epoch of Best Val AUROC
{epoch_of_val_best_auroc}"""

                run_log_file.write(f"""Model {training_type} end: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Time elapsed (hh:mm:ss.ms) {time_elapsed}

Final Val AUROC: {np.mean(val_final_auroc)}
Best Val AUROC: {np.mean(val_best_auroc)}

{val_summary_str}""")

        elif training_type == 'cross_validation':

            # Train
            val_final_aurocs, val_best_aurocs, epoch_of_val_best_aurocs = cross_validation(
                subject_list=training_set,
                fold_count = fold_count,
                img_dirs = run['img_dirs'],
                total_epochs = run['total_epochs'],
                batch_size = run['batch_size'],
                dropout = run['dropout'],
                learning_rate = run['learning_rate'],
                net_architecture = run['net_architecture'],
                additional_feature_cols = run['additional_feature_cols'],
                transforms = run['transforms'],
                resnet_class = run['resnet_class'],
                densenet_class = run['densenet_class'],
                pretrained_path = run['pretrained_path'],
                basic_block_depth = run['basic_block_depth'],
                efficientnet_model_name = run['efficientnet_model_name'],
                ensemble_model_params = run['ensemble_model_params'],
                save_base_dir=run['save_base_dir'],
                fine_tune_unfreeze=run['fine_tune_unfreeze'],
                label_csv_columns=run['label_csv_columns'],
                num_workers=run['num_workers']
            )

            end_time = datetime.now()
            time_elapsed = end_time - start_time 

            val_final_auroc = np.mean(val_final_aurocs)
            val_best_auroc = np.mean(val_best_aurocs)

            # More logging
            with open(run_log_file_path, 'a') as run_log_file:
                val_summary_str = f"""Final Val AUROCs
{val_final_aurocs}
Best Val AUROCs
{val_best_aurocs}
Epoch of Best Val AUROCs
{epoch_of_val_best_aurocs}"""

                run_log_file.write(f"""Model {training_type} end: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Time elapsed (hh:mm:ss.ms) {time_elapsed}

Mean Final Val AUROC: {np.mean(val_final_aurocs)}
Mean Best Val AUROC: {np.mean(val_best_aurocs)}

{val_summary_str}""")

        with open(csv_run_log_file_path, 'a') as csv_run_log_file:
            csvwriter = csv.writer(csv_run_log_file)

            csv_row = ['', val_final_auroc, val_best_auroc, val_summary_str, f'version_{version_number}']

            for parameter in csv_header[5:-3]: # Skip first 4 because they are accounted for; last 3 will be added later
                if parameter in original_run_info.keys():
                    csv_row.append(original_run_info[parameter])
                else:
                    csv_row.append('')

            csv_row.append(start_time.strftime("%Y-%m-%d %H:%M"))
            csv_row.append(end_time.strftime("%Y-%m-%d %H:%M"))
            csv_row.append(time_elapsed)

            csvwriter.writerow(csv_row)

        print('\n####################################################################################\n')

    with open(csv_run_log_file_path, 'a') as csv_run_log_file:
        csvwriter = csv.writer(csv_run_log_file)
        csvwriter.writerow(['', '']) # Blank row for spacing


def grid_search(
    training_set,
    grid_search_parameters,
    stable_parameters = None,
    default_training_parameters = default_training_parameters,
    fold_count = 5,
    save_base_dir = None
):
    """
    Performs a grid search across parameters. Each permutation has a N-fold CV
    done. The best performing permutation then has a final model trained.  
    To see default parameters, see training_parameters.py

    Parameters
    ----------
    training_set: list of str
        List of patients to use
    grid_search_parameters: dict of (parameter_name, list of values)
        Dictionary with parameters and different values for each
    stable_parameters: dict
        Parameters to use for all grid search runs
    default_training_parameters: dict
        Default run parameters, see training_parameters.py
    fold_count: int
        Number of folds to use for CV
    save_base_dir: 
        Base dir to save everything. 

    Returns
    -------
    None

    """

    # Ensure parameters are correct
    incorrect_keys = [x for x in grid_search_parameters.keys() if x not in default_training_parameters.keys()]
    assert len(incorrect_keys) == 0, 'grid_search_parameters contains incorrect parameters'

    incorrect_keys = [x for x in stable_parameters.keys() if x not in default_training_parameters.keys()]
    assert len(incorrect_keys) == 0, 'stable_parameters contains incorrect parameters'

    # Set up save dir
    if save_base_dir:
        run_log_dir = Path(save_base_dir) / 'lightning_logs'
    else:
        run_log_dir = Path(default_training_parameters['save_base_dir']) / 'lightning_logs'

    if not os.path.exists(run_log_dir):
        os.mkdir(run_log_dir)

    run_log_file_path = run_log_dir / f'grid_search_run_log.txt' 
    csv_run_log_file_path = run_log_dir / f'grid_search_run_log.csv' 

    # Open file and write basic info 
    with open(run_log_file_path, 'a') as run_log_file:
        run_log_file.write(f"""------------------------------------------------------------------------
Set of grid_seach tests beginning on {datetime.now().strftime("%Y-%m-%d %H:%M")}
Default test run params:
{pformat(default_training_parameters)} \n\n""")

    csv_header = [
        'brief_run_description', 'AUC_val_final', 'AUC_val_best', 'AUC_data', 'version_number', 
        'img_dirs', 'net_architecture', 'additional_feature_cols',
        'transforms', 'basic_block_depth', 'total_epochs', 'batch_size', 'learning_rate',
        'ensemble_model_params', 'resnet_class', 'densenet_class', 'pretrained_path', 
        'efficientnet_model_name', 'fine_tune_unfreeze', 'dropout',
        'time_start', 'time_end', 'duration'
    ]

    with open(csv_run_log_file_path, 'a') as csv_run_log_file:
        csvwriter = csv.writer(csv_run_log_file)

        # Write header
        csvwriter.writerow(csv_header)
    
        # Write default parameters
        csv_row = ['DEFAULT_PARAMETERS', '', '', '', '']

        for parameter in csv_header[5:-3]: # Skip first 4 because they are accounted for; last 3 will be added later
            csv_row.append(default_training_parameters[parameter])

        csvwriter.writerow(csv_row)
        csvwriter.writerow(['', '']) # Blank row for spacing

    # Now we need to create all the combinations for the grid search
    # First sort by key
    grid_search_parameters = dict(sorted(grid_search_parameters.items()))

    # Get name of keys
    grid_search_parameter_names = grid_search_parameters.keys()

    # Get all the values in a list of lists
    grid_search_parameter_values_list = []
    for k, v in grid_search_parameters.items():
        grid_search_parameter_values_list.append(v)

    # Get all permutations
    grid_search_parameter_values_permutations = list(itertools.product(*grid_search_parameter_values_list))

    # Grid search performance dict
    # Will log performance as a tuple of parameters, sorted by alphabet of parameter name
    grid_search_performance = dict()

    # print(grid_search_parameter_names)
    # print(grid_search_parameter_values_permutations)

    run_list = []

    for parameter_permutation in grid_search_parameter_values_permutations:
        run_parameter_dict = dict()

        # Add each one to run dictionary
        for parameter_name, parameter_value in zip(grid_search_parameter_names, parameter_permutation):
            run_parameter_dict[parameter_name] = parameter_value

        # Now add stable parameters
        for parameter_name, parameter_value in stable_parameters.items():
            run_parameter_dict[parameter_name] = parameter_value

        # Store parameter permutation for easy access later
        run_parameter_dict['grid_search_permutation'] = parameter_permutation

        run_list.append(run_parameter_dict)

    # Now CV each permutation
    for index, run in enumerate(run_list):

        # Make sure all keys are correct
        incorrect_keys = [x for x in run.keys() if x not in default_training_parameters.keys()]
        assert len(incorrect_keys) == 1, f'Run dict contains incorrect keys: \n {run}' # Has an extra key: grid_search_permutation

        start_time = datetime.now()
        pprint(run)
        original_run_info = run.copy()

        # Fill out missing keys from default
        missing_keys = [x for x in default_training_parameters.keys() if x not in run.keys()]
        for missing_key in missing_keys:
            run[missing_key] = default_training_parameters[missing_key]   

        # Get area to save runs
        save_base_dir = Path(run['save_base_dir']) / 'lightning_logs'
        # Figure out how many runs have already occurred so it lines up with tensorboard
        version_number = len(os.listdir(save_base_dir))

        # Logging
        with open(run_log_file_path, 'a') as run_log_file:
            run_log_file.write(f"""####################################################################################
Grid search permutation {index+1} of {len(run_list)}
Model grid_search_permutation start: {datetime.now().strftime("%Y-%m-%d %H:%M")}
grid_search_permutation run info:
{pformat(original_run_info)}

grid_search_permutation RUN version_{version_number}""")

        print(f'Grid search permutation {index+1} of {len(run_list)}')
        print(f'grid_search_permutation RUN version_{version_number}\n')

        # CV
        val_final_aurocs, val_best_aurocs, epoch_of_val_best_aurocs = cross_validation(
            subject_list=training_set,
            fold_count = fold_count,
            img_dirs = run['img_dirs'],
            total_epochs = run['total_epochs'],
            batch_size = run['batch_size'],
            dropout = run['dropout'],
            learning_rate = run['learning_rate'],
            net_architecture = run['net_architecture'],
            additional_feature_cols = run['additional_feature_cols'],
            transforms = run['transforms'],
            resnet_class = run['resnet_class'],
            densenet_class = run['densenet_class'],
            pretrained_path = run['pretrained_path'],
            basic_block_depth = run['basic_block_depth'],
            efficientnet_model_name = run['efficientnet_model_name'],
            ensemble_model_params = run['ensemble_model_params'],
            save_base_dir=run['save_base_dir'],
            fine_tune_unfreeze=run['fine_tune_unfreeze'],
            label_csv_columns=run['label_csv_columns'], 
            num_workers=run['num_workers'], 
        )

        end_time = datetime.now()
        time_elapsed = end_time - start_time 

        val_final_auroc = np.mean(val_final_aurocs)
        val_best_auroc = np.mean(val_best_aurocs)

        # More logging
        with open(run_log_file_path, 'a') as run_log_file:
            val_summary_str = f"""Final Val AUROCs
{val_final_aurocs}
Best Val AUROCs
{val_best_aurocs}
Epoch of Best Val AUROCs
{epoch_of_val_best_aurocs}"""

            run_log_file.write(f"""Model grid_search_permutation end: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Time elapsed (hh:mm:ss.ms) {time_elapsed}

Mean Final Val AUROC: {np.mean(val_final_aurocs)}
Mean Best Val AUROC: {np.mean(val_best_aurocs)}

{val_summary_str}""")

        with open(csv_run_log_file_path, 'a') as csv_run_log_file:
            csvwriter = csv.writer(csv_run_log_file)

            csv_row = ['', val_final_auroc, val_best_auroc, val_summary_str, f'version_{version_number}']

            for parameter in csv_header[5:-3]: # Skip first 4 because they are accounted for; last 3 will be added later
                if parameter in original_run_info.keys():
                    csv_row.append(original_run_info[parameter])
                else:
                    csv_row.append('')

            csv_row.append(start_time.strftime("%Y-%m-%d %H:%M"))
            csv_row.append(end_time.strftime("%Y-%m-%d %H:%M"))
            csv_row.append(time_elapsed)

            csvwriter.writerow(csv_row)

        # Finally, store CV performance
        grid_search_performance[run['grid_search_permutation']] = val_final_auroc

        print('\n####################################################################################\n')

    # Get permutation with highest performance
    highest_performance_grid_search_permutation = max(grid_search_performance, key=grid_search_performance.get)

    # Now train that model on all training data to maximize performance
    highest_performance_training_parameters = dict()

    # Add each one to run dictionary
    for parameter_name, parameter_value in zip(grid_search_parameter_names, highest_performance_grid_search_permutation):
        highest_performance_training_parameters[parameter_name] = parameter_value

    # Now add stable parameters
    for parameter_name, parameter_value in stable_parameters.items():
        highest_performance_training_parameters[parameter_name] = parameter_value

    start_time = datetime.now()
    print('############################### GRID SEARCH COMPLETE ###############################')
    print('Highest performing permutation:')
    pprint(highest_performance_training_parameters)
    original_run_info = highest_performance_training_parameters.copy()

    # Fill out missing keys from default
    missing_keys = [x for x in default_training_parameters.keys() if x not in highest_performance_training_parameters.keys()]
    for missing_key in missing_keys:
        highest_performance_training_parameters[missing_key] = default_training_parameters[missing_key]   

    # Get area to save runs
    save_base_dir = Path(highest_performance_training_parameters['save_base_dir']) / 'lightning_logs'
    # Figure out how many runs have already occurred so it lines up with tensorboard
    version_number = len(os.listdir(save_base_dir))

    # Logging
    with open(run_log_file_path, 'a') as run_log_file:
        run_log_file.write(f"""############################### GRID SEARCH COMPLETE ###############################
Model grid_search_permutation start: {datetime.now().strftime("%Y-%m-%d %H:%M")}
grid_search_permutation run info:
{pformat(original_run_info)}

grid_search_permutation RUN version_{version_number}""")


    print(f'highest_performance_grid_search_permutation RUN version_{version_number}\n')

    # CV
    train_model(
        train_subject_list = training_set,
        img_dirs = highest_performance_training_parameters['img_dirs'],
        total_epochs = highest_performance_training_parameters['total_epochs'],
        batch_size = highest_performance_training_parameters['batch_size'],
        dropout = highest_performance_training_parameters['dropout'],
        learning_rate = highest_performance_training_parameters['learning_rate'],
        net_architecture = highest_performance_training_parameters['net_architecture'],
        additional_feature_cols = highest_performance_training_parameters['additional_feature_cols'],
        transforms = highest_performance_training_parameters['transforms'],
        resnet_class = highest_performance_training_parameters['resnet_class'],
        densenet_class = highest_performance_training_parameters['densenet_class'],
        pretrained_path = highest_performance_training_parameters['pretrained_path'],
        basic_block_depth = highest_performance_training_parameters['basic_block_depth'],
        efficientnet_model_name = highest_performance_training_parameters['efficientnet_model_name'],
        ensemble_model_params = highest_performance_training_parameters['ensemble_model_params'],
        save_base_dir=highest_performance_training_parameters['save_base_dir'],
        fine_tune_unfreeze=highest_performance_training_parameters['fine_tune_unfreeze'],
        label_csv_columns=highest_performance_training_parameters['label_csv_columns'],
        num_workers=highest_performance_training_parameters['num_workers'], 
    )

    end_time = datetime.now()
    time_elapsed = end_time - start_time 

    # More logging
    with open(run_log_file_path, 'a') as run_log_file:
        run_log_file.write(f"""Model highest_performance_grid_search_parameters end: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Time elapsed (hh:mm:ss.ms) {time_elapsed}""")

    with open(csv_run_log_file_path, 'a') as csv_run_log_file:
        csvwriter = csv.writer(csv_run_log_file)

        csv_row = ['highest_performance_grid_search_parameters', '', '', '', f'version_{version_number}']

        for parameter in csv_header[5:-3]: # Skip first 4 because they are accounted for; last 3 will be added later
            if parameter in original_run_info.keys():
                csv_row.append(original_run_info[parameter])
            else:
                csv_row.append('')

        csv_row.append(start_time.strftime("%Y-%m-%d %H:%M"))
        csv_row.append(end_time.strftime("%Y-%m-%d %H:%M"))
        csv_row.append(time_elapsed)

        csvwriter.writerow(csv_row)
        csvwriter.writerow(['', '']) # Blank row for spacing

    print('\n####################################################################################\n')

        

