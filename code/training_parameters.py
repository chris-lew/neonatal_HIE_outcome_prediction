# Dictionary that contains all parameters to be used in each model test
# Default dict if value is unspecified
default_training_parameters = {
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


# Running some tests
# sequence_combinations = [
#     ['../ext_storage/mri_data/preprocessed/DTI_eddy_MD_wm'],
#     ['../ext_storage/mri_data/preprocessed/T1_wm'],
#     ['../ext_storage/mri_data/preprocessed/T2_wm'],
#     ['../ext_storage/mri_data/preprocessed/DTI_eddy_trace_wm'],
#     ['../ext_storage/mri_data/preprocessed/DTI_eddy_FA_wm'],
#     [
#         '../ext_storage/mri_data/preprocessed/T1_wm',
#         '../ext_storage/mri_data/preprocessed/T2_wm',
#         '../ext_storage/mri_data/preprocessed/DTI_eddy_MD_wm',
#     ]
# ]