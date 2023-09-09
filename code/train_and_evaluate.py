import pandas as pd
from dataset import HEAL_Dataset
from models import *
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import gc
from pathlib import Path
from datetime import datetime 
from monai.networks.nets import EfficientNetBN
import random
import numpy as np

import rising.transforms as rtr
from rising.loading import DataLoader, default_transform_call

def get_model_by_params(
    img_dirs,
    dropout,
    net_architecture,
    additional_feature_cols,
    efficientnet_model_name,
    resnet_class = None,
    basic_block_depth = 3,
    pretrained_path = None,
    ensemble_model_params = None,
):

    # Will be replaced if there are pretrained layers
    pretrained_layers = None

    if net_architecture == 'basic':

        if additional_feature_cols:
            net = Basic3DCNN_with_additional_inputs(
                num_additional_inputs = len(additional_feature_cols),
                input_channels=len(img_dirs),
                conv_block_depth = basic_block_depth,
                dropout=dropout
            )
        else:
            net = Basic3DCNN(
                input_channels=len(img_dirs),
                conv_block_depth = basic_block_depth,
                dropout=dropout
            )

        # To do: add model loading
        if pretrained_path:
            pass # Add loading

    elif net_architecture == 'resnet':

        if pretrained_path:
            if additional_feature_cols:
                resnet, pretrained_layers = create_pretrained_medical_resnet(
                    pretrained_path = pretrained_path,
                    model_constructor = resnet_class,
                    spatial_dims = 3,
                    n_input_channels = len(img_dirs),
                    num_classes = 64
                )
                pretrained_layers = ['net.' + x for x in pretrained_layers]

                net = CombineNet(resnet, 64, len(additional_feature_cols))

            else:
                net, pretrained_layers = create_pretrained_medical_resnet(
                    pretrained_path = pretrained_path,
                    model_constructor = resnet_class,
                    spatial_dims = 3,
                    n_input_channels = len(img_dirs),
                    num_classes = 1
                )

        else:
            if additional_feature_cols:
                resnet = resnet_class(
                    pretrained=False,
                    spatial_dims=3,
                    n_input_channels=len(img_dirs),
                    num_classes = 64
                )
                net = CombineNet(resnet, 64, len(additional_feature_cols))

            else:
                net = resnet_class(
                    pretrained=False,
                    spatial_dims=3,
                    n_input_channels=len(img_dirs),
                    num_classes = 1
                )
            

    elif net_architecture == 'efficientnet':
        if additional_feature_cols:
            # Can probably adapt combinenet
            # I want to see if theres a better way to autoencode though

            # Efficient net did poorly so likely will skip implementing this

            pass

        else:
            net = EfficientNetBN(
                efficientnet_model_name, 
                spatial_dims=3, 
                in_channels=len(img_dirs), 
                num_classes=1
            )

        # To do: add model loading
        if pretrained_path:
            pass # Add loading

    elif net_architecture == 'logistic_regression':
        net = SimpleLogisticRegressionModel(len(additional_feature_cols))

        # To do: add model loading
        if pretrained_path:
            pass # Add loading

    elif net_architecture == 'ensemble':

        cnn_model_list = []

        # First, load each model and weights
        for i in range(len(ensemble_model_params['cnn_model_list'])):

            # Load appropriate model with params
            if ensemble_model_params['cnn_model_list'][i]['net_architecture'] == 'basic':
                net = Basic3DCNN(
                    input_channels=1,
                    conv_block_depth = ensemble_model_params['cnn_model_list'][i]['basic_block_depth'],
                    dropout = ensemble_model_params['cnn_model_list'][i]['dropout']
                )

            elif ensemble_model_params['cnn_model_list'][i]['net_architecture'] == 'efficientnet':
                net = EfficientNetBN(
                    ensemble_model_params['cnn_model_list'][i]['efficientnet_model_name'], 
                    spatial_dims=3, 
                    in_channels=len(img_dirs), 
                    num_classes=1
                )

            ## Insert for other model types here

            if 'pretrained_path' in ensemble_model_params['cnn_model_list'][i].keys() and pretrained_path == True:
                # Load weights
                saved_state_dict = torch.load(ensemble_model_params['cnn_model_list'][i]['pretrained_path'])['state_dict']
                saved_state_dict = {k.replace('net.', ''):v for k, v in saved_state_dict.items()}
                net.load_state_dict(saved_state_dict)

            # Append to model list
            cnn_model_list.append(net)

        lr_model_list = []

        for i in range(len(ensemble_model_params['lr_model_list'])):
            
            if ensemble_model_params['lr_model_list'][i]['net_architecture'] == 'logistic_regression':
                net = SimpleLogisticRegressionModel(
                    input_dim=len(additional_feature_cols)
                )

            ## Insert for other model types here

            if 'pretrained_path' in ensemble_model_params['lr_model_list'][i].keys():
                # Load weights
                saved_state_dict = torch.load(ensemble_model_params['lr_model_list'][i]['pretrained_path'])['state_dict']
                saved_state_dict = {k.replace('net.', ''):v for k, v in saved_state_dict.items()}
                net.load_state_dict(saved_state_dict)

            # Append to model list
            lr_model_list.append(net)

        # Now assemble ensemble model
        net = EnsembleModel(
            cnn_model_list=cnn_model_list, 
            cnn_image_index_list = ensemble_model_params['cnn_image_index_list'],
            lr_model_list=lr_model_list
        )

        pretrained_layers = []

        for name, param in net.named_parameters():
            if 'classifier' not in name:
                pretrained_layers.append(name)

    return net, pretrained_layers

def train_model(
    img_dirs,
    total_epochs,
    batch_size,
    dropout,
    learning_rate,
    net_architecture,
    additional_feature_cols,
    transforms,
    efficientnet_model_name,
    label_csv_columns,
    resnet_class = None,
    basic_block_depth = 3,
    pretrained_path = None,
    ensemble_model_params = None,
    num_workers = 4,
    save_base_dir = "../ext_storage/saved_models_storage",
    fine_tune_unfreeze = None,
):
    # See test_runs_script.py for an explanation on each parameter

    start_time = datetime.now()

    ###############################################################################

    # Load in sets
    training_set = pd.read_csv('../data/train_set.csv').values.flatten().tolist()
    val_set = pd.read_csv('../data/val_set.csv').values.flatten().tolist()

    # Create datasets

    label_csv_path = '../data/outcomes_updated.csv'

    train_dataset = HEAL_Dataset(
        image_dirs = img_dirs, 
        subject_list = training_set,
        label_csv_path = label_csv_path,
        subject_col = label_csv_columns['subject_col'],
        label_col = label_csv_columns['label_col'],
        addtional_feature_cols = additional_feature_cols,
    )

    val_dataset = HEAL_Dataset(
        image_dirs = img_dirs, 
        subject_list = val_set,
        label_csv_path = label_csv_path,
        subject_col = label_csv_columns['subject_col'],
        label_col = label_csv_columns['label_col'],
        addtional_feature_cols = additional_feature_cols,
    )

    # Create dataloaders

    # Include drop last for train; otherwise batch norm may fail if 1 leftover at end of epoch
    if transforms:
        transforms = rtr.Compose(transforms, transform_call=default_transform_call)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        drop_last=True,
        batch_transforms=transforms,
        pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    ###############################################################################

    # Model training

    # Get network from parameters; pretrained_layers will be none if not loading
    net, pretrained_layers = get_model_by_params(
        img_dirs,
        dropout,
        net_architecture,
        additional_feature_cols,
        efficientnet_model_name,
        resnet_class,
        basic_block_depth,
        pretrained_path,
        ensemble_model_params,
    )

    model = LitHEAL(
        net = net,
        additional_inputs = bool(additional_feature_cols),
        logistic_regression= bool(net_architecture == 'logistic_regression'),
        pretrained_params = pretrained_layers,
        lr = learning_rate
    )

    # print(model)
    
    # Checkpoint for best val auroc model; last model is also saved by default
    ckpt = pl.callbacks.ModelCheckpoint(
        monitor='avg_val_auroc',
        save_top_k=1,
        save_last=True,
        filename='checkpoint/{epoch:02d}_{avg_val_auroc:.4f}',
        mode='max'
    )

    # Monitor LR
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Unfreeze checkpoint if included
    if fine_tune_unfreeze:
        fine_tune_cb = FineTuneCB(int(fine_tune_unfreeze * total_epochs))
        callbacks = [ckpt, lr_monitor, fine_tune_cb]
    else:
        callbacks = [ckpt, lr_monitor]

    # Tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_base_dir)
    
    # Trainer with mixed procession to reduce memory and increase speed
    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=callbacks, 
        logger=tb_logger,
        max_epochs=total_epochs,
        precision='16-mixed',
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        # fast_dev_run = True
    )

    trainer.fit(model, train_loader, val_loader)

    ###############################################################################

    # Clean up

    del train_dataset, val_dataset
    del train_loader, val_loader
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    time_elapsed = datetime.now() - start_time 
    print('Model trained from {} to {}'.format(
        start_time.strftime("%Y-%m-%d %H:%M"),
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

def cross_validation(
    fold_count,
    img_dirs,
    total_epochs,
    batch_size,
    dropout,
    learning_rate,
    net_architecture,
    additional_feature_cols,
    transforms,
    efficientnet_model_name,
    label_csv_columns,
    resnet_class = None,
    basic_block_depth = 3,
    pretrained_path = None,
    ensemble_model_params = None,
    num_workers = 4,
    save_base_dir = "../ext_storage/saved_models_storage",
    fine_tune_unfreeze = None,
):
    # See test_runs_script.py for an explanation on each parameter

    start_time = datetime.now()

    ###############################################################################

    # Load in sets
    training_set = pd.read_csv('../data/train_set.csv').values.flatten().tolist()
    val_set = pd.read_csv('../data/val_set.csv').values.flatten().tolist()

    # Add sets together and shuffle
    training_and_val_sets = training_set + val_set
    random.shuffle(training_and_val_sets)

    fold_size = round(len(training_and_val_sets) / fold_count)

    # Create lists to hold performance
    val_final_aurocs = []
    val_best_aurocs = []
    epoch_of_val_best_aurocs = []

    for fold_k in range(fold_count):
        fold_start_time = datetime.now()

        val_set = training_and_val_sets[fold_k * fold_size: (fold_k + 1) * fold_size]
        training_set = [x for x in training_and_val_sets if x not in val_set]

        # Create datasets

        label_csv_path = '../data/outcomes_updated.csv'

        train_dataset = HEAL_Dataset(
            image_dirs = img_dirs, 
            subject_list = training_set,
            label_csv_path = label_csv_path,
            subject_col = label_csv_columns['subject_col'],
            label_col = label_csv_columns['label_col'],
            addtional_feature_cols = additional_feature_cols,
        )

        val_dataset = HEAL_Dataset(
            image_dirs = img_dirs, 
            subject_list = val_set,
            label_csv_path = label_csv_path,
            subject_col = label_csv_columns['subject_col'],
            label_col = label_csv_columns['label_col'],
            addtional_feature_cols = additional_feature_cols,
        )

        # Create dataloaders

        # Include drop last for train; otherwise batch norm may fail if 1 leftover at end of epoch
        if transforms:
            composed_transforms = rtr.Compose(transforms, transform_call=default_transform_call)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            drop_last=True,
            batch_transforms=composed_transforms,
            pin_memory=True
        )

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        ###############################################################################

        # Model training

        # Get network from parameters; pretrained_layers will be none if not loading
        net, pretrained_layers = get_model_by_params(
            img_dirs,
            dropout,
            net_architecture,
            additional_feature_cols,
            efficientnet_model_name,
            resnet_class,
            basic_block_depth,
            pretrained_path,
            ensemble_model_params,
        )

        model = LitHEAL(
            net = net,
            additional_inputs = bool(additional_feature_cols),
            logistic_regression= bool(net_architecture == 'logistic_regression'),
            pretrained_params = pretrained_layers,
            lr = learning_rate
        )

        # print(model)
        
        # Checkpoint for best val auroc model; last model is also saved by default
        ckpt = pl.callbacks.ModelCheckpoint(
            monitor='avg_val_auroc',
            save_top_k=1,
            save_last=True,
            filename='checkpoint/{epoch:02d}_{avg_val_auroc:.4f}',
            mode='max'
        )

        # Monitor LR
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

        # Unfreeze checkpoint if included
        if fine_tune_unfreeze:
            fine_tune_cb = FineTuneCB(int(fine_tune_unfreeze * total_epochs))
            callbacks = [ckpt, lr_monitor, fine_tune_cb]
        else:
            callbacks = [ckpt, lr_monitor]

        # Tensorboard
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_base_dir)
        
        # Trainer with mixed procession to reduce memory and increase speed
        trainer = pl.Trainer(
            accelerator="auto",
            callbacks=callbacks, 
            logger=tb_logger,
            max_epochs=total_epochs,
            precision='16-mixed',
            log_every_n_steps=1,
            accumulate_grad_batches=2,
            # fast_dev_run = True
        )

        trainer.fit(model, train_loader, val_loader)

        val_final_aurocs.append(model.current_val_auroc)
        val_best_aurocs.append(model.best_val_auroc)
        epoch_of_val_best_aurocs.append(model.epoch_of_best_val_auroc)

        ###############################################################################

        # Clean up

        del train_dataset, val_dataset
        del train_loader, val_loader
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

        time_elapsed = datetime.now() - fold_start_time 
        print('Fold {} trained from {} to {}'.format(
            fold_k,
            fold_start_time.strftime("%Y-%m-%d %H:%M"),
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    time_elapsed = datetime.now() - start_time 
    print('Cross validation trained from {} to {}'.format(
        start_time.strftime("%Y-%m-%d %H:%M"),
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    print('Mean Final Val AUROC: {} \n Mean Best Val AUROC: {}'.format(
        np.mean(val_final_aurocs),
        np.mean(val_best_aurocs)
    ))

    print('Epoch of best Val AUROCs:', epoch_of_val_best_aurocs)

    return val_final_aurocs, val_best_aurocs, epoch_of_val_best_aurocs

def predict_from_checkpoint(
    checkpoint_path,
    test_csv_path, 
    img_dirs,
    dropout,
    net_architecture,
    label_csv_columns,
    additional_feature_cols,
    efficientnet_model_name,
    resnet_class = resnet18,
    basic_block_depth = 3,
    ensemble_model_params = None,
    num_workers = 4
):
    # Similar to above, but only runs prediction, no training done
    # Has additional parameters for saved checkpoint and test set csv
    # Returns preds, test_labels
    #
    # Importantly, this loads from a pytorch lightning checkpoint

    start_time = datetime.now().tolist()

    ###############################################################################

    # Load in sets
    test_set = pd.read_csv(Path(test_csv_path))

    # Create datasets
    batch_size = 1

    label_csv_path = '../data/outcomes_updated.csv'

    test_dataset = HEAL_Dataset(
        image_dirs = img_dirs, 
        subject_list = test_set.values.flatten(),
        label_csv_path = label_csv_path,
        subject_col = label_csv_columns['subject_col'],
        label_col = label_csv_columns['label_col'],
        addtional_feature_cols = additional_feature_cols,
    )

    label_df = pd.read_csv(label_csv_path)
    test_labels = label_df.loc[label_df['studyid'].isin(test_set), 'primary_all'].values

    # Create dataloaders

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    ###############################################################################

    # Model training

    # Get network from parameters; pretrained_layers will be none if not loading
    net, _ = get_model_by_params(
        img_dirs,
        dropout,
        net_architecture,
        additional_feature_cols,
        efficientnet_model_name,
        resnet_class,
        basic_block_depth,
        ensemble_model_params,
        pretrained_path = None
    )


    model = LitHEAL(
        net = net,
        additional_inputs = bool(additional_feature_cols),
        logistic_regression= bool(net_architecture == 'logistic_regression'),
        lr = 3e-4
    )

    # Load checkpoint data
    model = model.load_from_checkpoint(checkpoint_path, net=net)

    if additional_feature_cols:
        # Loading the checkpoint resets this.. will investigate later
        model.additional_inputs = True

    # Added CPU support
    trainer = pl.Trainer(
        accelerator='auto',
        precision='16-mixed',
        log_every_n_steps=1,
        accumulate_grad_batches=2,
        logger=False,
        enable_model_summary=False
    )

    preds = trainer.predict(model, dataloaders=test_loader)

    ###############################################################################

    # Clean up

    del test_dataset
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    time_elapsed = datetime.now() - start_time 
    print('Model trained from {} to {}'.format(
        start_time.strftime("%Y-%m-%d %H:%M"),
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    return preds, test_labels