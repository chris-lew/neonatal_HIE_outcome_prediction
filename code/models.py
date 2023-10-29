
# Contains model classes

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.networks.nets import resnet18
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from monai.networks.nets import ResNet, resnet18
from efficientnet_pytorch_3d import EfficientNet3D
 

class LitHEAL(pl.LightningModule):
    """
    Lightning module that handles model training, metrics are logged via
    tensorboard. 

    """

    def __init__(
        self,
        net,
        additional_inputs = False,
        logistic_regression = False,
        pretrained_params = None,
        lr = 1e-3,
        optimizer = AdamW,
        decision_fusion = False,
    ):
        """
        Initialize. Some parameters are used to help define what outputs
        to get while iterating through dataset. 

        Parameters
        ----------
        net: nn.Module
            Pytorch module containing model
        additional_inputs: bool
            Whether additional tabular data is included
        logistic_regression: bool
            Whether only logistic regression will be used (no imaging data)
        pretraing_params: [str, str, ...]
            If using a pretrained model, which parameters to freeze
        lr: float
            Learning rate
        optimizer: 
            Optimizer to use

        """

        super(LitHEAL, self).__init__()

        self.net = net
        self.additional_inputs = additional_inputs
        self.logistic_regression = logistic_regression
        self.decision_fusion = decision_fusion

        # Freeze parameters if inputted network is pretrained
        if pretrained_params:
            self.pretrained_params = set(pretrained_params)
            
            for n, param in self.net.named_parameters():

                param.requires_grad = bool(n not in self.pretrained_params)
            
        self.learning_rate = lr
        self.optimizer = optimizer 

        # Create metrics for each set
        self.train_step_loss = []
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.train_auroc = torchmetrics.AUROC(task='binary')

        self.val_step_loss = []
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')

        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.test_auroc = torchmetrics.AUROC(task='binary')

        # Saving some additional parameters
        self.best_val_auroc = 0
        self.epoch_of_best_val_auroc = 0
        self.current_val_auroc = 0

        # For predictions
        self.predictions = []
        self.prediction_labels = []


    def forward(self, img, additional_inputs = None):
        # Depending on task, collect data from dataset and return prediction
        # if self.decision_fusion:
        #     if self.additional_inputs:
        #         return self.net(img, additional_inputs)
        #     else:
        #         return self.net(img)
        if self.additional_inputs:
            if self.logistic_regression:
                return torch.sigmoid(self.net(additional_inputs)[:, 0])
            else:
                return torch.sigmoid(self.net(img, additional_inputs)[:, 0])
        else:
            return torch.sigmoid(self.net(img)[:, 0])

    @staticmethod
    def compute_loss(y_hat: Tensor, y: Tensor):
        # Compute loss

        # return F.binary_cross_entropy(y_hat, y.to(y_hat.dtype))
        return F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

    def training_step(self, batch, batch_idx):
        # Get data and labels -> predict
        img, y = batch["image"], batch["label"]

        if self.additional_inputs:
            additional_features = batch['additional_features']
            y_hat = self(img, additional_features)
        else:
            y_hat = self(img)
        
        # Loss and metrics
        loss = self.compute_loss(y_hat, y)

        self.train_step_loss.append(loss)
        self.train_acc(y_hat, y)
        self.train_auroc(y_hat, y)

        return loss

    def on_train_epoch_end(self):
        # Compute overall metrics and reset for next epoch
        avg_train_loss = torch.stack(self.train_step_loss).mean()
        self.train_step_loss.clear()

        self.log_dict({
            'avg_train_loss': avg_train_loss,
            'avg_train_acc': self.train_acc.compute(),
            'avg_train_auroc': self.train_auroc.compute(),
            # 'avg_train_f1': self.train_f1_score.compute()
        }, on_step=False, on_epoch=True)

        self.train_acc.reset()
        self.train_auroc.reset()


    def validation_step(self, batch, batch_idx):
        # Get data and labels -> predict
        img, y = batch["image"], batch["label"]

        if self.additional_inputs:
            additional_features = batch['additional_features']
            y_hat = self(img, additional_features)
        else:
            y_hat = self(img)

        # Loss and metrics
        loss = self.compute_loss(y_hat, y)

        self.val_step_loss.append(loss)
        self.val_acc(y_hat, y)
        self.val_auroc(y_hat, y)

        return loss

    def on_validation_epoch_end(self):
        # Compute overall metrics and reset for next epoch
        avg_val_loss = torch.stack(self.val_step_loss).mean()
        self.val_step_loss.clear()

        val_auroc = self.val_auroc.compute()

        if val_auroc > self.best_val_auroc:
            self.best_val_auroc = val_auroc
            self.epoch_of_best_val_auroc = self.current_epoch
        self.current_val_auroc = val_auroc

        self.log_dict({
            'avg_val_loss': avg_val_loss,
            'avg_val_acc': self.val_acc.compute(),
            'avg_val_auroc': val_auroc,
        }, on_step=False, on_epoch=True)

        self.val_acc.reset()
        self.val_auroc.reset()


    def test_step(self, batch, batch_idx):
        # Get data and labels -> predict
        img, y = batch["image"], batch["label"]

        if self.additional_inputs:
            additional_features = batch['additional_features']
            y_hat = self(img, additional_features)
        else:
            y_hat = self(img)

        # Loss and metrics
        loss = self.compute_loss(y_hat, y)
    
        self.test_acc(y_hat, y)
        self.test_auroc(y_hat, y)

        return loss

    def on_test_epoch_end(self):
        # Compute overall metrics and reset for next epoch
        self.log_dict({
            'avg_test_acc': self.test_acc.compute(),
            'avg_test_auroc': self.test_auroc.compute(),
        }, on_step=False, on_epoch=True)

        self.test_acc.reset()
        self.test_auroc.reset()

    def predict_step(self, batch, batch_idx):
        # Get data and predict
        img = batch["image"]

        if 'label' in batch.keys():
            y = batch["label"]
            self.prediction_labels.append(y)

        if self.additional_inputs:
            additional_features = batch['additional_features']

            y_hat = self(img, additional_features)

        else:
            y_hat = self(img)

        self.predictions.append(y_hat)

        return y_hat

    def configure_optimizers(self):
        # Optimizer and scheduler
        optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        return [optimizer], [scheduler]
    
    def get_current_val_auroc(self):
        if type(self.current_val_auroc) == torch.Tensor:
            return self.current_val_auroc.detach().cpu().numpy()
        else:
            return self.current_val_auroc
    
    def get_best_val_auroc(self):
        if type(self.best_val_auroc) == torch.Tensor:
            return self.best_val_auroc.detach().cpu().numpy()
        else:
            return self.best_val_auroc
    
    def get_epoch_of_best_val_auroc(self):     
        return self.epoch_of_best_val_auroc
    
    def get_all_predictions(self):
        return [x.detach().cpu().item() for x in self.predictions]

    def get_all_prediction_labels(self):
        return [x.detach().cpu().item() for x in self.prediction_labels]


# Fine tuning
class FineTuneCB(pl.Callback):
    # Callback that can unfreeze frozen layers at a certain epoch
    def __init__(self, unfreeze_epoch: int) -> None:
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.current_epoch != self.unfreeze_epoch:
            return
        for n, param in pl_module.net.named_parameters():
            param.requires_grad = True
        optimizers, _ = pl_module.configure_optimizers()
        trainer.optimizers = optimizers


def _create_conv_layer_set(in_channels, out_channels):
        conv_3d = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=(3, 3, 3), 
            stride=1,
            padding=0,
        )
        nn.init.kaiming_normal_(conv_3d.weight)

        conv_block = nn.Sequential(
            conv_3d,
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
        )
        
        return conv_block

def _create_conv_block_set(conv_block_depth, input_channels, conv_channel_multipier):
    conv_block_list = nn.ModuleList()

    for i in range(conv_block_depth):
            if i == 0:
                conv_block = _create_conv_layer_set(input_channels, conv_channel_multipier)
            else:
                conv_block = _create_conv_layer_set(
                    conv_channel_multipier * (2**(i-1)), 
                    conv_channel_multipier * (2**i)
                )

            bn_layer = nn.BatchNorm3d(conv_channel_multipier * (2**i))

            conv_block_list.append(conv_block)
            conv_block_list.append(bn_layer)

    return conv_block_list

class Basic3DCNN(nn.Module):
    # Baseline CNN

    def __init__(
        self,
        input_channels = 1,
        out_classes = 1,
        conv_block_depth = 3,
        conv_channel_multipier = 16,
        dropout = 0.3
    ):  
        """
        Parameters
        ----------
        input_channels: int
            Number of channels in images
        out_classes: int
            Number of classes to predict
        conv_block_depth: int
            Number of conv blocks to include
        conv_channel_multipier: int
            Factor to multiply channels within conv blocks
        dropout: float
            Dropout percent after first linear layer
        """

        super(Basic3DCNN, self).__init__()

        self.conv_block_list = _create_conv_block_set(
            conv_block_depth, 
            input_channels, 
            conv_channel_multipier
        )        

        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, out_classes)

        self.fc1_relu = nn.LeakyReLU()
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2_relu = nn.LeakyReLU()
        self.fc2_bn = nn.BatchNorm1d(128)

        self.drop = nn.Dropout(p = dropout)    

    def forward(self, x):
        
        for i in range(len(self.conv_block_list)):
            x = self.conv_block_list[i](x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc1_bn(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.fc2_relu(x)
        x = self.fc2_bn(x)

        x = self.fc3(x)

        return x


class Basic3DCNN_with_additional_inputs(nn.Module):
    # Similar to above class, but allows for tabular data
    # Tabular data is added to features derived from images

    def __init__(
        self,
        num_additional_inputs, 
        input_channels = 1,
        out_classes = 1,
        conv_block_depth = 3,
        conv_channel_multipier = 16,
        dropout = 0.3,
        img_features_count = 64,
        final_fc_size = 32
    ):
        """
        Parameters
        ----------
        num_additional_inputs: int
            Number of additional inputs to include
        input_channels: int
            Number of channels in images
        out_classes: int
            Number of classes to predict
        conv_block_depth: int
            Number of conv blocks to include
        conv_channel_multipier: int
            Factor to multiply channels within conv blocks
        dropout: float
            Dropout percent after first linear layer
        img_features_count: int
            Number of features to derive from images
        final_fc_size: 
            Size of final fc before going to size of 1
        """
       
        super(Basic3DCNN_with_additional_inputs, self).__init__()

        self.conv_block_list = nn.ModuleList()

        self.conv_block_list = _create_conv_block_set(
            conv_block_depth, 
            input_channels, 
            conv_channel_multipier
        )      

        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.Linear(1024, img_features_count)
        self.fc3 = nn.Linear(img_features_count + num_additional_inputs, final_fc_size)

        self.fc4 = nn.Linear(final_fc_size, out_classes)

        self.fc1_relu = nn.LeakyReLU()
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2_relu = nn.LeakyReLU()
        self.fc2_bn = nn.BatchNorm1d(img_features_count)
        self.fc3_relu = nn.LeakyReLU()
        # Note: no BN for this layer as loss is calculated as NaN if included.

        self.drop = nn.Dropout(p = dropout)

    def forward(self, x, additional_inputs):

        for i in range(len(self.conv_block_list)):
            x = self.conv_block_list[i](x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc1_bn(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.fc2_relu(x)
        x = self.fc2_bn(x)

        x = torch.cat([x, additional_inputs], axis=1)

        x = self.fc3(x)
        x = self.fc3_relu(x)

        x = self.fc4(x)

        return x

class CombineNet(nn.Module):
    # Used for combining output from another network with tabular data

    def __init__(
        self,
        net,
        net_output_len,
        num_additional_inputs, 
        out_classes = 1,
        final_fc_layer_size = 32
    ):
        """
        Parameters
        ----------
        net: nn.Module
            Network to get outputs from
        net_output_len: int
            Length of output from network
        num_additional_inputs: int
            Number of additional inputs to include
        out_classes: int
            Number of classes to predict
        final_fc_size: 
            Size of final fc before going to size of 1
        """
       
        super(CombineNet, self).__init__()

        self.net = net

        self.fc1 = nn.Linear(net_output_len + num_additional_inputs, final_fc_layer_size)
        self.fc2 = nn.Linear(final_fc_layer_size, out_classes)

        self.fc1_relu = nn.LeakyReLU()

    def forward(self, x, additional_inputs):

        x = self.net(x)

        x = torch.cat([x, additional_inputs], axis=1)

        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc2(x)

        return x

class EnsembleModel(nn.Module):
    # Ensemble model
    # Can ensemble both CNN (models that take imaging data) and LR (takes tabular)
    # Does not yet allow for models that take both data

    def __init__(
        self,
        cnn_model_list,
        cnn_image_index_list,
        lr_model_list,
        out_classes = 1
    ):
        """
        Initializes the model. 
        When imaging data is loaded, it is loaded as a multi-channel image.
        The index list is to allow for the CNN to decide which channel to use. 

        It is assumed that LR models take all tabular data. 
        
        Parameters
        ----------
        cnn_model_list: [nn.Module, nn.Module, ...]
            List of CNN models
        cnn_image_index_list: [int, int, ...]
            Length of output from network
        lr_model_list: [nn.Module, nn.Module, ...]
            List of LR models
        out_classes: int
            Number of classes to predict
        """

        super().__init__()

        # To allow for undetermined number of models to be included, will use a module list
        self.cnn_model_list = nn.ModuleList(cnn_model_list)
        self.lr_model_list = nn.ModuleList(lr_model_list)
        self.cnn_image_index_list = cnn_image_index_list

        # Classifier that will predict based on all included models
        self.classifier = nn.Linear(len(cnn_model_list) + len(lr_model_list), out_classes)

    def forward(self, images, additional_inputs):
        model_outputs = []

        for i in range(len(self.cnn_model_list)):
            # Get appropriate channel with an extra dim to feed model
            out = self.cnn_model_list[i](torch.unsqueeze(images[:, self.cnn_image_index_list[i]], 1))
            model_outputs.append(out)

        for model in self.lr_model_list:
            out = model(additional_inputs)
            model_outputs.append(out)

        model_outputs = torch.cat(model_outputs, axis=1)
        out = self.classifier(model_outputs)

        return out
    
class DecisionFusionEnsemble(nn.Module):
    # Decision fusion ensemble
    # Similar to ensemble, but no linear layer after predictions
    # Averages predictions; if weights are frozen no training is needed

    def __init__(
        self,
        cnn_model_list,
        cnn_image_index_list,
        lr_model_list
    ):
        """
        Initializes the model. 
        When imaging data is loaded, it is loaded as a multi-channel image.
        The index list is to allow for the CNN to decide which channel to use. 

        It is assumed that LR models take all tabular data. 
        
        Parameters
        ----------
        cnn_model_list: [nn.Module, nn.Module, ...]
            List of CNN models
        cnn_image_index_list: [int, int, ...]
            Length of output from network
        lr_model_list: [nn.Module, nn.Module, ...]
            List of LR models
        """

        super().__init__()

        # To allow for undetermined number of models to be included, will use a module list
        self.cnn_model_list = nn.ModuleList(cnn_model_list)
        self.lr_model_list = nn.ModuleList(lr_model_list)
        self.cnn_image_index_list = cnn_image_index_list


    def forward(self, images, additional_inputs):
        model_outputs = []

        for i in range(len(self.cnn_model_list)):
            # Get appropriate channel with an extra dim to feed model
            out = self.cnn_model_list[i](torch.unsqueeze(images[:, self.cnn_image_index_list[i]], 1))
            model_outputs.append(out)

        for model in self.lr_model_list:
            out = model(additional_inputs)
            model_outputs.append(out)

        model_outputs = torch.cat(model_outputs, axis=1)
        out = torch.unsqueeze(torch.mean(model_outputs, dim=1), 1)

        return out

class SimpleLogisticRegressionModel(nn.Module):
    # Simple LR model
    def __init__(
        self,
        input_dim,
        out_classes = 1
    ):
        super().__init__()

        self.fc = nn.Linear(input_dim, out_classes)

    def forward(self, x):
        out = self.fc(x)

        return out

def create_pretrained_medical_resnet(
    pretrained_path,
    model_constructor = resnet18,
    spatial_dims = 3,
    n_input_channels = 1,
    num_classes = 1,
    **kwargs_monai_resnet
):
    """
    Constructor for MONAI ResNet module loading MedicalNet weights.
    Slightly modified from: https://github.com/Borda/kaggle_vol-3D-classify/tree/main

    See:
    - https://github.com/Project-MONAI/MONAI
    - https://github.com/Borda/MedicalNet
    """
    net = model_constructor(
        pretrained=False,
        spatial_dims=spatial_dims,
        n_input_channels=n_input_channels,
        num_classes=num_classes,
        **kwargs_monai_resnet
    )
    net_dict = net.state_dict()
    pretrain = torch.load(pretrained_path)
    pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
    missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})
    print(f"missing in pretrained: {len(missing)}")
    inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})
    print(f"inside pretrained: {len(inside)}")
    unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})
    print(f"unused pretrained: {len(unused)}")
    assert len(inside) > len(missing)
    assert len(inside) > len(unused)

    pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net.load_state_dict(pretrain['state_dict'], strict=False)
    return net, inside
