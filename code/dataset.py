# Defines dataset class

import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd

class HEAL_Dataset(Dataset):
    """
    Dataset that handles imaging and tabular data. All data is kept in memory
    to improve speed. 
    """
    def __init__(
        self, 
        image_dirs, 
        subject_list = None,
        label_csv_path = None,
        subject_col = None,
        label_col = None,
        addtional_feature_cols = None
    ):
        """
        Initializing class. If label_csv_path is included, then it is assumed
        that dataset will not be used for only prediction and that labels are included.
        See class below if only tabular data is desired. 

        Parameters
        ----------
        image_dirs: [str, str, ...]
            List of paths to directories with images that will be used for each channel
        subject_list: [str, str, ...]
            List of subjects to use from directories; if not included will use all images
        label_csv_path: str
            Path to csv where labels and tabular data can be found
        subject_col: str
            Column of csv where the subject ID is found
        label_col: str
            Column of csv where label is found
        addtional_feature_cols: [str, str, ...]
            Columns of additional tabular data to be included
        """

        self.transforms = transforms

        # For later use; seeing if labels are included for training
        self.input_only = not bool(label_csv_path)

        if addtional_feature_cols:
            self.include_additional_features = True
        else:
            self.include_additional_features = False

        # Convert to paths
        image_dirs = [Path(x) for x in image_dirs]

        # Make subject list if not included
        if not subject_list:
            subject_list = [x.split[-1] for x in os.listdir(image_dirs[0]) if x.split[-1] == '.npy']

        # Store images
        self.image_arrays = []

        # Iterate through
        for subject in subject_list:
            
            # If only one channel, then no need to iterate and stack
            if len(image_dirs) == 1:
                subject_image_array = np.load(image_dirs[0] / f'{subject}.npy')
                subject_image_array = np.expand_dims(subject_image_array, axis=0)
                self.image_arrays.append(subject_image_array)

            # If multiple channels, iterate and stack
            else:
                subject_image_array = []
                for image_dir in image_dirs:
                    subject_channel = np.load(image_dirs[0] / f'{subject}.npy')
                    subject_image_array.append(subject_channel)
                subject_image_array = np.stack(subject_image_array)
                self.image_arrays.append(subject_image_array)

        self.image_arrays = np.stack(self.image_arrays)

        # Handling csv
        if label_csv_path:

            # Load csv
            label_csv_path = Path(label_csv_path)
            label_df = pd.read_csv(label_csv_path)

            # Keep only included subjects
            label_df = label_df.loc[label_df[subject_col].isin(subject_list)]

            # Reorder so its the same order as the list
            label_df[subject_col] = pd.Categorical(
                label_df[subject_col], ordered=True, categories=subject_list
            )
            label_df = label_df.sort_values(subject_col)

            self.labels = np.array(label_df[label_col])

            if self.include_additional_features:
                self.additional_features = np.array(label_df[addtional_feature_cols]).astype('float32')

        print(f'Loaded {len(self.image_arrays)} images each with shape {self.image_arrays.shape[1:]}')

    def __len__(self):
        return len(self.image_arrays)
    
    def __getitem__(self, i):

        # Get image array based on index
        image_array = torch.from_numpy(np.array(self.image_arrays[i]))

        # Get features and labels if needed
        if self.include_additional_features:
            additional_features_array = torch.from_numpy(self.additional_features[i])
        if not self.input_only:
            label = torch.from_numpy(np.array(self.labels[i]))

        # Now return data based on what is needed
        if self.input_only and not self.include_additional_features:
            return {'image': image_array}
        
        elif self.input_only and self.include_additional_features:
            return {
                'image': image_array, 
                'additional_features': additional_features_array
            }
        
        elif not self.input_only and not self.include_additional_features:
            return {
                'image': image_array, 
                'label': label
            }
        
        elif not self.input_only and self.include_additional_features:
            return {
                'image': image_array, 
                'additional_features': additional_features_array,
                'label': label
            }


class HEAL_Dataset_Tabular(Dataset):
    def __init__(
        self, 
        addtional_feature_cols,
        subject_list,
        label_csv_path = None,
        subject_col = None,
        label_col = None,
        input_only = False
    ):
        """
        Similar to base class but only for tabular data. Therefore it requires
        additional feature cols. 

        Parameters
        ----------
        addtional_feature_cols: [str, str, ...]
            Columns of additional tabular data to be included
        subject_list: [str, str, ...]
            List of subjects to use from csv
        label_csv_path: str
            Path to csv where labels and tabular data can be found
        subject_col: str
            Column of csv where the subject ID is found
        label_col: str
            Column of csv where label is found
        """
        self.input_only = input_only

        # Handle csv data
        if label_csv_path:

            # Load csv
            label_csv_path = Path(label_csv_path)
            label_df = pd.read_csv(label_csv_path)

            # Keep only included subjects
            label_df = label_df.loc[label_df[subject_col].isin(subject_list)]

            # Reorder so its the same order as the list
            label_df[subject_col] = pd.Categorical(
                label_df[subject_col], ordered=True, categories=subject_list
            )
            label_df = label_df.sort_values(subject_col)

            if not input_only:
                self.labels = np.array(label_df[label_col])

            self.additional_features = np.array(label_df[addtional_feature_cols]).astype('float32')


    def __len__(self):
        return len(self.additional_features)
    
    def __getitem__(self, i):

        additional_features_array = torch.from_numpy(self.additional_features[i])

        if not self.input_only:
            label = torch.from_numpy(np.array(self.labels[i]))

        # Now return data based on what is needed
        if self.input_only:
            return {
                'additional_features': additional_features_array
            }
        
        else:
            return {
                'additional_features': additional_features_array,
                'label': label
            }
        

