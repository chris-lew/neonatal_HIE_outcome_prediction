# Preprocessing functions

import SimpleITK as sitk
from pathlib import Path
import os
import nibabel as nib
import numpy as np
from itertools import product
from tqdm import tqdm
import torchio as tio
import traceback

def sitk_N4B_correction(img_path):
    """
    Uses SimpleITK method of N4Bias correction

    Parameters
    ----------
    img_path: str or pathlib.Path
        Path to image stored as .nii data

    Returns
    -------
    np.array
        Corrected image

    """

    input_img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    mask_image = sitk.OtsuThreshold(input_img, 0, 1, 200)

    corrected_image = corrector.Execute(input_img, mask_image)
    corrected_image_arr = sitk.GetArrayFromImage(corrected_image)

    return corrected_image_arr

def normalize_image(image_array, min_cutoff = 0.001, max_cutoff = 0.001):
    """
    Normalize the intensity of an image array by cutting off min and max values 
    to a certain percentile and set all values above/below that percentile to 
    the new max/min. 

    Parameters
    ----------
    image_array: np.array
        3D numpy array constructed from dicom files
    min_cutoff: float
        Minimum percentile of image to keep. (0.1% = 0.001)
    max_cutoff: float
        Maximum percentile of image to keep. (0.1% = 0.001)

    Returns
    -------
    np.array
        Normalized image

    """

    # Sort image values
    sorted_array = np.sort(image_array.flatten())

    # Find %ile index and get values
    min_index = int(len(sorted_array) * min_cutoff)
    min_intensity = sorted_array[min_index]

    max_index = int(len(sorted_array) * min_cutoff) * -1
    max_intensity = sorted_array[max_index]

    # Normalize image and cutoff values
    image_array = (image_array - min_intensity) / \
        (max_intensity - min_intensity)
    image_array[image_array < 0.0] = 0.0
    image_array[image_array > 1.0] = 1.0

    return image_array

def zscore_image(image_array):
    """
    Convert intensity values in an image to zscores:
    zscore = (intensity_value - mean) / standard_deviation

    Parameters
    ----------
    image_array: np.array
        3D numpy array constructed from dicom files
        
    Returns
    -------
    np.array
        Image with zscores for values

    """

    image_array = (image_array - np.mean(image_array)) / np.std(image_array)

    return image_array

def range_scale_image(
        image_array, 
        range_scale_params = (0, 3000e-6)
    ):
    """
    Scales the intensity range of an image array to a normalized range between 0 and 1
    according to minimum and maximum values given. 

    For ADC:
    0 - 3000e-6
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4524324/ shows high values around 2300e-6

    For FA:
    0 - 0.6
    https://iovs.arvojournals.org/article.aspx?articleid=2738360 high values around 0.3
    https://link.springer.com/article/10.1007/s00247-007-0626-7 high values around 0.45

    Parameters
    ----------
        image_array (numpy.ndarray): The input image array.
        range_scale_params (float, float): The minimum and maximum intensity value to scale from (default: 0, 3000e-6).

    Returns
    -------
        numpy.ndarray: The scaled image array.
        
    """

    min_intensity, max_intensity = range_scale_params

    image_array = (image_array - min_intensity) / \
        (max_intensity - min_intensity)
    image_array[image_array < 0.0] = 0.0
    image_array[image_array > 1.0] = 1.0

    return image_array

def crop_image(
    image_array, crop_params
):
    """
    Crops image along 3 axes according to parameters

    Parameters
    ----------
        image_array (numpy.ndarray): The input image array.
        crop_params (int, int, int, int, int, int): x_min, x_max, y_min, y_max, z_min, z_max

    Returns
    -------
        numpy.ndarray: Cropped image
        
    """

    x_min, x_max, y_min, y_max, z_min, z_max = crop_params
    return image_array[x_min:x_max, y_min:y_max, z_min:z_max]

def tight_crop(image_array, threshold=50, pad=2):
    """
    Crops image tightly along 3 axes by removing background space below a threshold

    Parameters
    ----------
        image_array (numpy.ndarray): The input image array.
        threshold (int): Minimum number of non-zero values along a plane to 
            consider a plane as NOT part of the background
        pad (int): after background is determined, how much to pad image to include
            a small area of the background

    Returns
    -------
        numpy.ndarray: Cropped image
        
    """

    # First find the number of occurrence of values that are not equal to zero for each axis
    x_sum = image_array.astype(bool).sum(axis=(1,2))
    y_sum = image_array.astype(bool).sum(axis=(0,2))
    z_sum = image_array.astype(bool).sum(axis=(0,1))

    # Now need the first index that has occurrences greater than our threshold
    x_min = np.argmax(x_sum > threshold) - pad
    x_min = x_min if x_min >= 0 else 0

    x_max = len(x_sum) - np.argmax(x_sum[::-1] > threshold) - 1 + pad
    x_max = x_max if x_max <= image_array.shape[0] else image_array.shape[0]

    y_min = np.argmax(y_sum > threshold) - pad
    y_min = y_min if y_min >= 0 else 0

    y_max = len(y_sum) - np.argmax(y_sum[::-1] > threshold) - 1 + pad
    y_max = y_max if y_max <= image_array.shape[1] else image_array.shape[1]

    z_min = np.argmax(z_sum > threshold) - pad
    z_min = z_min if z_min >= 0 else 0

    z_max = len(z_sum) - np.argmax(z_sum[::-1] > threshold) - 1 + pad
    z_max = z_max if z_max <= image_array.shape[2] else image_array.shape[2]

    return image_array[x_min:x_max, y_min:y_max, z_min:z_max]

def reshape_image(image_array, shape):
    """
    Resizes image using torchio to a different shape

    Parameters
    ----------
        image_array (numpy.ndarray): The input image array.
        shape (int, int, int): Shape to resize image to 

    Returns
    -------
        numpy.ndarray: Reshape image
        
    """
    transform = tio.transforms.Resize(shape)
    image_array = transform(np.expand_dims(image_array, axis=0)).squeeze()
    return image_array


def preprocess_image_data(
    data_dir,
    sequence_suffixes,
    base_save_dir,
    subjects_to_process,
    dir_suffix,
    preprocess_methods,
    reshape_shape = None,
    crop_params = None,
    range_scale_params = None,
    rot90_params = None,
    check_shape = None,
    skip_missing = False,
    skip_errors = False
):
    """
    Processes imaging data for subjects across different sequences.
    For this function, data must be stored in the following format:

    data_dir / {subject_name} / {subject_name}_{sequence_1_name}.nii.gz
    data_dir / {subject_name} / {subject_name}_{sequence_2_name}.nii.gz
    ...

    Processed data will be served as:

    base_save_dir / {sequence_name}_{dir_suffix} / {subject_name}.npy
    
    preprocess methods include and are limited to:
        ['N4BC', 'tight_crop', 'crop', 'range_scale', 'intensity_norm', 'z_score', 'reshape', 'rot90']
        N4BC: N4 bias correction
        tight_crop OR crop: crop method to use
        range_scale OR intensity_norm OR z_score OR (intensity_norm AND z_score): methods to scale/normalize intensity values
        reshape: reshape images
        rot90: rotate by 90 degrees
    
    Parameters
    ----------
        data_dir: str or pathlib.Path
            Base directory for data
        sequence_suffixes: [str, str, ...]
            List of sequence suffixes
        base_save_dir: 
            Base directory to save processed
        subjects_to_process: [str, str, str, ...]
            List of subjects to process; only the list subjects will be processed
        dir_suffix: str
            Suffix appended to sequence when saving data
        preprocess_methods: [str, str, str ...]
            Methods to use when processing, see above
        reshape_shape: (int, int, int)
            Shape to reshape to, if included
        crop_params: (int, int, int, int, int, int)
            Area to crop to, if included
        range_scale_params: (float, float)
            Range to scale to, if included
        rot90_params: (int, int)
            Axes to rotate by 90 degrees
        check_shape: (int, int, int)
            Will ensure that all loaded images are this shape
        skip_missing: bool
            If True, will not raise error if there are missing files
        skip_errors: bool
            If True, will not raise error if there are errors in processing

    Returns
    -------
        None
        
    """

    # Convert to paths
    data_dir = Path(data_dir)
    base_save_dir = Path(base_save_dir)

    # Make save dirs if needed
    if not os.path.isdir(base_save_dir):
        os.mkdir(base_save_dir)

    for sequence in sequence_suffixes:
        if len(dir_suffix) > 0:
            dir_suffix = '_' + dir_suffix

        if not os.path.isdir(base_save_dir / f'{sequence}{dir_suffix}'):
            os.mkdir(base_save_dir / f'{sequence}{dir_suffix}')

    # Check that methods are correct
    all_methods = ['N4BC', 'tight_crop', 'crop', 'range_scale', 'intensity_norm', 'z_score', 'reshape', 'rot90']
    for method in preprocess_methods:
        assert method in all_methods, f'method must be one of the following {all_methods}'

    # Get list of subjects
    subject_list = [x for x in os.listdir(data_dir) if os.path.isdir(data_dir / x)]
    if subjects_to_process:
        subject_list = [x for x in subject_list if x in subjects_to_process]

    # Iterate
    for subject, sequence in tqdm(product(subject_list, sequence_suffixes), 
                                  total= len(subject_list) * len(sequence_suffixes)):
        
        # Get path to data
        img_path = data_dir / subject / f'{subject}_{sequence}.nii.gz'

        if skip_missing:
            if not os.path.isfile(img_path):
                print('Missing file:', img_path)
                continue

        if check_shape:
            img = nib.load(img_path)
            if img.get_fdata().shape != check_shape:
                print(f'wrong shape for: {subject}; {sequence}')

        try:
            # N4Bias correction or just load image
            if 'N4BC' in preprocess_methods:
                img = sitk_N4B_correction(img_path)
            else:
                img = nib.load(img_path).get_fdata()

            # Crop method if included
            if 'tight_crop' in preprocess_methods:
                img = tight_crop(img)
            elif 'crop' in preprocess_methods:
                img = crop_image(img, crop_params)

            # Scaling/norm method if including
            if 'range_scale' in preprocess_methods:
                img = range_scale_image(img, range_scale_params)
            elif 'intensity_norm' in preprocess_methods and 'z_score' in preprocess_methods:
                img = zscore_image(normalize_image(img))
            elif 'intensity_norm' in preprocess_methods:
                img = normalize_image(img)
            elif 'z_score' in preprocess_methods:
                img = zscore_image(img)

            # Reshape if included
            if 'reshape' in preprocess_methods:
                img = reshape_image(img, reshape_shape)

            if 'rot90' in preprocess_methods:
                img = np.rot90(img, axes=rot90_params)

            # Save data
            img = img.astype('float32')
            np.save(base_save_dir / f'{sequence}{dir_suffix}' / subject, img)

        # Handling errors
        except Exception as error:
            print(f'Error for {subject}; {sequence}')
            traceback.print_exc()

            if skip_errors:
                print(error)
            else:
                raise error
