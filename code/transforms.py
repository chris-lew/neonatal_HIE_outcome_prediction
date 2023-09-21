# Various transforms used throughout model training and development

import random
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import rising.transforms as rtr
import torch
import torch.nn.functional as F
from rising.random import DiscreteParameter
from torch import Tensor


class RandomAffine(rtr.BaseAffine):
    """Base Affine with random parameters for scale, rotation and translation
    taken from this notebooks: https://github.com/PhoenixDL/rising/blob/master/notebooks/lightning_segmentation.ipynb
    """

    def __init__(
        self,
        scale_range: tuple,
        rotation_range: tuple,
        translation_range: tuple,
        degree: bool = True,
        image_transform: bool = True,
        keys: Sequence = ('data', ),
        grad: bool = False,
        output_size: Optional[tuple] = None,
        adjust_size: bool = False,
        interpolation_mode: str = 'nearest',
        padding_mode: str = 'zeros',
        align_corners: bool = False,
        reverse_order: bool = False,
        **kwargs
    ):
        super().__init__(
            scale=None,
            rotation=None,
            translation=None,
            degree=degree,
            image_transform=image_transform,
            keys=keys,
            grad=grad,
            output_size=output_size,
            adjust_size=adjust_size,
            interpolation_mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
            reverse_order=reverse_order,
            **kwargs
        )

        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range

    def assemble_matrix(self, **data) -> torch.Tensor:
        ndim = data[self.keys[0]].ndim - 2

        if self.scale_range is not None:
            self.scale = [random.uniform(*self.scale_range) for _ in range(ndim)]

        if self.translation_range is not None:
            self.translation = [random.uniform(*self.translation_range) for _ in range(ndim)]

        if self.rotation_range is not None:
            if ndim == 3:
                self.rotation = [random.uniform(*self.rotation_range) for _ in range(ndim)]
            elif ndim == 1:
                self.rotation = random.uniform(*self.rotation_range)

        return super().assemble_matrix(**data)
    

TRAIN_TRANSFORMS_ROT = [
    rtr.Rot90((0, 1, 2), keys=["image"]),
]

TRAIN_TRANSFORMS_ROT_PROB = [
    rtr.Rot90((0, 1, 2), keys=["image"], p=0.75),
]

TRAIN_TRANSFORMS_MIRROR = [
    rtr.Mirror(dims=DiscreteParameter([0, 1, 2]), keys=["image"]),
]

TRAIN_TRANSFORMS_MIRROR_PROB = [
    rtr.Mirror(dims=DiscreteParameter([0, 1, 2]), keys=["image"], p=0.75),
]

TRAIN_TRANSFORMS_ROT_MIRROR_PROB = [
    rtr.Rot90((0, 1, 2), keys=["image"], p=0.75),
    rtr.Mirror(dims=DiscreteParameter([0, 1, 2]), keys=["image"], p=.75),
]

TRAIN_TRANSFORMS_AFFINE_ORIGINAL = [
    RandomAffine(scale_range=(0.9, 1.1), rotation_range=(-10, 10), translation_range=(-0.1, 0.1), keys=['image'])
]

TRAIN_TRANSFORMS_AFFINE_SMALL = [
    RandomAffine(scale_range=(0.95, 1.05), rotation_range=(-5, 5), translation_range=(-0.05, 0.05), keys=['image'])
]

TRAIN_TRANSFORMS_AFFINE_LARGE = [
    RandomAffine(scale_range=(0.80, 1.20), rotation_range=(-20, 20), translation_range=(-0.2, 0.2), keys=['image'])
]

TRAIN_TRANSFORMS_ALL = [
    rtr.Rot90((0, 1, 2), keys=["image"], p=0.5),
    rtr.Mirror(dims=DiscreteParameter([0, 1, 2]), keys=["image"], p=0.5),
    RandomAffine(scale_range=(0.9, 1.1), rotation_range=(-10, 10), translation_range=(-0.1, 0.1), keys=['image'], p=0.5)
]

TRAIN_TRANSFORMS_PLUS_NOISE_SMALL = [
    rtr.Rot90((0, 1, 2), keys=["image"], p=0.5),
    rtr.Mirror(dims=DiscreteParameter([0, 1, 2]), keys=["image"], p=0.5),
    RandomAffine(scale_range=(0.9, 1.1), rotation_range=(-10, 10), translation_range=(-0.1, 0.1), keys=['image'], p=0.5),
    rtr.GaussianNoise(.2, .1, keys=['image'])
]

TRAIN_TRANSFORMS_PLUS_NOISE_MED = [
    rtr.Rot90((0, 1, 2), keys=["image"], p=0.5),
    rtr.Mirror(dims=DiscreteParameter([0, 1, 2]), keys=["image"], p=0.5),
    RandomAffine(scale_range=(0.9, 1.1), rotation_range=(-10, 10), translation_range=(-0.1, 0.1), keys=['image'], p=0.5),
    rtr.GaussianNoise(.5, .25, keys=['image'])
]

TRAIN_TRANSFORMS_PLUS_NOISE_BIG = [
    rtr.Rot90((0, 1, 2), keys=["image"], p=0.5),
    rtr.Mirror(dims=DiscreteParameter([0, 1, 2]), keys=["image"], p=0.5),
    RandomAffine(scale_range=(0.9, 1.1), rotation_range=(-10, 10), translation_range=(-0.1, 0.1), keys=['image'], p=0.5),
    rtr.GaussianNoise(1, .5, keys=['image'])
]