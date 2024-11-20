
import math
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as F

# from pl_bolts.transforms.dataset_normalizations import (
#     imagenet_normalization
# )

from monai.transforms import (        
    BorderPad,
    CenterSpatialCrop,
    EnsureChannelFirst,    
    Compose,      
    NormalizeIntensity,      
    RandAffine,       
    RandAxisFlip, 
    RandRotate,
    RandFlip,
    RandZoom,
    RepeatChannel,
    Resize,
    ScaleIntensityRange,
    ScaleIntensity,
    ToTensor, 
    RandAdjustContrast,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandSpatialCrop,
    LoadImaged,        
    EnsureChannelFirstd,
    RandAxisFlipd,
    RandAffined,
    RandRotated,
    RandZoomd,
    Resized,
    ScaleIntensityRanged,
    ToTensord
)

from monai import transforms as monai_transforms

class VolumeTrainTransforms:
    def __init__(self):

        # image augmentation functions
        self.train_transform = Compose(
            [   
                EnsureChannelFirst(channel_dim='no_channel'),
                # RandSpatialCrop((256, 256, 256), random_size=False),
                RandRotate(range_x=math.pi, mode='nearest', prob=1.0, padding_mode='zeros'),
                # RandAffine(prob=0.5, shear_range=(0.1, 0.1), mode='nearest', padding_mode='zeros'),
                RandAxisFlip(prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, mode='nearest', prob=0.5, padding_mode='constant')
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)
    
class VolumeEvalTransforms:
    def __init__(self):

        # image augmentation functions
        self.transform = Compose(
            [   
                EnsureChannelFirst(channel_dim='no_channel'),
            ]
        )

    def __call__(self, inp):
        return self.transform(inp)