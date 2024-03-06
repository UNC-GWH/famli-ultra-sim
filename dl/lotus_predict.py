import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from torchvision import transforms as T
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset import LotusDataset
from transforms.ultrasound_transforms import LotusEvalTransforms
import monai
from callbacks import logger as LOGGER

from nets import lotus
import SimpleITK as sitk

import pickle

import nrrd


def main(args):

    if(os.path.splitext(args.csv)[1] == ".csv"):
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_parquet(args.csv)

    NN = getattr(lotus, args.nn)    
    model = NN.load_from_checkpoint(args.model)
    model.eval()

    model2 = NN(num_labels=model.hparams.num_labels, grid_w=256, grid_h=256, center_x=128, center_y=-30, r1=20, r2=215, theta=np.pi/4.5, alpha_coeff_boundary_map=model.hparams.alpha_coeff_boundary_map, tgc=model.hparams.tgc, beta_coeff_scattering=model.hparams.beta_coeff_scattering)
    model2.acoustic_impedance_dict = model.acoustic_impedance_dict
    model2.attenuation_dict = model.attenuation_dict
    model2.mu_0_dict = model.mu_0_dict
    model2.mu_1_dict = model.mu_1_dict
    model2.sigma_0_dict = model.sigma_0_dict
    model2.eval()

    
    valid_transform = LotusEvalTransforms()

    test_ds = monai.data.Dataset(data=LotusDataset(df, args.mount_point, img_column=args.img_column, seg_column=args.seg_column), transform=valid_transform)
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    t = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256), T.Lambda(lambda x: torch.clip(x*255, min=0, max=255))])

    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            row = df.iloc[idx]
            us = model2(batch['seg'])

            if isinstance(us, tuple):
                us = us[0]

            if args.use_out_transform:
                us = t(us)

            seg_path = row[args.seg_column]

            out_seg_path = os.path.join(args.out, os.path.basename(args.model), seg_path)
            out_seg_dir = os.path.dirname(out_seg_path)

            if not os.path.exists(out_seg_dir):
                os.makedirs(out_seg_dir)
            print('Writing:', out_seg_path)
            nrrd.write(out_seg_path, us.squeeze().cpu().numpy(), index_order='C')
        


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Lotus prediction')
    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="UltrasoundRendering")        
    input_group.add_argument('--model', help='Model with trained weights', type=str, required=True)
           
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default='img_path', help='Column name for image')  
    input_group.add_argument('--seg_column', type=str, default='seg_path', help='Column name for labeled/seg image')  
    input_group.add_argument('--use_out_transform', type=int, default=1, help='Use the output transform of pad and crop to match real US')  

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")

    args = parser.parse_args()

    main(args)
