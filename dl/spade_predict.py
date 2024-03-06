import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from torchvision import transforms as T
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset import LotusDataset
from transforms.ultrasound_transforms import LotusEvalTransforms, LotusEvalTransformsRandom
import monai
from callbacks import logger as LOGGER

from nets import spade
import SimpleITK as sitk

import pickle

import nrrd


def main(args):

    if(os.path.splitext(args.csv)[1] == ".csv"):
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_parquet(args.csv)

    NN = getattr(spade, args.nn)    
    model = NN.load_from_checkpoint(args.model)
    model.init_grid(args.grid_w, args.grid_h, args.center_x, args.center_y, args.r1, args.r2, args.theta)
    model.cuda()
    model.eval()
    
    if args.use_random:
        valid_transform = LotusEvalTransformsRandom()
    else:
        valid_transform = LotusEvalTransforms()

    test_ds = monai.data.Dataset(data=LotusDataset(df, args.mount_point, img_column=args.img_column, seg_column=args.seg_column), transform=valid_transform)
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            row = df.iloc[idx]
            us = model(batch['seg'].cuda())
            
            if isinstance(us, tuple):
                us = us[0]
            us = torch.clip(us, min=0, max=1.0)
            seg_path = row[args.seg_column]

            out_seg_path = os.path.join(args.out, os.path.basename(args.model), seg_path)
            out_seg_dir = os.path.dirname(out_seg_path)

            if not os.path.exists(out_seg_dir):
                os.makedirs(out_seg_dir)
            print('Writing:', out_seg_path)
            nrrd.write(out_seg_path, us.squeeze().cpu().numpy(), index_order='C')


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='SPADE prediction')
    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="SPADELotus")        
    input_group.add_argument('--model', help='Model with trained weights', type=str, required=True)
           
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default='img_path', help='Column name for image')  
    input_group.add_argument('--seg_column', type=str, default='seg_path', help='Column name for labeled/seg image')

    us_simu = parser.add_argument_group('Ultrasound simulation parameters')
    us_simu.add_argument('--grid_w', help='Grid size for the simulation', type=int, default=256)
    us_simu.add_argument('--grid_h', help='Grid size for the simulation', type=int, default=256)
    us_simu.add_argument('--center_x', help='Position of the circle that creates the transducer', type=float, default=128.0)
    us_simu.add_argument('--center_y', help='Position of the circle that creates the transducer', type=float, default=-30.0)
    us_simu.add_argument('--r1', help='Radius of first circle', type=float, default=20.0)
    us_simu.add_argument('--r2', help='Radius of second circle', type=float, default=215.0)
    us_simu.add_argument('--theta', help='Aperture angle of transducer', type=float, default=np.pi/4.0)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--use_random', help='Use random transform', type=int, default=0)

    args = parser.parse_args()

    main(args)
