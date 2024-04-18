import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from torchvision import transforms as T
from torch.utils.data import DataLoader

from loaders.ultrasound_dataset import USDatasetV2
from transforms.ultrasound_transforms import RealUSTrainTransforms, RealUSEvalTransforms, RealUSTrainTransformsV2, RealUSEvalTransformsV2
import monai

from nets import cut
import SimpleITK as sitk

import pickle

import nrrd


def main(args):

    if(os.path.splitext(args.csv)[1] == ".csv"):
        df = pd.read_csv(args.csv)
    else:
        df = pd.read_parquet(args.csv)

    NN = getattr(cut, args.nn)    
    model = NN.load_from_checkpoint(args.model)
    
    model.cuda()
    model.eval()
    
    valid_transform = RealUSEvalTransforms()

    
    test_ds = monai.data.Dataset(data=USDatasetV2(df, mount_point=args.mount_point, img_column=args.img_column, transform=valid_transform))
    test_dl = DataLoader(test_ds, batch_size=1, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            
            fake_us = model(batch.cuda())
            fake_us = fake_us.clip(min=0, max=1.0)*255
            row = df.iloc[idx]
            out_path = os.path.join(args.out, os.path.basename(args.model), row[args.img_column])
            out_dir = os.path.dirname(out_path)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print('Writing:', out_path)
            nrrd.write(out_path, fake_us.squeeze().cpu().numpy().astype(np.ubyte), index_order='C')


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='LotusCut prediction')
    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="Cut")        
    input_group.add_argument('--model', help='Model with trained weights', type=str, required=True)
           
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default='img_path', help='Column name for image')    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--use_random', help='Use random transform', type=int, default=0)

    args = parser.parse_args()

    main(args)
