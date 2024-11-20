import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders.ultrasound_dataset import LotusDataModule
from loaders.mr_us_dataset import VolumeSlicingProbeParamsDataset
from transforms.ultrasound_transforms import LotusEvalTransforms, LotusTrainTransforms
from loaders.mr_us_dataset import DiffusorSampleDataModule
from transforms.mr_transforms import VolumeTrainTransforms, VolumeEvalTransforms
# from callbacks.logger import ImageLoggerLotusNeptune

from nets import lotus
from callbacks import logger


from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger


import pickle

import SimpleITK as sitk


def main(args):

    # if(os.path.splitext(args.csv_train_params)[1] == ".csv"):
    #     df_train_params = pd.read_csv(args.csv_train_params)
    #     df_val_params = pd.read_csv(args.csv_valid_params)   
    # else:
    #     df_train_params = pd.read_parquet(args.csv_train_label)
    #     df_val_params = pd.read_parquet(args.csv_valid_label)   

    # if(os.path.splitext(args.csv_train_us)[1] == ".csv"):
    #     df_train_us = pd.read_csv(args.csv_train_us)
    #     df_val_us = pd.read_csv(args.csv_valid_us)   
    # else:
    #     df_train_us = pd.read_parquet(args.csv_train_us)
    #     df_val_us = pd.read_parquet(args.csv_valid_us)   

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_val = pd.read_csv(args.csv_valid)
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid) 
        df_test = pd.read_parquet(args.csv_test)  

    NN = getattr(lotus, args.nn)    
    model = NN(**vars(args))

    if args.init_params:
        df_params = pd.read_csv(args.init_params)
        model.init_params(df_params)

    # train_transform = LotusTrainTransforms()
    # valid_transform = LotusEvalTransforms()
    # lotus_data = LotusDataModule(df_train, df_val, df_test, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=4, img_column="img_path", seg_column="seg_path", train_transform=train_transform, valid_transform=valid_transform, test_transform=valid_transform, drop_last=False)
    train_transform = VolumeTrainTransforms()
    valid_transform = VolumeEvalTransforms()
    diffusor_data = DiffusorSampleDataModule(df_train, df_val, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=4, img_column="img_path", train_transform=train_transform, valid_transform=valid_transform, drop_last=False, num_samples_train=500, num_samples_val=20)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    callbacks=[early_stop_callback, checkpoint_callback]
    logger_neptune = None

    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/Lotus',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )

        LOGGER = getattr(logger, args.logger)    
        image_logger = LOGGER(log_steps=args.log_steps)
        callbacks.append(image_logger)

    
    trainer = Trainer(
        logger=logger_neptune,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False)
        # detect_anomaly=True
    )
    
    trainer.fit(model, datamodule=diffusor_data, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Lotus training')


    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="UltrasoundRendering")        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')    
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')  
    input_group.add_argument('--img_column', type=str, default='img_path', help='Column name for image')  
    input_group.add_argument('--seg_column', type=str, default='seg_path', help='Column name for labeled/seg image') 
    input_group.add_argument('--init_params', help='Use the dataframe to initialize the mean and std of the diffusor', type=str, default=None)

    input_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    input_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    input_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    input_group.add_argument('--batch_size', help='Batch size', type=int, default=2)
     
    # input_group.add_argument('--labeled_img', required=True, type=str, help='Labeled volume to grap slices from')    
    # input_group.add_argument('--csv_train_us', required=True, type=str, help='Train CSV')
    # input_group.add_argument('--csv_valid_us', required=True, type=str, help='Valid CSV')    

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, nargs="+", default="ImageLoggerLotusNeptune")
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=100)

    args, unknownargs = parser.parse_known_args()

    NN = getattr(lotus, args.nn)    
    NN.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
