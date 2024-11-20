import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders.mr_us_dataset import DiffusorSampleSurfDataModule, DiffusorSampleDataModule
from transforms.mr_transforms import VolumeTrainTransforms, VolumeEvalTransforms
# from callbacks.logger import ImageLoggerLotusNeptune

from nets import us_simu
from callbacks import logger

import lightning as L

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
# from pytorch_lightning.strategies.ddp import DDPStrategy
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.loggers import NeptuneLogger
# from pytorch_lightning.plugins import MixedPrecisionPlugin

import pickle

import SimpleITK as sitk


def main(args):
    
    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_val = pd.read_csv(args.csv_valid)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid)

    NN = getattr(us_simu, args.nn)    
    model = NN(**vars(args))

    train_transform = VolumeTrainTransforms()
    valid_transform = VolumeEvalTransforms()
    surf_data = DiffusorSampleDataModule(df_train, df_val, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=4, img_column="img_path", train_transform=train_transform, valid_transform=valid_transform, drop_last=False, num_samples_train=args.num_samples_train, num_samples_val=args.num_samples_val)

    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True
        
    )

    callbacks.append(checkpoint_callback)

    if args.monitor:
        checkpoint_callback_d = ModelCheckpoint(
            dirpath=args.out,
            filename='{epoch}-{' + args.monitor + ':.2f}',
            save_top_k=5,
            monitor=args.monitor,
            save_last=True
            
        )

        callbacks.append(checkpoint_callback_d)


    if args.use_early_stopping:
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")
        callbacks.append(early_stop_callback)

    
    logger_neptune = None

    if args.neptune_tags:
        logger_neptune = NeptuneLogger(
            project='ImageMindAnalytics/us-simu',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN'],
            log_model_checkpoints=False
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
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    
    trainer.fit(model, datamodule=surf_data, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=2)

    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="USAEReconstruction")        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--img_column', type=str, default='img_path', help='Column name for image')  
    input_group.add_argument('--num_samples_train', type=int, default=1000, help='Number of samples for training')  
    input_group.add_argument('--num_samples_val', type=int, default=10, help='Number of samples for validation')  
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    output_group.add_argument('--use_early_stopping', help='Use early stopping criteria', type=int, default=0)
    output_group.add_argument('--monitor', help='Additional metric to monitor to save checkpoints', type=str, default=None)
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, default="USAEReconstructionNeptuneLogger")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=5)

    args, unknownargs = parser.parse_known_args()

    NN = getattr(us_simu, args.nn)    
    NN.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
