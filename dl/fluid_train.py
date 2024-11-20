import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders import ultrasound_dataset as us_dataset

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

torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    

    NN = getattr(us_simu, args.nn)    
    model = NN(**vars(args))


    DATAMODULE = getattr(us_dataset, args.data_module)

    data = DATAMODULE(**vars(args))    


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    callbacks=[early_stop_callback, checkpoint_callback]
    neptune_logger = None

    if args.neptune_tags:
        neptune_logger = NeptuneLogger(
            project='ImageMindAnalytics/us-simu',
            tags=args.neptune_tags,
            api_key=os.environ['NEPTUNE_API_TOKEN']
        )

        LOGGER = getattr(logger, args.logger)    
        image_logger = LOGGER(log_steps=args.log_steps)
        callbacks.append(image_logger)

    
    trainer = Trainer(
        logger=neptune_logger,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=callbacks,
        accelerator='gpu', 
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False)
        # reload_dataloaders_every_n_epochs=1
        # detect_anomaly=True
    )
    
    trainer.fit(model, datamodule=data, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Fluid seg training')

    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="USPCSegmentation")        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    input_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    input_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)

    input_group.add_argument('--data_module', help='Data module type', type=str, default="FluidDataModule")

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, nargs="+", default="FluidLogger")
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=10)

    args, unknownargs = parser.parse_known_args()

    NN = getattr(us_simu, args.nn)    
    NN.add_model_specific_args(parser)


    data_module = getattr(us_dataset, args.data_module)
    parser = data_module.add_data_specific_args(parser)

    args = parser.parse_args()

    main(args)