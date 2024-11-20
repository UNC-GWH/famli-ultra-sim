import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders.ultrasound_dataset import LotusDataModule, USDataModuleV2
from loaders.mr_us_dataset import VolumeSlicingProbeParamsDataset, VolumeSlicingDataModule, ConcatDataModule
from transforms.ultrasound_transforms import VolumeSlicingTrainTransforms, RealUSTrainTransformsV2, RealUSEvalTransformsV2
# from callbacks.logger import ImageLoggerLotusNeptune

from nets import lotus, cut, diffusion
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

    # if(os.path.splitext(args.csv_train_params)[1] == ".csv"):
    #     df_train_params = pd.read_csv(args.csv_train_params)
    #     df_val_params = pd.read_csv(args.csv_valid_params)   
    # else:
    #     df_train_params = pd.read_parquet(args.csv_train_label)
    #     df_val_params = pd.read_parquet(args.csv_valid_label)   

    if(os.path.splitext(args.csv_train_b)[1] == ".csv"):
        df_train_b = pd.read_csv(args.csv_train_b)
        df_val_b = pd.read_csv(args.csv_valid_b)                   
        df_test_b = pd.read_csv(args.csv_test_b)        
    else:
        df_train_b = pd.read_parquet(args.csv_train_b)
        df_val_b = pd.read_parquet(args.csv_valid_b)   
        df_test_b = pd.read_parquet(args.csv_test_b)        

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train)
        df_val = pd.read_csv(args.csv_valid)
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid) 
        df_test = pd.read_parquet(args.csv_test)  

    if(os.path.splitext(args.csv_probe_params)[1] == ".csv"):
        df_probe_params = pd.read_csv(args.csv_probe_params)
    else:
        df_probe_params = pd.read_parquet(args.csv_probe_params)

    NN = getattr(cut, args.nn)    
    model = NN(**vars(args))

    if args.nn2 is not None:
        print("Loading! model 2")
        model2 = getattr(diffusion, args.nn2).load_from_checkpoint(args.model2)
        model2.eval()
        model2.freeze()
        model2 = model2.autoencoderkl
        model.autoencoderkl = model2
        # model.quant_conv_mu = model2.quant_conv_mu
        # model.quant_conv_log_sigma = model2.quant_conv_log_sigma

    if args.init_params:
        df_params = pd.read_csv(args.init_params)
        mean_diffusor_dict = torch.tensor(df_params['mean']).to(torch.float)
        variance_diffusor_dict = torch.tensor(df_params['std']).to(torch.float)
        
        model.USR.init_params(mean_diffusor_dict, variance_diffusor_dict)

    if args.acoustic_params:
        df_params = pd.read_csv(args.acoustic_params)        
        model.USR.init_params(df_params)

    train_transform = VolumeSlicingTrainTransforms()
    n_samples_train = len(df_train)
    n_samples_val = len(df_val)
    volume_slicing_data = VolumeSlicingDataModule(df_train, df_val, df_test, df_probe_params, mount_point=args.mount_point, batch_size=args.batch_size, num_workers=args.num_workers, img_column="img_path", seg_column="seg_path", train_transform=train_transform, drop_last=True, n_samples_train=n_samples_train, n_samples_val=n_samples_val)

    volume_slicing_data.setup()


    train_transform_b = RealUSTrainTransformsV2()
    valid_transform_b = RealUSEvalTransformsV2()
    usdata = USDataModuleV2(df_train_b, df_val_b, df_test_b, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, drop_last=True, train_transform=train_transform_b, valid_transform=valid_transform_b)
    usdata.setup()

    concat_data = ConcatDataModule(volume_slicing_data.train_ds, volume_slicing_data.val_ds, usdata.train_ds, usdata.val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    # concat_data.setup()
    # train_dl = concat_data.train_dataloader()
    # for idx, batch in enumerate(train_dl):
    #     label, us = batch
    #     print("__")
    #     # print(label['img'].shape)
    #     # print(label['seg'].shape)
    #     print(label.shape)
    #     print(us.shape)
    #     print("..")
    # quit()


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
            project='ImageMindAnalytics/Lotus',
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
        strategy=DDPStrategy(find_unused_parameters=False),
        reload_dataloaders_every_n_epochs=1
        # detect_anomaly=True
    )
    
    trainer.fit(model, datamodule=concat_data, ckpt_path=args.model)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Diffusion training')

    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="UltrasoundRendering")        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--nn2', help='Type of neural network. ', type=str, default= None)
    input_group.add_argument('--model2', help='Trained autoencoder model, must have function encode and sampling implemented', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')    
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')  
    input_group.add_argument('--csv_probe_params', required=True, type=str, help='Probe params CSV')

    input_group.add_argument('--init_params', help='Use the dataframe to initialize the mean and std of the diffusor', type=str, default=None)
    input_group.add_argument('--acoustic_params', help='Use the dataframe to initialize the acoustic params', type=str, default=None)

    input_group.add_argument('--csv_train_b', required=True, type=str, help='Train CSV B domain')
    input_group.add_argument('--csv_valid_b', required=True, type=str, help='Valid CSV B domain')    
    input_group.add_argument('--csv_test_b', required=True, type=str, help='Test CSV')  
    input_group.add_argument('--img_column', type=str, default='img_path', help='Column name for image')  
    input_group.add_argument('--seg_column', type=str, default='seg_path', help='Column name for labeled/seg image')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, nargs="+", default="CutLogger")
    log_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    log_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="diffusion")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=100)

    args, unknownargs = parser.parse_known_args()

    NN = getattr(cut, args.nn)    
    NN.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)