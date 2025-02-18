import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch
from torch.distributed import is_initialized, get_rank

from loaders.ultrasound_dataset import LotusDataModule, USDataModuleV2
from loaders.ultrasound_dataset import VolumeSlicingProbeParamsDataset, ConcatDataModule
from transforms.ultrasound_transforms import RealUSTrainTransforms, RealUSEvalTransforms, RealUSTrainTransformsV2, RealUSEvalTransformsV2
# from callbacks.logger import ImageLoggerLotusNeptune

from nets import lotus, cut, diffusion
from callbacks import logger

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

    NN = getattr(cut, args.nn)    
    model = NN(**vars(args))

    # train_transform = RealUSTrainTransforms()
    # valid_transform = RealUSEvalTransforms()
    train_transform = RealUSTrainTransformsV2()
    valid_transform = RealUSEvalTransformsV2()
    dsA_data = USDataModuleV2(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, drop_last=True, train_transform=train_transform, valid_transform=valid_transform)

    dsA_data.setup()

    # train_dl = dsA_data.train_dataloader()
    # for idx, batch in enumerate(train_dl):
    #     X = batch
    #     print("__")
    #     print(torch.min(X), torch.max(X))
    #     print(X.shape)
    #     print("..")
    # quit()


    # train_transform_b = RealUSTrainTransformsV2()
    # valid_transform_b = RealUSEvalTransformsV2()
    train_transform_b = RealUSTrainTransforms()
    valid_transform_b = RealUSEvalTransforms()
    dsB_data = USDataModuleV2(df_train_b, df_val_b, df_test_b, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, drop_last=True, train_transform=train_transform_b, valid_transform=valid_transform_b)
    dsB_data.setup()

    concat_data = ConcatDataModule(dsA_data.train_ds, dsA_data.val_ds, dsB_data.train_ds, dsB_data.val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # concat_data.setup()
    # train_dl = concat_data.train_dataloader()
    # for idx, batch in enumerate(train_dl):
    #     X, Y = batch
    #     print("__")
    #     print(torch.min(X), torch.max(X))
    #     print(X.shape)
    #     print(Y.shape)
    #     print("..")
    # quit()


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=4,
        save_last=True, 
        monitor='val_loss'
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    callbacks=[early_stop_callback, checkpoint_callback]
    neptune_logger = None

    if args.neptune_tags:
        neptune_logger = NeptuneLogger(
            project='ImageMindAnalytics/Cut',
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


    parser = argparse.ArgumentParser(description='Cut original method training')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
    hparams_group.add_argument('--betas', help='Betas for optimizer', nargs='+', type=float, default=(0.9, 0.999))    
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience for early stopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=2)
    
    hparams_group.add_argument('--lambda_y', help='CUT model will compute the identity and calculate_NCE_loss', type=int, default=1)

    input_group = parser.add_argument_group('Input')
    
    input_group.add_argument('--nn', help='Type of neural network', type=str, default="CutG")        
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')    
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')  
    input_group.add_argument('--csv_train_b', required=True, type=str, help='Train CSV B domain')
    input_group.add_argument('--csv_valid_b', required=True, type=str, help='Valid CSV B domain')    
    input_group.add_argument('--csv_test_b', required=True, type=str, help='Test CSV')  
    input_group.add_argument('--img_column', type=str, default='img_path', help='Column name for image')

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output directory', type=str, default="./")
    
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--neptune_tags', help='Neptune tags', type=str, nargs="+", default=None)
    log_group.add_argument('--logger', help='Neptune tags', type=str, nargs="+", default="CutGLogger")
    log_group.add_argument('--log_steps', help='Log every N steps', type=int, default=10)

    args = parser.parse_args()

    main(args)