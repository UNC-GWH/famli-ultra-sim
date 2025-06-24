import os
import torch
from datetime import datetime
import torchvision
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import NeptuneLogger
from nets.DiffusionModel import DiffusionModel128
from USButterfly import USButterflyBlindSweepDataModule
import argparse

def train(n_epochs, precision, log_every_n_steps, patience, save_top_k, batch_size, num_workers, train_root_dir, val_root_dir):
    device = torch.device("cuda")
    torchvision.disable_beta_transforms_warning()

    # Checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs_lightning/CSVoutput_{timestamp}/"
    os.makedirs(output_dir, exist_ok=True)

    train_root_dir = train_root_dir
    val_root_dir = val_root_dir 

    # Dataset & DataLoader
    batch_size = batch_size
    num_workers = num_workers
    dm = USButterflyBlindSweepDataModule(batch_size=batch_size, num_workers=num_workers, num_frames=8, img_column='file_path', csv_train=train_root_dir, csv_valid=val_root_dir, mount_point='/mnt/raid/C1_ML_Analysis', drop_last=0)
        
    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,  # Directory to save checkpoints
        filename="{epoch:02d}-{val_loss:.2f}",  # Naming format
        save_top_k=save_top_k,  # Save only the 3 best models
        monitor="val_loss",  # Monitor validation loss
        mode="min",  # Save models with the lowest val_loss
        save_last=True  # Always keep the last epoch checkpoint
    )

    # Early stopping to stop training when validation loss stops improving
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,  # Stop if no improvement for 5 epochs
        mode="min",
        verbose=True
    )

    # PyTorch Lightning Trainer
    trainer = Trainer(
        accelerator="gpu", 
        devices=torch.cuda.device_count(),  
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=precision, 
        max_epochs=n_epochs,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # Train the model
    model = DiffusionModel128()
    model = model.to(device)
    model = model.float()
    trainer.fit(model, dm)

    final_model_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), final_model_path)

def main():

    parser = argparse.ArgumentParser(
        description="Train a diffusion model"
    )

    parser.add_argument('--n_epochs', type=int, default=50, help="Maximum number of epochs")
    parser.add_argument('--precision', type=int, default=16, help='Double precision (64, \'64\' or \'64-true\'), full precision (32, \'32\' or \'32-true\'), 16bit mixed precision (16, \'16\', \'16-mixed\') or bfloat16 mixed precision (\'bf16\', \'bf16-mixed\'). Can be used on CPU, GPU, TPUs, or HPUs. Default: \'32-true\'.')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='How often to log within steps. Default: 50.')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to go through without improvement before early stoppage')
    parser.add_argument('--save_top_k', type=int, default=3, help='Save top k checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of images to process at once')
    parser.add_argument('--num_workers', type=int, default = 8, help='Number of workers')
    parser.add_argument('--train_root_dir', type=str, default='/mnt/raid/C1_ML_Analysis/CSV_files/ALL_C2_cines_gt_ga_withmeta_20221031_butterfly_train.csv', help='Path to CSV containing paths to training data')
    parser.add_argument('--val_root_dir', type=str, default='/mnt/raid/C1_ML_Analysis/CSV_files/ALL_C2_cines_gt_ga_withmeta_20221031_butterfly_valid.csv', help='Path to CSV containing paths to validation data')

    args = parser.parse_args()

    train(n_epochs= args.n_epochs, precision=args.precision, log_every_n_steps=args.log_every_n_steps, patience=args.patience, save_top_k=args.save_top_k, batch_size=args.batch_size, num_workers=args.num_workers, train_root_dir=args.train_root_dir, val_root_dir=args.val_root_dir)

if __name__ == "__main__":
    main()