from torch.utils.data import Dataset, DataLoader 
import torch
import SimpleITK as sitk
import os
from lightning import LightningDataModule
import pandas as pd


class USButterflyBlindSweep(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', transform=None, num_frames=-1, continous_frames=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.keys = self.df.index
        self.num_frames = num_frames
        self.continous_frames = continous_frames 

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        try:
            img = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img)
            img_t = torch.tensor(img_np, dtype=torch.float32)
            
            if img.GetNumberOfComponentsPerPixel() == 1:
                img_t = img_t.unsqueeze(-1)
            elif  img.GetNumberOfComponentsPerPixel() == 3:
                img_t = img_t[:,:,:,0].unsqueeze(-1)

            
            img_t = img_t.permute(3, 0, 1, 2)/255.0  # Change to (C, D, H, W)             

            if self.num_frames > 0:
                
                idx = torch.randint(low=0, high=img_t.shape[1], size=(self.num_frames,))
                idx = idx.sort().values

            img_t = img_t[:, idx, :, :]
                    
        except:
            print("Error reading cine: " + img_path)
            n = self.num_frames if self.num_frames > 0 else 1
            img_t = torch.zeros(1, n, 256, 256, dtype=torch.float32)

        if self.transform:
            img_t = self.transform(img_t)

        return img_t.permute(1,0,2,3)
    
class USButterflyBlindSweepDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)

        self.train_transform = None
        self.valid_transform = None

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("USButterflyBlindSweepDataModule")
        group.add_argument('--batch_size', type=int, default=2)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--num_frames', type=int, default=8)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--drop_last', type=int, default=0)

        return parent_parser
        
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_ds = USButterflyBlindSweep(self.df_train, self.hparams.mount_point, img_column=self.hparams.img_column, transform=self.train_transform, num_frames=self.hparams.num_frames, continous_frames=True)
        self.val_ds = USButterflyBlindSweep(self.df_val, self.hparams.mount_point, img_column=self.hparams.img_column, transform=self.valid_transform, num_frames=self.hparams.num_frames, continous_frames=True)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=True, pin_memory=True, drop_last=bool(self.hparams.drop_last), shuffle=True, prefetch_factor=2,collate_fn=self.collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, drop_last=bool(self.hparams.drop_last), prefetch_factor=2,collate_fn=self.collate_fn)
    def collate_fn(self,batch):
        return torch.cat(batch, dim=0)
