from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
import nrrd
import os
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


from lightning.pytorch.core import LightningDataModule

import monai
from monai.transforms import (    
    LoadImage,
    LoadImaged
)
from monai.data import ITKReader

from transforms import ultrasound_transforms

class USDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, repeat_channel=True, return_head=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.repeat_channel = repeat_channel
        self.return_head = return_head

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            if os.path.splitext(img_path)[1] == ".nrrd":
                img, head = nrrd.read(img_path, index_order="C")
                img = img.astype(float)
                # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                img = img.squeeze()
                if self.repeat_channel:
                    img = img.unsqueeze(0).repeat(3,1,1)
            else:
                img = np.array(Image.open(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                if len(img.shape) == 3:                    
                    img = torch.permute(img, [2, 0, 1])[0:3, :, :]
                else:                    
                    img = img.unsqueeze(0).repeat(3,1,1)            
        except:
            print("Error reading frame:" + img_path, file=sys.stderr)
            img = torch.tensor(np.zeros([3, 256, 256]), dtype=torch.float32)

        if(self.transform):
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            return img, torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            return img, torch.tensor(scalar)            
        if self.return_head:
            return img, head

        return img

class USDatasetV2(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, return_head=False):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.return_head = return_head
        self.loader = LoadImage()

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        img = self.loader(img_path)        

        if(self.transform):
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            return img, torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            return img, torch.tensor(scalar)            
        if self.return_head:
            return img, head

        return img

class SimuDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, repeat_channel=True, return_head=False, target_column=None, target_transform=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.target_column = target_column
        self.target_transform = target_transform
        self.class_column = class_column
        self.ga_column = ga_column
        self.scalar_column = scalar_column
        self.repeat_channel = repeat_channel
        self.return_head = return_head

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            if os.path.splitext(img_path)[1] == ".nrrd":
                img, head = nrrd.read(img_path, index_order="C")
                img = img.astype(float)
                # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                img = img.squeeze()
                if self.repeat_channel:
                    img = img.unsqueeze(0).repeat(3,1,1)
            else:
                img = np.array(Image.open(img_path))
                img = torch.tensor(img, dtype=torch.float32)
                if len(img.shape) == 3:                    
                    img = torch.permute(img, [2, 0, 1])[0:3, :, :]
                else:                    
                    img = img.unsqueeze(0).repeat(3,1,1)            
        except:
            print("Error reading frame:" + img_path, file=sys.stderr)
            img = torch.tensor(np.zeros([1, 256, 256]), dtype=torch.float32)

        img = img/339
        if(self.transform):
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        if self.ga_column:
            ga = self.df.iloc[idx][self.ga_column]
            return img, torch.tensor([ga])

        if self.scalar_column:
            scalar = self.df.iloc[idx][self.scalar_column]
            return img, torch.tensor(scalar)            
        if self.return_head:
            return img, head

        if self.target_column:
            try:
                target_img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.target_column])
                if os.path.splitext(img_path)[1] == ".nrrd":
                    target, head = nrrd.read(target_img_path, index_order="C")
                    # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                    target = torch.tensor(target, dtype=torch.float32)
                    target = target.squeeze()
                    target = target[:,:,0]
                    if self.repeat_channel:
                        target = target.unsqueeze(0).repeat(3,1,1)
                else:
                    target = np.array(Image.open(target_img_path))
                    target = torch.tensor(target, dtype=torch.float32)
                    if len(img.shape) == 3:                    
                        target = torch.permute(img, [2, 0, 1])[0:3, :, :]
                    else:                    
                        target = target.unsqueeze(0).repeat(3,1,1)            
            except:
                print("Error reading frame: " + target_img_path, file=sys.stderr)
                target = torch.tensor(np.zeros([1, 256, 256]), dtype=torch.float32)
            
            target = target/255
            if(self.transform):
                target = self.transform(target)

            return img, target

        return img


class LotusDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column="img_path", seg_column="seg_path"):
        self.df = df
        self.mount_point = mount_point
        self.img_column = img_column
        self.seg_column = seg_column
        
        self.loader = LoadImaged(keys=["img", "seg"])

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        seg_path = os.path.join(self.mount_point, self.df.iloc[idx][self.seg_column])

        d = {"img": img_path, "seg": seg_path}
        
        return self.loader(d)
    
class LotusDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", seg_column="seg_path", train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column        
        self.seg_column = seg_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last        

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(data=LotusDataset(self.df_train, self.mount_point, img_column=self.img_column, seg_column=self.seg_column), transform=self.train_transform)
        self.val_ds = monai.data.Dataset(data=LotusDataset(self.df_val, self.mount_point, img_column=self.img_column, seg_column=self.seg_column), transform=self.valid_transform)
        self.test_ds = monai.data.Dataset(data=LotusDataset(self.df_test, self.mount_point, img_column=self.img_column, seg_column=self.seg_column), transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=False, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

class USDatasetBlindSweep(Dataset):
    def __init__(self, df, mount_point = "./", num_frames=0, img_column='img_path', ga_column=None, transform=None, id_column=None, max_sweeps=4):
        self.df = df
        self.mount_point = mount_point
        self.num_frames = num_frames
        self.transform = transform
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.max_sweeps = max_sweeps

        self.keys = self.df.index

        if self.id_column:        
            self.df_group = self.df.groupby(id_column)            
            self.keys = list(self.df_group.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        if self.id_column:
            df_group = self.df_group.get_group(self.keys[idx])
            ga = float(df_group[self.ga_column].unique()[0])
        
            img = self.create_seq(df_group)
        
            return img, torch.tensor([ga], dtype=torch.float32)
        else:
        
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            try:
                # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
                img, head = nrrd.read(img_path, index_order="C")
                img = torch.tensor(img, dtype=torch.float32)
                if self.num_frames > 0:
                    idx = torch.randint(low=0, high=img.shape[0], size=self.num_frames)
                    # idx = torch.randperm(img.shape[0])[:self.num_frames]
                    if self.num_frames == 1:
                        img = img[idx[0]]
                    else:
                        img = img[idx]
            except:
                print("Error reading cine: " + img_path)
                if self.num_frames == 1:
                    img = torch.zeros(256, 256, dtype=torch.float32)
                else:
                    img = torch.zeros(self.num_frames, 256, 256, dtype=torch.float32)
            
            if self.num_frames == 1:
                img = img.unsqueeze(0).repeat(3,1,1).contiguous()
            else:
                img = img.unsqueeze(1).repeat(1,3,1,1).contiguous()

            if self.transform:
                img = self.transform(img)

            if self.ga_column:
                ga = self.df.iloc[idx][self.ga_column]
                return img, torch.tensor([ga])

            return img

    def create_seq(self, df):

        # shuffle
        df = df.sample(frac=1)

        # get maximum number of samples, -1 uses all
        max_sweeps = len(df.index)
        if self.max_sweeps > -1:
            max_sweeps = min(max_sweeps, self.max_sweeps)        

        # get the rows from the shuffled dataframe and sort them
        df = df[0:max_sweeps].sort_index()

        # read all of them
        
        imgs = []

        for idx, row in df.iterrows():
            # try:
            img_path = os.path.join(self.mount_point, row[self.img_column])                
            img_np, head = nrrd.read(img_path, index_order="C")
            img_t = torch.tensor(img_np)

            if self.transform:
                img_t = self.transform(img_t)                
            imgs.append(img_t)
            # except Exception as e:
            #     print(e, file=sys.stderr)
        return torch.cat(imgs)
    
class USDatasetBlindSweepWTag(Dataset):
    def __init__(self, df, mount_point = "./", img_column='file_path', tag_column='tag', ga_column='ga_boe', transform=None, id_column='study_id', max_sweeps=3):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.max_sweeps = max_sweeps
        self.tag_column = tag_column

        self.keys = self.df.index

        if self.id_column:        
            self.df_group = self.df.groupby(id_column)            
            self.keys = list(self.df_group.groups.keys())

        self.tags_dict = {'M': 0,
            'L0': 1,
            'L1': 2,
            'R0': 3,
            'R1': 4,
            'C1': 5,
            'C2': 6,
            'C3': 7,
            'C4': 8}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        
        df_group = self.df_group.get_group(self.keys[idx])
        ga = float(df_group[self.ga_column].unique()[0])
    
        img_d = self.create_seq(df_group)
        img_d['ga_boe'] = torch.tensor([ga], dtype=torch.float32)
    
        return img_d

    def create_seq(self, df):

        # shuffle
        df = df.sample(frac=1)

        # get maximum number of samples, -1 uses all
        max_sweeps = len(df.index)
        if self.max_sweeps > -1:
            max_sweeps = min(max_sweeps, self.max_sweeps)        

        # get the rows from the shuffled dataframe and sort them
        df = df[0:max_sweeps].sort_index()

        # read all of them
        
        imgs = {}

        imgs['tag'] = []

        num_sweeps = 0

        for idx, row in df.iterrows():
            
            img_path = os.path.join(self.mount_point, row[self.img_column])                
            img_np, head = nrrd.read(img_path, index_order="C")

            if len(img_np.shape) == 4:
                img_np = img_np[:,:,:,0]

            img_t = torch.tensor(img_np)

            if self.transform:
                img_t = self.transform(img_t)

            imgs[num_sweeps] = img_t
            num_sweeps += 1

            imgs['tag'].append(self.tags_dict[row[self.tag_column]])

        for k in range(self.max_sweeps):
            if k not in imgs:
                imgs[k] = torch.zeros(16, 256, 256, dtype=torch.float32)
                imgs['tag'].append(0)
        
        imgs['tag'] = torch.tensor(imgs['tag'], dtype=torch.long)
        
        return imgs

class USDatasetVolumes(Dataset):
    def __init__(self, df, mount_point = "./", num_frames=0, img_column='img_path', ga_column='ga_boe', id_column='study_id', max_seq=-1, transform=None):
        self.df = df
        
        self.mount_point = mount_point
        self.num_frames = num_frames
        self.transform = transform
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.max_seq = max_seq

        print(id_column)
        self.df_group = self.df.groupby(id_column)
        print(len(self.df_group.groups.keys()))
        self.keys = list(self.df_group.groups.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        
        df_group = self.df_group.get_group(self.keys[idx])
        ga = float(df_group[self.ga_column].unique()[0])
        
        img = self.create_seq(df_group)
        
        return torch.tensor(img, dtype=torch.float32), torch.tensor([ga], dtype=torch.float32)

    def create_seq(self, df):

        # shuffle
        df = df.sample(frac=1)

        # get maximum number of samples, -1 uses all
        max_seq = len(df.index)
        if self.max_seq > -1:
            max_seq = min(max_seq, self.max_seq)        

        # get the rows from the shuffled dataframe and sort them
        df = df[0:max_seq].sort_index()

        # read all of them
        imgs = []
        time_steps = 0 
        for idx, row in df.iterrows():
            try:
                img_path = os.path.join(self.mount_point, row[self.img_column])
                img_np, head = nrrd.read(img_path, index_order="C")

                if self.transform:
                    img_np = self.transform(img_np)

                imgs.append(img_np)
            except Exception as e:
                print(e, file=sys.stderr)

        return np.stack(imgs)

    # def has_all_types(self, df, seqo):
    #     if(seqo[0] == "all"):
    #         return True
    #     seq_found = np.zeros(len(seqo))
    #     for i, t in df["tag"].items():
    #         scan_index = np.where(np.array(seqo) == t)[0]
    #         for s_i in scan_index:
    #             seq_found[s_i] += 1
    #     return np.prod(seq_found) > 0

# class ITKImageDataset(Dataset):
#     def __init__(self, csv_file, transform=None, target_transform=None):
#         self.df = pd.read_csv(csv_file)

#         self.transform = transform
#         self.target_transform = target_transform
#         self.sequence_order = ["all"]

#         self.df = self.df.groupby('study_id').filter(lambda x: has_all_types(x, self.sequence_order))

#         self.df_group = self.df.groupby('study_id')
#         self.keys = list(self.df_group.groups.keys())
#         self.data_frames = []

#     def __len__(self):
#         return len(self.df_group)

#     def __getitem__(self, idx):

#         df_group = self.df_group.get_group(self.keys[idx])

#         seq_np, df = create_seq(df_group, self.sequence_order)
#         ga = df["ga_boe"]
#         # img = self.df.iloc[idx]['file_path']
#         # ga = self.df.iloc[idx]['ga_boe']

#         # reader = ITKReader()
#         # img = reader.read(img)

#         # if self.transform:
#         #     img = self.transform(img)
#         # if self.target_transform:
#         #     ga = self.target_transform(ga)

#         self.data_frames.append(df)

#         return (self.transform(seq_np), np.array([ga]))


class USDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False, repeat_channel=True, target_column=None):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.ga_column = ga_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last
        self.repeat_channel = repeat_channel
        self.target_column = target_column

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform, repeat_channel=self.repeat_channel)
        self.val_ds = USDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform, repeat_channel=self.repeat_channel)
        self.test_ds = USDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform, repeat_channel=self.repeat_channel)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

class USDataModuleV2(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False, target_column=None):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.ga_column = ga_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last    
        self.target_column = target_column

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetV2(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform)
        self.val_ds = USDatasetV2(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform)
        self.test_ds = USDatasetV2(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)


class SimuDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, ga_column=None, scalar_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False, repeat_channel=True, target_column=None):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column
        self.scalar_column = scalar_column
        self.ga_column = ga_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last
        self.repeat_channel = repeat_channel
        self.target_column = target_column

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = SimuDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.train_transform, repeat_channel=self.repeat_channel, target_column=self.target_column)
        self.val_ds = SimuDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.valid_transform, repeat_channel=self.repeat_channel, target_column=self.target_column)
        self.test_ds = SimuDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, ga_column=self.ga_column, scalar_column=self.scalar_column, transform=self.test_transform, repeat_channel=self.repeat_channel, target_column=self.target_column)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

class USDataModuleBlindSweep(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, num_frames=50, max_sweeps=-1, img_column='uuid_path', ga_column=None, id_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column
        self.num_frames = num_frames
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last
        self.max_sweeps = max_sweeps

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetBlindSweep(self.df_train, mount_point=self.mount_point, num_frames=self.num_frames, img_column=self.img_column, ga_column=self.ga_column,id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.train_transform)
        self.val_ds = USDatasetBlindSweep(self.df_val, mount_point=self.mount_point, num_frames=self.num_frames, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.valid_transform)
        self.test_ds = USDatasetBlindSweep(self.df_test, mount_point=self.mount_point, num_frames=self.num_frames, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.test_transform)
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.pad_seq)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def pad_seq(self, batch):

        blind_sweeps = [bs for bs, g in batch]
        ga = [g for v, g in batch]    
        
        blind_sweeps = pad_sequence(blind_sweeps, batch_first=True, padding_value=0.0)
        ga = torch.stack(ga)

        return blind_sweeps, ga
    

class USDataModuleBlindSweepWTag(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=2, num_workers=4, max_sweeps=-1, max_sweeps_val=-1, img_column='uuid_path', ga_column=None, id_column=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column = id_column


        self.train_transform = ultrasound_transforms.BlindSweepWTagTrainTransforms()
        self.valid_transform = ultrasound_transforms.BlindSweepWTagEvalTransforms()
        self.test_transform = ultrasound_transforms.BlindSweepWTagEvalTransforms()
        self.drop_last=drop_last
        self.max_sweeps = max_sweeps
        self.max_sweeps_val = max_sweeps_val

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetBlindSweepWTag(self.df_train, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column,id_column=self.id_column, max_sweeps=self.max_sweeps, transform=self.train_transform)
        self.val_ds = USDatasetBlindSweepWTag(self.df_val, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=-1, transform=self.valid_transform)
        self.test_ds = USDatasetBlindSweepWTag(self.df_test, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_sweeps=-1, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.pad_seq)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_seq)

    def pad_seq(self, batch):

        img_d = {}

        keys = batch[0].keys()

        for k in keys:
            if isinstance(k, int):
                img_d[k] = [bs[k].squeeze(0) for bs in batch]
                img_d[k] = pad_sequence(img_d[k], batch_first=True, padding_value=0.0).unsqueeze(1)

        img_d[self.ga_column] = torch.stack([g[self.ga_column] for g in batch])

        if len(img_d[self.ga_column].shape) == 1:
            img_d[self.ga_column] = img_d[self.ga_column].unsqueeze(-1)
        
        img_d['tag'] = torch.stack([g['tag'] for g in batch])

        return img_d


class USDataModuleVolumes(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, max_seq=5, img_column='img_path', ga_column='ga_boe', id_column='study_id', train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.ga_column = ga_column
        self.id_column= id_column
        self.max_seq = max_seq
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USDatasetVolumes(self.df_train, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_seq=self.max_seq, transform=self.train_transform)
        self.val_ds = USDatasetVolumes(self.df_val, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_seq=self.max_seq, transform=self.valid_transform)
        self.test_ds = USDatasetVolumes(self.df_test, mount_point=self.mount_point, img_column=self.img_column, ga_column=self.ga_column, id_column=self.id_column, max_seq=self.max_seq, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.pad_volumes)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.pad_volumes)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_volumes)

    def pad_volumes(self, batch):

        volumes = [v for v, g in batch]
        ga = [g for v, g in batch]    
        
        volumes = pad_sequence(volumes, batch_first=True, padding_value=0.0)
        ga = torch.stack(ga)

        return volumes, ga





class USZDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path"):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        img_path_z_mu = img_path.replace(".nrrd", "z_mu.nrrd")
        img_path_z_sigma = img_path.replace(".nrrd", "z_mu.nrrd")

        z_mu, head = nrrd.read(img_path_z_mu, index_order="C")
        z_sigma, head = nrrd.read(img_path_z_mu, index_order="C")

        
        z_mu = torch.tensor(z_mu, dtype=torch.float32)
        z_sigma = torch.tensor(z_sigma, dtype=torch.float32)

        if len(z_mu.shape) == 2:
            z_mu = z_mu.unsqueeze(0)
        if len(z_sigma.shape) == 2:
            z_sigma = z_sigma.unsqueeze(0)
            
        
        img = {"z_mu": z_mu, "z_sigma": z_sigma}
        
        if(self.transform):
            img = self.transform(img)

        
        return img

class USZDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last        

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = USZDataset(self.df_train, self.mount_point, img_column=self.img_column, transform=self.train_transform)
        self.val_ds = USZDataset(self.df_val, self.mount_point, img_column=self.img_column, transform=self.valid_transform)
        self.test_ds = USZDataset(self.df_test, self.mount_point, img_column=self.img_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)
    


class FluidDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img", seg_column="seg", rois=["Amniotic_fluid", "Maternal_bladder", "Fetal_chest_abdomen", "Fetal_head", "Fetal_heart", "Fetal_limb_other", "Placenta", "Umbilical_cord"]):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column        
        self.seg_column = seg_column
        self.rois = rois

        # self.colname_roi_name_map = {'Amniotic_fluid_indicator': 'Amniotic_fluid',
        #                 'Maternal_bladder_indicator': 'Maternal_bladder',
        #                 'fetal_chest_abdomen_indicator': 'Fetal_chest_abdomen',
        #                 'fetal_head_indicator': 'Fetal_head',
        #                 'fetal_heart_indicator': 'Fetal_heart',
        #                 'fetal_limb_other_indicator': 'Fetal_limb_other',
        #                 'placenta_indicator': 'Placenta',
        #                 'umbilical_cord_indicator': 'Umbilical_cord',
        #                 'dropout_indicator': 'Dropout',
        #                 'shadowing_indicator': 'Shadowing'}

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        seg_path = os.path.join(self.mount_point, self.df.iloc[idx][self.seg_column])

        reader = ITKReader()
        img, _ = reader.get_data(reader.read(img_path))

        # img = sitk.ReadImage(img_path)
        img_t = torch.tensor(img)

        reader = ITKReader()
        seg, _ = reader.get_data(reader.read(seg_path))
        seg_t = torch.tensor(seg)
        
        img_d = {self.img_column: img_t, self.seg_column: seg_t}
        
        if self.transform:
            return self.transform(img_d)
        return img_d

class FluidDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.df_train = pd.read_csv(self.hparams.csv_train)
        self.df_val = pd.read_csv(self.hparams.csv_valid)
        self.df_test = pd.read_csv(self.hparams.csv_test)
        
        self.mount_point = self.hparams.mount_point
        self.batch_size = self.hparams.batch_size
        self.num_workers = self.hparams.num_workers
        self.img_column = self.hparams.img_column
        self.seg_column = self.hparams.seg_column
        self.drop_last = bool(self.hparams.drop_last)

        self.train_transform = ultrasound_transforms.FluidTrainTransforms()
        self.valid_transform = ultrasound_transforms.FluidEvalTransforms()
        self.test_transform = ultrasound_transforms.FluidEvalTransforms()

    @staticmethod
    def add_data_specific_args(parent_parser):

        group = parent_parser.add_argument_group("SaxiOctreeDataModule")
        
        group.add_argument('--batch_size', type=int, default=8)
        group.add_argument('--num_workers', type=int, default=6)
        group.add_argument('--img_column', type=str, default="img")
        group.add_argument('--seg_column', type=str, default="seg")
        group.add_argument('--csv_train', type=str, default=None, required=True)
        group.add_argument('--csv_valid', type=str, default=None, required=True)
        group.add_argument('--csv_test', type=str, default=None, required=True)
        group.add_argument('--mount_point', type=str, default="./")
        group.add_argument('--drop_last', type=int, default=0)

        return parent_parser

    def collate_fn(self, batch):
        X = [b[self.img_column] for b in batch]

        x_v = [x[0] for x in X]
        x_f = [x[1] for x in X]
        
        X_v = pad_sequence(x_v, batch_first=True, padding_value=0.0)
        X_f = pad_sequence(x_f, batch_first=True, padding_value=0.0)


        Y = [b[self.seg_column] for b in batch]

        y_v = [y[0] for y in Y]
        y_f = [y[1] for y in Y]
        
        Y_v = pad_sequence(y_v, batch_first=True, padding_value=0.0)
        Y_f = pad_sequence(y_f, batch_first=True, padding_value=0.0)
        
        return {self.img_column: (X_v, X_f), self.seg_column: (Y_v, Y_f)}

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = FluidDataset(self.df_train, self.mount_point, img_column=self.img_column, seg_column=self.seg_column, transform=self.train_transform)
        self.val_ds = FluidDataset(self.df_val, self.mount_point, img_column=self.img_column, seg_column=self.seg_column, transform=self.valid_transform)
        self.test_ds = FluidDataset(self.df_test, self.mount_point, img_column=self.img_column, seg_column=self.seg_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.collate_fn)



