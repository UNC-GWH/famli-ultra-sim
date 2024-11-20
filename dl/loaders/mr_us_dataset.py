
from torch.utils.data import Dataset, DataLoader, default_collate
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

import pickle
import monai 
import vtk
import random
import pandas as pd

from torch.nn import functional as F

sys.path.append('/mnt/raid/C1_ML_Analysis/source/ShapeAXI/src')

from shapeaxi import utils

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, use_max=True):
        self.datasets = datasets
        self.use_max = use_max

    def __getitem__(self, i):
        # print(f"Fetching index: {i}") #debug
        # for d in self.datasets:
            # print(f"Dataset length: {len(d)}") #debug
        return tuple(self.check_len(d, i) for d in self.datasets)

    def __len__(self):
        if self.use_max:
            return max(len(d) for d in self.datasets)
        return min(len(d) for d in self.datasets)

    def shuffle(self):
        for d in self.datasets:
            if isinstance(d, monai.data.Dataset):                
                d.data.df = d.data.df.sample(frac=1.0).reset_index(drop=True)                
            else:
                d.df = d.df.sample(frac=1.0).reset_index(drop=True)

    def check_len(self, d, i):
        if i < len(d):
            return d[i]
        else:
            j = i % len(d)
            return d[j]


class StackDataset(Dataset):
    def __init__(self, dataset, stack_slices=10, shuffle_df=False):        
        self.dataset = dataset
        if shuffle_df:
            self.dataset.df = self.dataset.df.sample(frac=1).reset_index(drop=True)
        self.stack_slices = stack_slices       

    def __len__(self):
        return len(self.dataset)//self.stack_slices

    def __getitem__(self, idx):
        
        start_idx = idx*self.stack_slices

        return torch.stack([self.dataset[idx] for idx in range(start_idx, start_idx + self.stack_slices)], dim=1)

class MRDatasetVolumes(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', id_column='study_id', transform=None):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        return self.transform(img_path)

class VolumeSlicingProbeParamsDataset(Dataset):
    def __init__(self, diffusor_df: pd.DataFrame, probe_params_df: pd.DataFrame, mount_point="./", transform=None, us_plane_fn='source/blender/simulated_data_export/studies_merged/simulation_ultrasound_plane.stl', n_samples=1000):
        
        self.mount_point = mount_point
        self.transform = transform

        self.df = diffusor_df
        self.probe_params_df = probe_params_df

        self.diffusors = []        

        for _, row in self.df.iterrows():
            diffusor_fn = os.path.join(mount_point, row['img_path'])
            
            diffusor = sitk.ReadImage(os.path.join(self.mount_point, diffusor_fn))
            diffusor_np = sitk.GetArrayFromImage(diffusor)            
            diffusor_t = torch.tensor(diffusor_np.astype(int)).unsqueeze(0)            

            diffusor_spacing = np.array(diffusor.GetSpacing())
            diffusor_size = np.array(diffusor.GetSize())

            diffusor_origin = np.array(diffusor.GetOrigin())
            diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size

            self.diffusors.append({"diffusor_t": diffusor_t, "diffusor_origin": diffusor_origin, "diffusor_end": diffusor_end})

        print("All diffusors loaded!")
            

        self.probe_directions = []
        self.probe_origins = []
        self.n_samples = n_samples

        for _, row in self.probe_params_df.iterrows():
            probe_params = pickle.load(open(os.path.join(self.mount_point, row['probe_param_fn']), 'rb'))
        
            probe_direction = torch.tensor(probe_params['probe_direction'], dtype=torch.float32)
            probe_origin = torch.tensor(probe_params['probe_origin'], dtype=torch.float32)

            self.probe_directions.append(probe_direction.T)
            self.probe_origins.append(probe_origin)

        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(mount_point, us_plane_fn))
        reader.Update()
        simulation_ultrasound_plane = reader.GetOutput()
        simulation_ultrasound_plane_bounds = np.array(simulation_ultrasound_plane.GetBounds())

        simulation_ultrasound_plane_mesh_grid_size = [256, 256, 1]
        simulation_ultrasound_plane_mesh_grid_params = [torch.arange(start=start, end=end, step=(end - start)/simulation_ultrasound_plane_mesh_grid_size[idx]) for idx, (start, end) in enumerate(zip(simulation_ultrasound_plane_bounds[[0,2,4]], simulation_ultrasound_plane_bounds[[1,3,5]]))]
        self.simulation_ultrasound_plane_mesh_grid = torch.stack(torch.meshgrid(simulation_ultrasound_plane_mesh_grid_params), dim=-1).squeeze().to(torch.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
            
        diffusor_d = random.choice(self.diffusors)
        diffusor_t = diffusor_d['diffusor_t']
        diffusor_origin = diffusor_d['diffusor_origin']
        diffusor_end = diffusor_d['diffusor_end']

        is_empty = True

        while is_empty:

            ridx = random.randint(0, len(self.probe_directions)-1)
            probe_direction = self.probe_directions[ridx]
            probe_origin = self.probe_origins[ridx]

            simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane_single(probe_direction, probe_origin, diffusor_origin, diffusor_end)

            diffusor_sample_t = self.diffusor_sampling(diffusor_t.unsqueeze(0).to(torch.float32), simulation_ultrasound_plane_mesh_grid_transformed_t.unsqueeze(0)).squeeze().unsqueeze(0)
            
            if torch.any(torch.isin(diffusor_sample_t[:, 128:, 64:192], torch.tensor([4, 7]))):
                is_empty = False

        if self.transform:
            diffusor_sample_t = self.transform(diffusor_sample_t)
        
        return diffusor_sample_t

    def transform_simulation_ultrasound_plane(self, probe_directions, probe_origins, diffusor_origin, diffusor_end):

        simulation_ultrasound_plane_mesh_grid_transformed_t = []
        for probe_origin, probe_direction in zip(probe_origins, probe_directions):

            simulation_ultrasound_plane_mesh_grid_transformed = self.transform_simulation_ultrasound_plane_single(probe_direction, probe_origin, diffusor_origin, diffusor_end)

            simulation_ultrasound_plane_mesh_grid_transformed_t.append(simulation_ultrasound_plane_mesh_grid_transformed)
        
        simulation_ultrasound_plane_mesh_grid_transformed_t = torch.cat(simulation_ultrasound_plane_mesh_grid_transformed_t, dim=0)
        
        return simulation_ultrasound_plane_mesh_grid_transformed_t
    
    def transform_simulation_ultrasound_plane_single(self, probe_direction, probe_origin, diffusor_origin, diffusor_end):

        theta = torch.rand(3) * torch.tensor([0.05*torch.pi, 0.05*torch.pi, 2*torch.pi])
        # Generate rotation matrix
        R = self.euler_angles_to_rotation_matrix(theta)
        probe_direction_rotated = torch.matmul(probe_direction, R)
        probe_origin_translated = torch.randn(3)*0.001 + probe_origin

        simulation_ultrasound_plane_mesh_grid_transformed = torch.matmul(self.simulation_ultrasound_plane_mesh_grid, probe_direction_rotated) + probe_origin_translated
        simulation_ultrasound_plane_mesh_grid_transformed = 2.*(simulation_ultrasound_plane_mesh_grid_transformed - diffusor_origin)/(diffusor_end - diffusor_origin) - 1.

        return simulation_ultrasound_plane_mesh_grid_transformed.to(torch.float32).unsqueeze(0)
    
    def diffusor_sampling(self, diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_t):
        return F.grid_sample(diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_t, mode='nearest', align_corners=True)
    
    def read_probe_params(self, probe_params_fn):
        return pickle.load(open(os.path.join(self.mount_point, probe_params_fn), 'rb'))

    # Function to create a rotation matrix from Euler angles
    def euler_angles_to_rotation_matrix(self, theta):
        R_x = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(theta[0]), -torch.sin(theta[0])],
            [0, torch.sin(theta[0]), torch.cos(theta[0])]
        ])

        R_y = torch.tensor([
            [torch.cos(theta[1]), 0, torch.sin(theta[1])],
            [0, 1, 0],
            [-torch.sin(theta[1]), 0, torch.cos(theta[1])]
        ])

        R_z = torch.tensor([
            [torch.cos(theta[2]), -torch.sin(theta[2]), 0],
            [torch.sin(theta[2]), torch.cos(theta[2]), 0],
            [0, 0, 1]
        ])

        R = torch.mm(torch.mm(R_z, R_y), R_x)
        return R


class VolumeSlicingDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, df_probe_params, mount_point="./", batch_size=256, num_workers=1, img_column="img_path", seg_column="seg_path", train_transform=None, valid_transform=None, test_transform=None, drop_last=False, n_samples_train=1000, n_samples_val=100):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.df_probe_params = df_probe_params
        self.n_samples_train = n_samples_train
        self.n_samples_val = n_samples_val
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
        self.train_ds = monai.data.Dataset(data=VolumeSlicingProbeParamsDataset(self.df_train, probe_params_df=self.df_probe_params, mount_point=self.mount_point, transform=self.train_transform, n_samples=self.n_samples_train))
        self.val_ds = monai.data.Dataset(data=VolumeSlicingProbeParamsDataset(self.df_val, probe_params_df=self.df_probe_params, mount_point=self.mount_point, transform=self.train_transform, n_samples=self.n_samples_val))
        self.test_ds = monai.data.Dataset(data=VolumeSlicingProbeParamsDataset(self.df_test, probe_params_df=self.df_probe_params, mount_point=self.mount_point, transform=self.train_transform))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)

class MRDataModuleVolumes(LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, img_column='img_path', ga_column='ga_boe', id_column='study_id', train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
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
        self.train_ds = MRDatasetVolumes(self.df_train, mount_point=self.mount_point, img_column=self.img_column, id_column=self.id_column, transform=self.train_transform)
        self.val_ds = MRDatasetVolumes(self.df_val, mount_point=self.mount_point, img_column=self.img_column, id_column=self.id_column, transform=self.valid_transform)
        self.test_ds = MRDatasetVolumes(self.df_test, mount_point=self.mount_point, img_column=self.img_column, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, collate_fn=self.arrange_slices)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, collate_fn=self.arrange_slices)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.arrange_slices)

    def arrange_slices(self, batch):
        batch = torch.cat(batch, axis=1).permute(dims=(1,0,2,3))
        print(f"Batch shape after concatenation and permutation: {batch.shape}") #debug
        idx = torch.randperm(batch.shape[0])
        return batch[idx]

class MRUSDataModule(LightningDataModule):
    def __init__(self, mr_dataset_train, mr_dataset_val, us_dataset_train, us_dataset_val, batch_size=4, num_workers=4):
        super().__init__()

        self.mr_dataset_train = mr_dataset_train
        self.us_dataset_train = us_dataset_train

        self.mr_dataset_val = mr_dataset_val
        self.us_dataset_val = us_dataset_val
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = ConcatDataset(self.mr_dataset_train, self.us_dataset_train)
        self.val_ds = ConcatDataset(self.mr_dataset_val, self.us_dataset_val)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=True, collate_fn=self.arrange_slices)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.arrange_slices)    

    def arrange_slices(self, batch):
        mr_batch = [mr for mr, us in batch]
        us_batch = [us for mr, us in batch]        
        mr_batch = torch.cat(mr_batch, axis=1).permute(dims=(1,0,2,3))
        us_batch = torch.cat(us_batch, axis=1).permute(dims=(1,0,2,3))        
        return mr_batch[torch.randperm(mr_batch.shape[0])], us_batch

class ConcatDataModule(LightningDataModule):
    def __init__(self, datasetA_train, datasetA_val, datasetB_train, datasetB_val, batch_size=8, num_workers=4):
        super().__init__()

        self.datasetA_train = datasetA_train
        self.datasetB_train = datasetB_train

        self.datasetA_val = datasetA_val
        self.datasetB_val = datasetB_val
        
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders        
        self.train_ds = ConcatDataset(self.datasetA_train, self.datasetB_train)
        self.val_ds = ConcatDataset(self.datasetA_val, self.datasetB_val)

    def train_dataloader(self):
        self.train_ds.shuffle()
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)    
        

class DiffusorDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', transform=None):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        diffusor_np, diffusor_head = nrrd.read(img_path)
        diffusor_t = torch.tensor(diffusor_np.astype(int)).permute(2, 1, 0)
        diffusor_size = torch.tensor(diffusor_head['sizes'])
        diffusor_spacing = torch.tensor(np.diag(diffusor_head['space directions']))

        diffusor_origin = torch.tensor(diffusor_head['space origin']).flip(dims=[0])
        diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size

        if self.transform:
            return self.transform(diffusor_t), diffusor_origin, diffusor_end
        return diffusor_t, diffusor_origin, diffusor_end    
    
class DiffusorDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", train_transform=None, valid_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.drop_last = drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = DiffusorDataset(self.df_train, self.mount_point, img_column=self.img_column, transform=self.train_transform)
        self.val_ds = DiffusorDataset(self.df_val, self.mount_point, img_column=self.img_column, transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)

class DiffusorSampleDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', transform=None, num_samples=1000, return_ridx=False):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        self.num_samples = num_samples
        self.return_ridx = return_ridx        

        self.buffer = []
        for idx in range(len(self.df.index)):
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            diffusor_np, diffusor_head = nrrd.read(img_path)            
            diffusor_size = diffusor_head['sizes']
            diffusor_spacing = np.diag(diffusor_head['space directions'])

            diffusor_origin = np.flip(diffusor_head['space origin'], axis=0)
            diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size

            diffusor_arr = [diffusor_np, diffusor_origin, diffusor_end]

            self.buffer.append(diffusor_arr)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        r_idx = random.randint(0, len(self.buffer)-1)
        diffusor_np, diffusor_origin, diffusor_end = self.buffer[r_idx] 
        diffusor_t = torch.tensor(diffusor_np.copy().astype(int)).permute(2, 1, 0)     
        if self.transform:
            diffusor_t = self.transform(diffusor_t)            
        if self.return_ridx:
            return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy()), torch.tensor(r_idx)
        return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy())
    
class DiffusorSampleDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", train_transform=None, valid_transform=None, drop_last=False, num_samples_train=1000, num_samples_val=100):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.drop_last = drop_last
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = DiffusorSampleDataset(self.df_train, self.mount_point, img_column=self.img_column, transform=self.train_transform, num_samples=self.num_samples_train)
        self.val_ds = DiffusorSampleDataset(self.df_val, self.mount_point, img_column=self.img_column, transform=self.valid_transform, num_samples=self.num_samples_val)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, shuffle=True, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)
    

class DiffusorSampleSurfDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', surf_column='surf_path', transform=None, num_samples=1000, return_ridx=False):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        self.num_samples = num_samples
        self.return_ridx = return_ridx
        self.surf_column = surf_column

        self.buffer = []
        for idx in range(len(self.df.index)):
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            diffusor_np, diffusor_head = nrrd.read(img_path)            
            diffusor_size = diffusor_head['sizes']
            diffusor_spacing = np.diag(diffusor_head['space directions'])

            diffusor_origin = np.flip(diffusor_head['space origin'], axis=0)
            diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size
            
            surf_path = os.path.join(self.mount_point, self.df.iloc[idx][self.surf_column])
            surf = utils.ReadSurf(surf_path)
            verts, faces = utils.PolyDataToTensors_v_f(surf)

            self.buffer.append((diffusor_np, diffusor_origin, diffusor_end, verts, faces))


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        r_idx = random.randint(0, len(self.buffer)-1)
        diffusor_np, diffusor_origin, diffusor_end, V, F = self.buffer[r_idx] 
        diffusor_t = torch.tensor(diffusor_np.copy().astype(int)).permute(2, 1, 0)     
        if self.transform:
            diffusor_t = self.transform(diffusor_t)            
        if self.return_ridx:
            return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy()), torch.tensor(r_idx)
        return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy()), V, F
    
class DiffusorSampleSurfDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", surf_column="surf_path", train_transform=None, valid_transform=None, drop_last=False, num_samples_train=1000, num_samples_val=100):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.surf_column = surf_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.drop_last = drop_last
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = DiffusorSampleSurfDataset(self.df_train, self.mount_point, img_column=self.img_column, surf_column=self.surf_column, transform=self.train_transform, num_samples=self.num_samples_train)
        self.val_ds = DiffusorSampleSurfDataset(self.df_val, self.mount_point, img_column=self.img_column, surf_column=self.surf_column, transform=self.valid_transform, num_samples=self.num_samples_val)

    def pad_verts_faces(self, batch):
        # Collate function for the dataloader to know how to comine the data

        diffusor = [d for d, do, de, v, f  in batch]
        diffusor_origin = [do for d, do, de, v, f in batch]
        diffusor_end = [de for d, do, de, v, f in batch]
        
        verts = [v for d, do, de, v, f  in batch]
        faces = [f for d, do, de, v, f in batch]
        
        verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)
            
        return torch.stack(diffusor), torch.stack(diffusor_origin), torch.stack(diffusor_end), verts, faces

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, shuffle=True, prefetch_factor=2, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)
    


class RealUSSurfDataset(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', surf_column='surf_path', transform=None, num_samples=1000, return_ridx=False):
        self.df = df
        
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        self.num_samples = num_samples
        self.return_ridx = return_ridx
        self.surf_column = surf_column

        self.buffer = []
        for idx in range(len(self.df.index)):
            img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

            diffusor_np, diffusor_head = nrrd.read(img_path)            
            diffusor_size = diffusor_head['sizes']
            diffusor_spacing = np.diag(diffusor_head['space directions'])

            diffusor_origin = np.flip(diffusor_head['space origin'], axis=0)
            diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size
            
            surf_path = os.path.join(self.mount_point, self.df.iloc[idx][self.surf_column])
            surf = utils.ReadSurf(surf_path)
            verts, faces = utils.PolyDataToTensors_v_f(surf)

            self.buffer.append((diffusor_np, diffusor_origin, diffusor_end, verts, faces))

        self.tags_dict = {'M': 0, 'L0': 1, 'L1': 2, 'R0': 3, 'R1': 4, 'C1': 5, 'C2': 6, 'C3': 7, 'C4': 8}


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        r_idx = random.randint(0, len(self.buffer)-1)
        diffusor_np, diffusor_origin, diffusor_end, V, F = self.buffer[r_idx] 
        diffusor_t = torch.tensor(diffusor_np.copy().astype(int)).permute(2, 1, 0)     
        if self.transform:
            diffusor_t = self.transform(diffusor_t)            
        if self.return_ridx:
            return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy()), torch.tensor(r_idx)
        return diffusor_t, torch.tensor(diffusor_origin.copy()), torch.tensor(diffusor_end.copy()), V, F
    
class RealUSSurfDataModule(LightningDataModule):
    def __init__(self, df_train, df_val, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", surf_column="surf_path", train_transform=None, valid_transform=None, drop_last=False, num_samples_train=1000, num_samples_val=100):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.surf_column = surf_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.drop_last = drop_last
        self.num_samples_train = num_samples_train
        self.num_samples_val = num_samples_val

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = RealUSSurfDataset(self.df_train, self.mount_point, img_column=self.img_column, surf_column=self.surf_column, transform=self.train_transform, num_samples=self.num_samples_train)
        self.val_ds = RealUSSurfDataModule(self.df_val, self.mount_point, img_column=self.img_column, surf_column=self.surf_column, transform=self.valid_transform, num_samples=self.num_samples_val)

    def pad_verts_faces(self, batch):
        # Collate function for the dataloader to know how to comine the data

        diffusor = [d for d, do, de, v, f  in batch]
        diffusor_origin = [do for d, do, de, v, f in batch]
        diffusor_end = [de for d, do, de, v, f in batch]
        
        verts = [v for d, do, de, v, f  in batch]
        faces = [f for d, do, de, v, f in batch]
        
        verts = pad_sequence(verts, batch_first=True, padding_value=0.0)        
        faces = pad_sequence(faces, batch_first=True, padding_value=-1)
            
        return torch.stack(diffusor), torch.stack(diffusor_origin), torch.stack(diffusor_end), verts, faces

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, shuffle=True, prefetch_factor=2, collate_fn=self.pad_verts_faces)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.pad_verts_faces)

