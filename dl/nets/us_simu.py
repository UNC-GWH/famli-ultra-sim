import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms as T

import numpy as np 

import pandas as pd

import pickle

import vtk
import math 
import SimpleITK as sitk

from .layers import TimeDistributed, ProjectionHead, AttentionChunk, SelfAttention, MHAContextModulated, MHABlock, OrientationPredictor
from .lotus import UltrasoundRenderingLinear

from lightning.pytorch import LightningModule

import monai 

class VolumeSamplingBlindSweep(nn.Module):
    def __init__(self, mount_point="/mnt/raid/C1_ML_Analysis", simulation_fov_grid_size=[64, 128, 128], simulation_fov_fn='simulated_data_export/studies_merged/simulation_fov.stl', simulation_ultrasound_plane_fn='simulated_data_export/studies_merged/simulation_ultrasound_plane.stl', random_probe_pos_factor=0.0005, random_rotation_angles_ranges=((-10, 10), (-10, 10), (-10, 10))):
        super().__init__()

        # The simulation_fov is a cube that represent the bounds of the simulation. 
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(mount_point, simulation_fov_fn))
        reader.Update()
        simulation_fov = reader.GetOutput()
        simulation_fov_bounds = torch.tensor(simulation_fov.GetBounds())
        self.register_buffer('simulation_fov_bounds', simulation_fov_bounds)

        simulation_fov_origin = torch.tensor(simulation_fov_bounds[[0,2,4]])
        self.register_buffer('simulation_fov_origin', simulation_fov_origin)

        simulation_fov_end = torch.tensor(simulation_fov_bounds[[1,3,5]])
        self.register_buffer('simulation_fov_end', simulation_fov_end)

        # The simulation_fov_grid_size is the size of the grid that we will use to sample the simulation_fov
        simulation_fov_grid_size = torch.tensor(simulation_fov_grid_size)
        self.register_buffer('simulation_fov_grid_size', simulation_fov_grid_size)

        # The simulation_ultrasound_plane is the plane where the ultrasound simulation will be performed        
        if os.path.splitext(simulation_ultrasound_plane_fn)[1] == ".stl":
            reader = vtk.vtkSTLReader()
            reader.SetFileName(os.path.join(mount_point, simulation_ultrasound_plane_fn))
            reader.Update()
            simulation_ultrasound_plane = reader.GetOutput()
        elif os.path.splitext(simulation_ultrasound_plane_fn)[1] == ".obj":
            reader = vtk.vtkOBJReader()
            reader.SetFileName(os.path.join(mount_point, simulation_ultrasound_plane_fn))
            reader.Update()
            simulation_ultrasound_plane = reader.GetOutput()
        self.simulation_ultrasound_plane_bounds = np.array(simulation_ultrasound_plane.GetBounds())        
        
        # This is the ultrasound plane or the output image size of the ultrasound simulation
        simulation_ultrasound_plane_mesh_grid_size = [256, 256, 1]
        simulation_ultrasound_plane_mesh_grid_params = [torch.arange(start=start, end=end, step=(end - start)/simulation_ultrasound_plane_mesh_grid_size[idx]) for idx, (start, end) in enumerate(zip(self.simulation_ultrasound_plane_bounds[[0,2,4]], self.simulation_ultrasound_plane_bounds[[1,3,5]]))]
        simulation_ultrasound_plane_mesh_grid = torch.stack(torch.meshgrid(simulation_ultrasound_plane_mesh_grid_params, indexing='ij'), dim=-1).squeeze().to(torch.float32)

        self.register_buffer('simulation_ultrasound_plane_mesh_grid', simulation_ultrasound_plane_mesh_grid)

        self.random_probe_pos_factor = random_probe_pos_factor
        self.random_rotation_angles_ranges = random_rotation_angles_ranges
        
    def init_probe_params(self, mount_point="/mnt/raid/C1_ML_Analysis", probe_params_csv='simulated_data_export/studies_merged/probe_params.csv'):
        self.probe_params_df = pd.read_csv(os.path.join(mount_point, probe_params_csv))

        self.tags = self.probe_params_df['tag'].unique()
        self.tags_dict = {tag: idx for idx, tag in enumerate(self.tags)}
        grouped_df = self.probe_params_df.groupby('tag')

        # Load the probe parameters for the different sweeps. The paths of the sweeps are predefined
        for tag, group in grouped_df:
            sorted_group = group.sort_values('idx')

            probe_directions = []
            probe_origins = []

            for idx, row in sorted_group.iterrows():
                probe_params = pickle.load(open(os.path.join(mount_point, row['probe_param_fn']), 'rb'))
        
                probe_direction = torch.tensor(probe_params['probe_direction'], dtype=torch.float32, requires_grad=False)
                probe_origin = torch.tensor(probe_params['probe_origin'], dtype=torch.float32, requires_grad=False)

                probe_directions.append(probe_direction.T.unsqueeze(0))
                probe_origins.append(probe_origin.unsqueeze(0))

            self.register_buffer(f'probe_directions_{tag}', torch.cat(probe_directions, dim=0))
            self.register_buffer(f'probe_origins_{tag}', torch.cat(probe_origins, dim=0))

    def init_probe_params_from_pos(self, probe_paths, tags=["M", "L0", "L1", "R0", "R1", "C1", "C2", "C3", "C4"]):
        
        self.tags = tags

        self.tags_dict = {tag: idx for idx, tag in enumerate(self.tags)}

        for tag in self.tags:
            probe_origins = np.load(os.path.join(probe_paths, tag + ".npy"))

            self.register_buffer(f'probe_origins_{tag}', torch.tensor(probe_origins, dtype=torch.float32, requires_grad=False))

            probe_directions = np.load(os.path.join(probe_paths, tag + "_rotations.npy"))
            self.register_buffer(f'probe_directions_{tag}', torch.tensor(probe_directions, dtype=torch.float32, requires_grad=False))
            

    def transform_simulation_ultrasound_plane_tag(self, tag, probe_origin_rand=None, probe_direction_rand=None, use_random=False):
        # Given a sweep tag, we get ALL the planes in that sweep. We can use a random displacement/rotation for the probe. The use_random flag is used to determine if we want to use random displacements/rotations
        
        if torch.is_tensor(tag):
            tag = tag.item()
            tag = self.tags[tag]
        probe_directions = getattr(self, f'probe_directions_{tag}')
        probe_origins = getattr(self, f'probe_origins_{tag}')

        simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane(probe_directions, probe_origins, probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)

        return simulation_ultrasound_plane_mesh_grid_transformed_t
    
    def transform_simulation_ultrasound_plane_norm(self, simulation_ultrasound_plane_mesh_grid_transformed, diffusor_origin, diffusor_end):
        # According to the documentation of F.grid_sample -> https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html, 
        # the values of the sampling grid must be between -1 and 1, i.e., the diffusor origin and end are [-1,-1,-1] and [1,1,1] respectively
        # We need transform the current set of transformed planes "simulation_ultrasound_plane_mesh_grid_transformed" to that space.
        diffusor_origin = diffusor_origin.view(diffusor_origin.shape[0], 1, 1, 1, 3)
        diffusor_end = diffusor_end.view(diffusor_end.shape[0], 1, 1, 1, 3)
        return 2.*(simulation_ultrasound_plane_mesh_grid_transformed - diffusor_origin)/(diffusor_end - diffusor_origin) - 1.

    def transform_simulation_ultrasound_plane(self, probe_directions, probe_origins, probe_origin_rand=None, probe_direction_rand=None, use_random=False):
        # Given probe directions and origins, we transform the default simulation ultrasound plane to those locations and directions
        simulation_ultrasound_plane_mesh_grid_transformed_t = []
        for probe_origin, probe_direction in zip(probe_origins, probe_directions): # We iterate through the BATCH otherwise this will not work for training
            simulation_ultrasound_plane_mesh_grid_transformed = self.transform_simulation_ultrasound_plane_single(probe_direction, probe_origin, probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)
            simulation_ultrasound_plane_mesh_grid_transformed_t.append(simulation_ultrasound_plane_mesh_grid_transformed.unsqueeze(0))
        
        return torch.cat(simulation_ultrasound_plane_mesh_grid_transformed_t, dim=0)
    
    def transform_simulation_ultrasound_plane_single(self, probe_direction, probe_origin, probe_origin_rand=None, probe_direction_rand=None, use_random=False):
        # In a typical sweep we have 150+ probe_directions/_origin
        # Given a probe directions and origins, we transform the simulation ultrasound plane to those locations/directions
        # The ouptut is a locations in 3D space where the ultrasound simulation will be performed. We can use this to sample the diffusor at those locations
        if probe_origin_rand is not None:
            probe_origin = probe_origin + probe_origin_rand
        if probe_direction_rand is not None:
            probe_direction = torch.matmul(probe_direction, probe_direction_rand)

        if use_random:
            probe_origin_single_rand = torch.rand(3, device=probe_origin.device)*self.random_probe_pos_factor            
            probe_direction_single_rand = self.random_affine_matrix(self.random_rotation_angles_ranges).to(probe_direction.device)[0:3,0:3]            

            probe_origin = probe_origin + probe_origin_single_rand
            probe_direction = torch.matmul(probe_direction, probe_direction_single_rand)

        simulation_ultrasound_plane_mesh_grid_transformed = torch.matmul(self.simulation_ultrasound_plane_mesh_grid, probe_direction.T) + probe_origin
        return simulation_ultrasound_plane_mesh_grid_transformed.to(torch.float32)
    
    def diffusor_sampling_tag(self, tag, diffusor_t, diffusor_origin, diffusor_end, probe_origin_rand=None, probe_direction_rand=None, use_random=False):
        # Given a sweep tag, we sample the diffusor at the simulation ultrasound plane locations. The diffusor_origin and end are used to place this volume in space. 
        simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane_tag(tag, probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)
        
        batch_size = diffusor_t.shape[0]
        simulation_ultrasound_plane_mesh_grid_transformed_norm_t = simulation_ultrasound_plane_mesh_grid_transformed_t.repeat(batch_size, 1, 1, 1, 1)
        simulation_ultrasound_plane_mesh_grid_transformed_norm_t = self.transform_simulation_ultrasound_plane_norm(simulation_ultrasound_plane_mesh_grid_transformed_norm_t, diffusor_origin, diffusor_end)
        
        return self.diffusor_sampling(diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_norm_t)
    
    def diffusor_sampling(self, diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_t):
        # sample the diffusor at the simulation ultrasound plane locations
        return F.grid_sample(diffusor_t, simulation_ultrasound_plane_mesh_grid_transformed_t, mode='nearest', align_corners=False)
    
    def diffusor_in_fov(self, diffusor_t, diffusor_origin, diffusor_end):
        # transforms the diffusor_t to the simulation fov. This is a resampling operation
        simulation_fov_origin = self.simulation_fov_bounds[[0,2,4]].flip(dims=[0])
        simulation_fov_end = self.simulation_fov_bounds[[1,3,5]].flip(dims=[0])
        
        simulation_fov_mesh_grid_params = [torch.arange(start=start, end=end, step=(end - start)/self.simulation_fov_grid_size[idx], device=self.simulation_fov_bounds.device) for idx, (start, end) in enumerate(zip(simulation_fov_origin, simulation_fov_end))]
        simulation_fov_mesh_grid = torch.stack(torch.meshgrid(simulation_fov_mesh_grid_params, indexing='ij'), dim=-1).squeeze().to(torch.float32)
        

        simulation_fov_mesh_grid_transformed = simulation_fov_mesh_grid.unsqueeze(0)
        repeats = [1,]*len(simulation_fov_mesh_grid_transformed.shape)
        repeats[0] = diffusor_t.shape[0]

        simulation_fov_mesh_grid_transformed = simulation_fov_mesh_grid_transformed.repeat(repeats)

        diffusor_origin = diffusor_origin.view(diffusor_t.shape[0], 1, 1, 1, 3)
        diffusor_end = diffusor_end.view(diffusor_t.shape[0], 1, 1, 1, 3)
        simulation_fov_mesh_grid_transformed = 2.*(simulation_fov_mesh_grid_transformed - diffusor_origin)/(diffusor_end - diffusor_origin) - 1.

        return F.grid_sample(diffusor_t.permute(0, 1, 4, 3, 2), simulation_fov_mesh_grid_transformed.float(), mode='nearest', align_corners=False)

    def simulated_sweep_in_fov(self, tag, sampled_sweep_simu):
        # Transform the simulated sweep to the simulation FOV
        # Get the default probe directions and origins
        assert len(sampled_sweep_simu.shape) == 5

        sampled_sweep_simu_shape = sampled_sweep_simu.shape[-3:]
        
        simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane_tag(tag)
        
        if simulation_ultrasound_plane_mesh_grid_transformed_t.shape[:3] != sampled_sweep_simu_shape:

                mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/s), device=sampled_sweep_simu.device) for s in simulation_ultrasound_plane_mesh_grid_transformed_t.shape[:3]]
                z, y, x = torch.meshgrid(mesh_grid_params, indexing='ij')
                mesh_grid = torch.stack([x, y, z], dim=-1).to(torch.float32).unsqueeze(0)

                repeats = [1,]*len(mesh_grid.shape)
                repeats[0] = sampled_sweep_simu.shape[0]

                mesh_grid = mesh_grid.repeat(repeats)

                sampled_sweep_simu = F.grid_sample(sampled_sweep_simu, mesh_grid, align_corners=True)
        
        simulation_ultrasound_plane_mesh_grid_transformed_t = simulation_ultrasound_plane_mesh_grid_transformed_t.flip(dims=[-1])
        # Get the origin and end of the simulation FOV
        simulation_fov_origin = self.simulation_fov_bounds[[0,2,4]].flip(dims=[0])
        simulation_fov_end = self.simulation_fov_bounds[[1,3,5]].flip(dims=[0])
        # Convert the transformed planes to an ijk mapping
        simulation_ultrasound_plane_mesh_grid_transformed_t_idx = torch.clip((simulation_ultrasound_plane_mesh_grid_transformed_t - simulation_fov_origin)/(simulation_fov_end - simulation_fov_origin), 0, 1)*(self.simulation_fov_grid_size - 1)
        simulation_ultrasound_plane_mesh_grid_transformed_t_idx = simulation_ultrasound_plane_mesh_grid_transformed_t_idx.to(torch.int)
        simulation_ultrasound_plane_mesh_grid_transformed_t_idx = simulation_ultrasound_plane_mesh_grid_transformed_t_idx.reshape(-1, 3)

        out_fovs = []
        for ss_t in sampled_sweep_simu: #iterate through the batch
            
            out_fov = torch.zeros(self.simulation_fov_grid_size.tolist()).to(torch.float).to(self.simulation_fov_grid_size.device)
            # Here the trick is that the simulation_ultrasound_plane_mesh_grid_transformed_t_idx has the same shape as the sampled_sweep_simu. Some idx will be repeated/overwritten. 
            out_fov[simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,0], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,1], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,2]] = ss_t.reshape(-1)
            out_fovs.append(out_fov.unsqueeze(0))
        out_fovs = torch.cat(out_fovs, dim=0)

        return out_fovs
    
    def fov_physical(self, simulation_fov_grid_size=None):

        if simulation_fov_grid_size is None:
            simulation_fov_grid_size = self.simulation_fov_grid_size

        simulation_fov_mesh_grid_params = [torch.arange(end=s, device=self.simulation_fov_bounds.device) for s in simulation_fov_grid_size]
        simulation_fov_mesh_grid_idx = torch.stack(torch.meshgrid(simulation_fov_mesh_grid_params, indexing='ij'), dim=-1).squeeze().to(torch.float32)

        simulation_fov_origin = self.simulation_fov_bounds[[0,2,4]]
        simulation_fov_end = self.simulation_fov_bounds[[1,3,5]]
        simulation_fov_size = simulation_fov_grid_size.flip(dims=[0])

        simulation_fov_spacing = (simulation_fov_end - simulation_fov_origin)/simulation_fov_size
        return simulation_fov_origin + simulation_fov_mesh_grid_idx*simulation_fov_spacing
    
    def transform_fov_norm(self, X_physical):
        simulation_fov_origin = self.simulation_fov_bounds[[0,2,4]]
        simulation_fov_end = self.simulation_fov_bounds[[1,3,5]]

        return 2.*(X_physical - simulation_fov_origin)/(simulation_fov_end - simulation_fov_origin) - 1.

    
    def diffusor_resample(self, diffusor_t, size=128):

        simulation_fov_mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/size), device=diffusor_t.device) for _ in range(3)]

        simulation_fov_mesh_grid = torch.stack(torch.meshgrid(simulation_fov_mesh_grid_params, indexing='ij'), dim=-1).to(torch.float32).unsqueeze(0)

        repeats = [1,]*len(simulation_fov_mesh_grid.shape)
        repeats[0] = diffusor_t.shape[0]

        simulation_fov_mesh_grid = simulation_fov_mesh_grid.repeat(repeats)

        return F.grid_sample(diffusor_t.permute(0, 1, 4, 3, 2), simulation_fov_mesh_grid.float(), mode='nearest', align_corners=False)
    
    def diffusor_tag_resample(self, diffusor_t, tag):

        assert len(diffusor_t.shape) == 5

        size = diffusor_t.shape[2:5]

        mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/size[i]), device=diffusor_t.device) for i in range(3)]
        mesh_grid = torch.stack(torch.meshgrid(mesh_grid_params, indexing='ij'), dim=-1).to(torch.float32).unsqueeze(0)

        repeats = [1,]*len(mesh_grid.shape)
        repeats[0] = len(tag)

        mesh_grid = mesh_grid.repeat(repeats)

        x_v = []
        for t in tag:
            x_v.append(self.transform_simulation_ultrasound_plane_tag(t).permute(3, 0, 1, 2))
        x_v = torch.stack(x_v)

        return F.grid_sample(x_v, mesh_grid.float(), mode='nearest', align_corners=False).flatten(2).permute(0, 2, 1)

    def get_rotation_matrix(self, angle, axis):
        
        if axis == 'x':
            return torch.tensor([
                [1, 0, 0, 0],
                [0, math.cos(angle), -math.sin(angle), 0],
                [0, math.sin(angle), math.cos(angle), 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'y':
            return torch.tensor([
                [math.cos(angle), 0, math.sin(angle), 0],
                [0, 1, 0, 0],
                [-math.sin(angle), 0, math.cos(angle), 0],
                [0, 0, 0, 1]
            ])
        elif axis == 'z':
            return torch.tensor([
                [math.cos(angle), -math.sin(angle), 0, 0],
                [math.sin(angle), math.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
    
    def random_affine_matrix(self, rotation_ranges):
        """
        Generate a random affine transformation matrix with rotation ranges specified.

        Args:
            rotation_ranges (tuple of tuples): ((min_angle_x, max_angle_x), (min_angle_y, max_angle_y), (min_angle_z, max_angle_z))
                                            Angles should be in degrees.

        Returns:
            torch.Tensor: A 4x4 affine transformation matrix.
        """

        rotation_angles = [torch.FloatTensor(1).uniform_(*range).item() for range in rotation_ranges]

        Rx = self.get_rotation_matrix(math.radians(rotation_angles[0]), 'x')
        Ry = self.get_rotation_matrix(math.radians(rotation_angles[1]), 'y')
        Rz = self.get_rotation_matrix(math.radians(rotation_angles[2]), 'z')

        affine_matrix = torch.mm(torch.mm(Rz, Ry), Rx)

        return affine_matrix

    def random_rotate_3d_batch(self, x):
        """
        Randomly rotates a batch of 3D image tensors and returns the rotated images along with the rotation matrices.
        
        Args:
            x (torch.Tensor): A tensor of shape (B, C, D, H, W).
        
        Returns:
            rotated_images (torch.Tensor): Rotated 3D tensors of shape (B, C, D, H, W).
            rotation_matrices (torch.Tensor): Rotation matrices of shape (B, 3, 3).
        """
        assert len(x.shape) == 5, "Input tensor must be 5D (B, C, D, H, W)."

        B, C, D, H, W = x.shape

        # Generate random angles for each batch (B, 3)
        angles = torch.rand(B, 3) * 2 * torch.pi  # Random angles between 0 and 2*pi

        # Generate rotation matrices for each sample in the batch
        rotation_matrices = torch.stack([
            self.get_rotation_matrix(angles[i, 2], 'z') @ 
            self.get_rotation_matrix(angles[i, 1], 'y') @ 
            self.get_rotation_matrix(angles[i, 0], 'x')
            for i in range(B)
        ]).to(x.device)

        # Convert 4x4 rotation matrices to 3x4 affine matrices
        affine_matrices = rotation_matrices.to(x.device)[:,:3,:]

        # Generate affine grids for each batch
        grids = F.affine_grid(
            affine_matrices,
            size=x.size(),
            align_corners=False
        )

        # Apply rotations using grid sampling
        x_rotated = F.grid_sample(
            x,
            grids,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )

        return x_rotated, rotation_matrices

    def apply_batch_rotation(self, V, rotation_matrices):
        """
        Applies a batch of rotation matrices to a batch of 3D points.

        Args:
            V (torch.Tensor): A tensor of shape (BS, N, 3) containing 3D points.
            rotation_matrices (torch.Tensor): A tensor of shape (BS, 3, 3) containing rotation matrices.

        Returns:
            rotated_points (torch.Tensor): A tensor of shape (BS, N, 3) with rotated points.
        """
        assert V.shape[-1] == 3, "Points tensor must have the last dimension of size 3 (3D coordinates)."
        assert rotation_matrices.shape[-2:] == (3, 3), "Rotation matrices must have shape (BS, 3, 3)."
        assert V.shape[0] == rotation_matrices.shape[0], "Batch size of points and rotation matrices must match."

        # Apply the rotation matrix to each batch element
        return torch.matmul(V, rotation_matrices)  # Transpose for proper multiplication but here we don't transpose because the V tensor is ordered XYZ while the rotation matrix is ordered ZYX
        # rotated_points = torch.matmul(V, rotation_matrices.transpose(1, 2))  # Transpose for proper multiplication
        # return rotated_points
    
    def get_sweep(self, X, X_origin, X_end, tag, use_random=False, simulator=None, grid=None, inverse_grid=None, mask_fan=None, return_masked=False, return_sampled=False):        

        probe_origin_rand = None
        probe_direction_rand = None

        if mask_fan == None and simulator is not None and return_masked:
            mask_fan = simulator.module.USR.mask_fan

        if use_random:
            probe_origin_rand = torch.rand(3, device=X.device)*0.0001
            probe_origin_rand = probe_origin_rand
            rotation_ranges = ((-5, 5), (-5, 5), (-10, 10))  # ranges in degrees for x, y, and z rotations
            probe_direction_rand = self.random_affine_matrix(rotation_ranges).to(X.device)[0:3,0:3]                        
        
        sampled_sweep = self.diffusor_sampling_tag(tag, X.to(torch.float), X_origin.to(torch.float), X_end.to(torch.float), probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)

        if simulator is not None:
            with torch.no_grad():

                sampled_sweep_simu = []

                for ss in sampled_sweep:
                    ss_chunk = []
                    for ss_ch in torch.chunk(ss, chunks=20, dim=1):
                        simulated = simulator(ss_ch.unsqueeze(dim=1), grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
                        if return_masked:
                            simulated = simulated*mask_fan
                        simulated = simulator.module.transform_us(simulated)
                        ss_chunk.append(simulated)
                    sampled_sweep_simu.append(torch.cat(ss_chunk, dim=2))

                sampled_sweep_simu = torch.stack(sampled_sweep_simu).detach()

                if return_sampled:
                    return sampled_sweep, sampled_sweep_simu

                return sampled_sweep_simu
        
        return sampled_sweep

    def embed_sweep(self, tag, sampled_sweep_simu):
        """ Embed the sweep with simulation FOV coordiantes
            Args: 
                tag: The tag of the sweep
                sampled_sweep_simu: The sampled sweep simulation with shape B, C, D, H, W
        """
        assert len(sampled_sweep_simu.shape) == 5 

        sampled_sweep_simu_shape = sampled_sweep_simu.shape[-3:]
        
        simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane_tag(tag)
        simulation_ultrasound_plane_mesh_grid_transformed_t = simulation_ultrasound_plane_mesh_grid_transformed_t.permute(3, 0, 1, 2).unsqueeze(0)

        if simulation_ultrasound_plane_mesh_grid_transformed_t.shape[2:] != sampled_sweep_simu_shape:

            mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/s), device=sampled_sweep_simu.device) for s in sampled_sweep_simu_shape]
            z, y, x = torch.meshgrid(mesh_grid_params, indexing='ij')
            mesh_grid = torch.stack([x, y, z], dim=-1).to(torch.float32).unsqueeze(0)
            simulation_ultrasound_plane_mesh_grid_transformed_t = F.grid_sample(simulation_ultrasound_plane_mesh_grid_transformed_t, mesh_grid, align_corners=True)
        
        repeats = [1,]*len(simulation_ultrasound_plane_mesh_grid_transformed_t.shape)
        repeats[0] = sampled_sweep_simu.shape[0]
        simulation_ultrasound_plane_mesh_grid_transformed_t = simulation_ultrasound_plane_mesh_grid_transformed_t.repeat(repeats)

        return torch.cat([sampled_sweep_simu, simulation_ultrasound_plane_mesh_grid_transformed_t], dim=1)
    
class SweepSampling(nn.Module):
    def __init__(self, diffusor_fn, probe_paths, mount_point='/mnt/raid/C1_ML_Analysis/', grid_w=256, grid_h=256, center_x=128, center_y=-40, r1=20.0, r2=255.0, theta=np.pi / 4.25, padding=55, params_csv='simulated_data_export/animation_export/shapes_intensity_map_nrrd_speckel.csv', *args, **kwargs):
        super().__init__()
        
        self.USR = UltrasoundRenderingLinear(grid_w=grid_w, grid_h=grid_h, center_x=center_x, center_y=center_y, r1=r1, r2=r2, theta=theta, num_labels=10)
        df = pd.read_csv(os.path.join(mount_point, params_csv))
        self.USR.init_params(torch.tensor(df['mean']), torch.tensor(df['stddev']))
        self.USR.transform_us = T.Compose([T.Pad((0, padding, 0, 0)), T.CenterCrop((grid_h, grid_w))])
        self.simulator = TimeDistributed(self.USR, time_dim=2)
        self.vs = VolumeSamplingBlindSweep(mount_point=mount_point, simulation_fov_fn='simulated_data_export/animation_export/simulation_fov.stl', simulation_ultrasound_plane_fn='simulated_data_export/animation_export/ultrasound_grid.stl')


        diffusor = sitk.ReadImage(diffusor_fn)
        diffusor_t = torch.tensor(sitk.GetArrayFromImage(diffusor).astype(int))
        diffusor_size = torch.tensor(diffusor.GetSize())
        diffusor_spacing = torch.tensor(diffusor.GetSpacing())
        diffusor_origin = torch.tensor(diffusor.GetOrigin())
        diffusor_end = diffusor_origin + diffusor_spacing * diffusor_size
        self.register_buffer('diffusor_t', diffusor_t.unsqueeze(0).unsqueeze(0))
        self.register_buffer('diffusor_origin', diffusor_origin.unsqueeze(0))
        self.register_buffer('diffusor_end', diffusor_end.unsqueeze(0))


        self.vs.init_probe_params_from_pos(probe_paths)
    
    # This is only called during inference time to set a custom grid
    def init_grid(self, w, h, center_x, center_y, r1, r2, theta, padding=80):
        grid = self.USR.compute_grid(w, h, center_x, center_y, r1, r2, theta)
        inverse_grid, mask = self.USR.compute_grid_inverse(grid)
        
        self.USR.grid = self.USR.normalize_grid(grid)
        self.USR.inverse_grid = self.USR.normalize_grid(inverse_grid)
        self.USR.mask_fan = mask

        self.USR.transform_us = T.Compose([T.Pad((0, padding, 0, 0)), T.CenterCrop((h, w))])

    def volume_sampling(self):
        with torch.no_grad():
            simulator = self.simulator
            
            grid = None
            inverse_grid = None
            mask_fan = None

            tags = self.vs.tags

            X_sweeps = []
            X_sweeps_simu = []
            X_sweeps_tags = []            

            for tag in tags:                
                
                sampled_sweep, sampled_sweep_simu = self.vs.get_sweep(self.diffusor_t, self.diffusor_origin, self.diffusor_end, tag, simulator=simulator, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan, return_sampled=True)
                
                X_sweeps.append(sampled_sweep)
                X_sweeps_simu.append(sampled_sweep_simu)
                X_sweeps_tags.append(self.vs.tags_dict[tag])
            
            X_sweeps = torch.cat(X_sweeps, dim=1).to(torch.float32)
            X_sweeps_simu = torch.cat(X_sweeps_simu, dim=1).to(torch.float32)
            X_sweeps_tags = torch.tensor(X_sweeps_tags, device=self.diffusor_t.device)

            return X_sweeps, X_sweeps_simu/255.0, X_sweeps_tags
        
    def adjust_contrast(self, x, factor=1.0):        
        mean = x.mean()
        return torch.clamp((x - mean) * factor + mean, min=0.0, max=1.0)

    def adjust_gain(self, x, factor=1.0):
        return torch.clamp(x * factor, min=0.0, max=1.0)

    def adjust_depth_gain(self, x, base_gain=1.0, slope=0.01):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        depth_profile = base_gain + slope * torch.linspace(0, 1, H, device=x.device).view(1, 1, 1, H, 1)
        return torch.clamp(x * depth_profile, min=0.0, max=1.0)

    def forward(self, X):
        return self.G(X)
    
class USBabyFrame(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        encoder = monai.networks.nets.EfficientNetBN('efficientnet-b0', pretrained=True, spatial_dims=2, in_channels=self.hparams.in_channels, num_classes=self.hparams.features)
        self.encoder = TimeDistributed(encoder, time_dim=2)

        p_encoding = torch.stack([self.positional_encoding(self.hparams.time_steps, self.hparams.features, tag) for tag in self.hparams.tags])        
        self.register_buffer("p_encoding", p_encoding)

        p_encoding_z = torch.stack([self.positional_encoding(self.hparams.n_chunks, self.hparams.embed_dim, tag) for tag in self.hparams.tags])        
        self.register_buffer("p_encoding_z", p_encoding_z)
        
        self.proj = ProjectionHead(input_dim=self.hparams.features, hidden_dim=self.hparams.features, output_dim=self.hparams.embed_dim, activation=nn.LeakyReLU)
        self.attn_chunk = AttentionChunk(input_dim=self.hparams.embed_dim, hidden_dim=64, chunks=self.hparams.n_chunks, time_dim=1)
        self.mha = MHABlock(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, dropout=self.hparams.dropout, causal_mask=True, return_weights=False)

        self.dropout = nn.Dropout(self.hparams.dropout)

        # self.pred = OrientationPredictor(input_dim=self.hparams.embed_dim)
        # 
        # self.mha = nn.MultiheadAttention(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, dropout=self.hparams.dropout, bias=False, batch_first=True)
        # self.mha = MHAContextModulated(embed_dim=self.hparams.embed_dim, num_heads=self.hparams.num_heads, output_dim=self.hparams.embed_dim, dropout=self.hparams.dropout)        
        
        self.attn = SelfAttention(input_dim=self.hparams.embed_dim, hidden_dim=64)
        self.proj_final = ProjectionHead(input_dim=self.hparams.embed_dim, hidden_dim=64, output_dim=4)

        # self.loss_fn = nn.CosineSimilarity(dim=2)
        # self.loss_fn = nn.MSELoss()

        
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("Calculate baby frame of orientation")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=1e-5)
        
        # Image Encoder parameters 
        group.add_argument("--spatial_dims", type=int, default=2, help='Spatial dimensions for the encoder')
        group.add_argument("--in_channels", type=int, default=1, help='Input channels for encoder')
        group.add_argument("--features", type=int, default=1280, help='Number of output features for the encoder')        
        group.add_argument("--n_chunks_e", type=int, default=16, help='Number of chunks in the encoder stage to reduce memory usage')
        group.add_argument("--n_chunks", type=int, default=64, help='Number of outputs in the time dimension')
        group.add_argument("--num_heads", type=int, default=8, help='Number of heads for multi_head attention')

        # Encoder parameters for the diffusion model        
        group.add_argument("--input_dim", type=int, default=3, help='Input dimension for the encoder')
        group.add_argument("--embed_dim", type=int, default=128, help='Embedding dimension')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
        group.add_argument("--dropout", type=float, default=0.25, help='Dropout rate')
        
        group.add_argument("--time_steps", type=int, default=128, help='Number of time steps in the sweep or sequence length')
        group.add_argument("--tags", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8], help='List of tags for the sequences')

        return parent_parser
    
    def positional_encoding(self, seq_len: int, d_model: int, tag: int) -> torch.Tensor:
        """
        Sinusoidal positional encoding with tag-based offset.

        Args:
            seq_len (int): Sequence length.
            d_model (int): Embedding dimension.
            tag (int): Unique tag for the sequence.
            device (str): Device to store the tensor.

        Returns:
            torch.Tensor: Positional encoding (seq_len, d_model).
        """
        pe = torch.zeros(seq_len, d_model)
        
        # Offset positions by a tag-dependent amount to make each sequence encoding unique
        position = torch.arange(tag * seq_len, (tag + 1) * seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def orthogonality_loss(self, R):
        I = torch.eye(3, device=self.device).unsqueeze(0)
        RtR = torch.bmm(R.transpose(1, 2), R)
        return ((RtR - I)**2).mean()
    
    def rodrigues(self, X_hat):
        """
        Convert axis-angle vector to rotation matrix using Rodrigues' formula.
        
        Args:
            X_hat (torch.Tensor): [B, 3] axis-angle vector (angle = norm, axis = direction)
            
        Returns:
            R (torch.Tensor): [B, 3, 3] rotation matrix
        """
        B = X_hat.shape[0]
        angle = torch.norm(X_hat, dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1]
        axis = X_hat / angle  # [B, 3]

        # Components for Rodrigues' formula
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        zeros = torch.zeros_like(x)
        K = torch.stack([
            zeros, -z,    y,
            z,   zeros, -x,
            -y,    x,   zeros
        ], dim=1).reshape(B, 3, 3)  # [B, 3, 3]

        I = torch.eye(3, device=self.device).unsqueeze(0).expand(B, -1, -1)  # [B, 3, 3]
        sin = torch.sin(angle).unsqueeze(-1)  # [B, 1, 1]
        cos = torch.cos(angle).unsqueeze(-1)  # [B, 1, 1]

        R = I + sin * K + (1 - cos) * torch.bmm(K, K)  # [B, 3, 3]
        return R
    
    def quaternion_to_rotation_matrix(self, q):  # [B, 4]
        q = F.normalize(q, dim=-1)
        w, x, y, z = q.unbind(-1)

        B = q.shape[0]
        R = torch.empty((B, 3, 3), device=self.device)
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        return R

    def compute_rotation_matrix_from_6d(self, X_hat):  # [B, 6]
        a1 = X_hat[:, 0:3]
        a2 = X_hat[:, 3:6]

        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        return torch.stack([b1, b2, b3], dim=-1)  # [B, 3, 3]
    
    def geodesic_loss(self, Y, X_hat):
        R_diff = torch.bmm(X_hat.transpose(1, 2), Y)
        trace = R_diff.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
        theta = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        return theta.mean()
    
    def compute_loss(self, Y, X_hat, step="train", sync_dist=False):
        
        # frobenius norm
        loss = ((X_hat - Y)**2).mean()        

        # geod_loss = self.geodesic_loss(Y, X_hat)
        # loss = loss + geod_loss

        # loss_orth = self.orthogonality_loss(X_hat) 
        # loss = loss + loss_orth
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)
        # self.log(f"{step}_loss_geodesic", geod_loss, sync_dist=sync_dist)
        # self.log(f"{step}_loss_orth", loss_orth, sync_dist=sync_dist)
        # self.log(f"{step}_loss_mean", loss_mean, sync_dist=sync_dist)
        # self.log(f"{step}_loss_std", loss_std, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X, tags, Y = train_batch

        x_hat = self(X, tags)

        return self.compute_loss(Y=Y, X_hat=x_hat, step="train")

    def validation_step(self, val_batch, batch_idx):
        
        X, tags, Y = val_batch

        x_hat = self(X, tags)     

        return self.compute_loss(Y=Y, X_hat=x_hat, step="val", sync_dist=True)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        z = []
        for x_chunk in x.chunk(self.hparams.n_chunks_e, dim=2):            
            z.append(self.encoder(x_chunk))
        z = torch.cat(z, dim=2)

        return z

    def forward(self, x_sweeps: torch.tensor, sweeps_tags: torch.tensor):
        
        batch_size = x_sweeps.shape[0]
        
        # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
        # tags shape torch.Size([2, 2])
        Nsweeps = x_sweeps.shape[1] # Number of sweeps -> T

        z = []
        x_v = []

        for n in range(Nsweeps):
            x_sweeps_n = x_sweeps[:, n, :, :, :, :] # [BS, C, T, H, W]
            
            tag = sweeps_tags[:,n]            

            z_ = self.encode(x_sweeps_n) # [BS, T, self.hparams.features]
            z_ = z_.permute(0, 2, 1) # Permute the time dim with the output features. -> Shape is now [BS, T, F]
            
            p_enc = self.p_encoding[tag]
            
            z_ = z_ + p_enc # [BS, T, self.hparams.features]            

            z_ = self.proj(z_) # [BS, self.hparams.n_chunks, self.hparams.embed_dim]

            z_ = self.attn_chunk(z_) # [BS, self.hparams.n_chunks, self.hparams.features]

            p_enc_z = self.p_encoding_z[tag]
            
            z_ = z_ + p_enc_z
            z_ = self.mha(z_) # [BS, self.hparams.n_chunks, self.hparams.embed_dim]
            z_ = z_ + p_enc_z
            
            z.append(z_)

        z = torch.stack(z, dim=1) # [BS, N, self.hparams.n_chunks, 64*64*self.hparams.latent_channels]
        z = z.view(batch_size, -1, self.hparams.embed_dim).contiguous()        

        # z = self.mha(z)
        z, z_s = self.attn(z, z)
        
        x_hat = self.proj_final(z)
        x_hat = self.quaternion_to_rotation_matrix(x_hat)  

        return x_hat