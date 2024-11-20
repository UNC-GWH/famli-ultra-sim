import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
# from torch.cuda.amp import autocast, GradScaler

from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets
from generative.networks.nets.autoencoderkl import Encoder



import numpy as np 
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from torchvision import transforms as T
from torchvision import models as tv_models
from torchvision.ops import Conv2dNormActivation

import monai
from monai.networks import nets as monai_nets
from monai.losses import DiceFocalLoss, DiceLoss, DiceCELoss

import lightning as L
from lightning.pytorch.core import LightningModule

import pandas as pd

import pickle

import vtk
import math 

from .layers import TimeDistributed, MultiHeadAttention3D, ScoreLayer
from .us_simulation_jit import MergedLinearCut1, MergedLinearCutLabel11, MergedCutLabel11, MergedUSRLabel11, MergedLinearLabel11


from monai.transforms import (            
    Compose,
    RandAffine,       
    RandAxisFlip, 
    RandRotate,    
    RandZoom,
)

from pytorch3d.loss import (
    chamfer_distance,
    point_mesh_edge_distance, 
    point_mesh_face_distance,
)

from pytorch3d.structures import (
    Meshes,
    Pointclouds)

from pytorch3d.ops import (sample_points_from_meshes,
                           knn_points, 
                           knn_gather)

import sys
sys.path.append('/mnt/raid/C1_ML_Analysis/source/ShapeAXI/src/')

from shapeaxi.saxi_nets import MHAEncoder_V, MHAIdxEncoder, MHAIdxDecoder
from shapeaxi.saxi_layers import AttentionChunk, FeedForward, SelfAttention, Residual

from torch.nn.utils.rnn import pad_sequence

class VolumeSamplingBlindSweep(nn.Module):
    def __init__(self, mount_point, simulation_fov_grid_size=[64, 128, 128], probe_params_csv='simulated_data_export/studies_merged/probe_params.csv', simulation_fov_fn='simulated_data_export/studies_merged/simulation_fov.stl', simulation_ultrasound_plane_fn='simulated_data_export/studies_merged/simulation_ultrasound_plane.stl', random_probe_pos_factor=0.0005, random_rotation_angles_ranges=((-10, 10), (-10, 10), (-10, 10))):
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
        reader = vtk.vtkSTLReader()
        reader.SetFileName(os.path.join(mount_point, simulation_ultrasound_plane_fn))
        reader.Update()
        simulation_ultrasound_plane = reader.GetOutput()
        self.simulation_ultrasound_plane_bounds = np.array(simulation_ultrasound_plane.GetBounds())

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
        
        # This is the ultrasound plane or the output image size of the ultrasound simulation
        simulation_ultrasound_plane_mesh_grid_size = [256, 256, 1]
        simulation_ultrasound_plane_mesh_grid_params = [torch.arange(start=start, end=end, step=(end - start)/simulation_ultrasound_plane_mesh_grid_size[idx]) for idx, (start, end) in enumerate(zip(self.simulation_ultrasound_plane_bounds[[0,2,4]], self.simulation_ultrasound_plane_bounds[[1,3,5]]))]
        simulation_ultrasound_plane_mesh_grid = torch.stack(torch.meshgrid(simulation_ultrasound_plane_mesh_grid_params), dim=-1).squeeze().to(torch.float32)

        self.register_buffer('simulation_ultrasound_plane_mesh_grid', simulation_ultrasound_plane_mesh_grid)

        self.random_probe_pos_factor = random_probe_pos_factor
        self.random_rotation_angles_ranges = random_rotation_angles_ranges
        

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
            probe_direction_single_rand = self.random_affine_matrix(self.random_rotation_angles_ranges).to(probe_direction.device)

            probe_origin = probe_origin + probe_origin_single_rand
            probe_direction = torch.matmul(probe_direction, probe_direction_single_rand)

        simulation_ultrasound_plane_mesh_grid_transformed = torch.matmul(self.simulation_ultrasound_plane_mesh_grid, probe_direction) + probe_origin
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
        simulation_fov_mesh_grid = torch.stack(torch.meshgrid(simulation_fov_mesh_grid_params), dim=-1).squeeze().to(torch.float32)
        

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
        simulation_ultrasound_plane_mesh_grid_transformed_t = self.transform_simulation_ultrasound_plane_tag(tag)

        simulation_ultrasound_plane_mesh_grid_transformed_t = simulation_ultrasound_plane_mesh_grid_transformed_t.flip(dims=[-1])

        # Get the origin and end of the simulation FOV
        simulation_fov_origin = self.simulation_fov_bounds[[0,2,4]].flip(dims=[0])
        simulation_fov_end = self.simulation_fov_bounds[[1,3,5]].flip(dims=[0])
        # Convert the transformed planes to an ijk mapping
        simulation_ultrasound_plane_mesh_grid_transformed_t_idx = torch.clip((simulation_ultrasound_plane_mesh_grid_transformed_t - simulation_fov_origin)/(simulation_fov_end - simulation_fov_origin), 0, 1)*(self.simulation_fov_grid_size - 1)
        simulation_ultrasound_plane_mesh_grid_transformed_t_idx = simulation_ultrasound_plane_mesh_grid_transformed_t_idx.to(torch.int)
        simulation_ultrasound_plane_mesh_grid_transformed_t_idx = simulation_ultrasound_plane_mesh_grid_transformed_t_idx.reshape(-1, 3)

        out_fovs = []
        for ss_t in sampled_sweep_simu:
            out_fov = torch.zeros(self.simulation_fov_grid_size.tolist()).to(torch.float).to(self.simulation_fov_grid_size.device)
            # # print(sampled_sweep.shape)
            # print(out_fov[simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,0], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,1], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,2]].shape)
            # print(out_fov[simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,0], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,1], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,2]].shape)
            out_fov[simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,0], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,1], simulation_ultrasound_plane_mesh_grid_transformed_t_idx[:,2]] = ss_t.reshape(-1)
            out_fovs.append(out_fov.unsqueeze(0))
        out_fovs = torch.cat(out_fovs, dim=0)

        return out_fovs
    
    def fov_physical(self, simulation_fov_grid_size=None):

        if simulation_fov_grid_size is None:
            simulation_fov_grid_size = self.simulation_fov_grid_size

        simulation_fov_mesh_grid_params = [torch.arange(end=s, device=self.simulation_fov_bounds.device) for s in simulation_fov_grid_size]
        simulation_fov_mesh_grid_idx = torch.stack(torch.meshgrid(simulation_fov_mesh_grid_params), dim=-1).squeeze().to(torch.float32)

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

        simulation_fov_mesh_grid = torch.stack(torch.meshgrid(simulation_fov_mesh_grid_params), dim=-1).to(torch.float32).unsqueeze(0)

        repeats = [1,]*len(simulation_fov_mesh_grid.shape)
        repeats[0] = diffusor_t.shape[0]

        simulation_fov_mesh_grid = simulation_fov_mesh_grid.repeat(repeats)

        return F.grid_sample(diffusor_t.permute(0, 1, 4, 3, 2), simulation_fov_mesh_grid.float(), mode='nearest', align_corners=False)
    
    def diffusor_tag_resample(self, diffusor_t, tag):

        assert len(diffusor_t.shape) == 5

        size = diffusor_t.shape[2:5]

        mesh_grid_params = [torch.arange(start=-1.0, end=1.0, step=(2.0/size[i]), device=diffusor_t.device) for i in range(3)]
        mesh_grid = torch.stack(torch.meshgrid(mesh_grid_params), dim=-1).to(torch.float32).unsqueeze(0)

        repeats = [1,]*len(mesh_grid.shape)
        repeats[0] = len(tag)

        mesh_grid = mesh_grid.repeat(repeats)

        x_v = []
        for t in tag:
            x_v.append(self.transform_simulation_ultrasound_plane_tag(t).permute(3, 0, 1, 2))
        x_v = torch.stack(x_v)

        return F.grid_sample(x_v, mesh_grid.float(), mode='nearest', align_corners=False).flatten(2).permute(0, 2, 1)
    
    def random_affine_matrix(self, rotation_ranges):
        """
        Generate a random affine transformation matrix with rotation ranges specified.

        Args:
            rotation_ranges (tuple of tuples): ((min_angle_x, max_angle_x), (min_angle_y, max_angle_y), (min_angle_z, max_angle_z))
                                            Angles should be in degrees.

        Returns:
            torch.Tensor: A 4x4 affine transformation matrix.
        """
        def get_rotation_matrix(angle, axis):
            angle = math.radians(angle)
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

        rotation_angles = [torch.FloatTensor(1).uniform_(*range).item() for range in rotation_ranges]

        Rx = get_rotation_matrix(rotation_angles[0], 'x')
        Ry = get_rotation_matrix(rotation_angles[1], 'y')
        Rz = get_rotation_matrix(rotation_angles[2], 'z')

        affine_matrix = torch.mm(torch.mm(Rz, Ry), Rx)

        return affine_matrix[0:3,0:3]
    
    def get_sweep(self, X, X_origin, X_end, tag, use_random=False, simulator=None, grid=None, inverse_grid=None, mask_fan=None):        

        probe_origin_rand = None
        probe_direction_rand = None            

        if use_random:
            probe_origin_rand = torch.rand(3, device=self.device)*0.0001
            probe_origin_rand = probe_origin_rand
            rotation_ranges = ((-5, 5), (-5, 5), (-10, 10))  # ranges in degrees for x, y, and z rotations
            probe_direction_rand = self.random_affine_matrix(rotation_ranges).to(self.device)            
        
        sampled_sweep = self.diffusor_sampling_tag(tag, X.to(torch.float), X_origin.to(torch.float), X_end.to(torch.float), probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)

        if simulator is not None:
            with torch.no_grad():

                sampled_sweep_simu = []

                for ss in sampled_sweep:
                    ss_chunk = []
                    for ss_ch in torch.chunk(ss, chunks=20, dim=1):
                        simulated = simulator(ss_ch.unsqueeze(dim=1), grid, inverse_grid, mask_fan)                        
                        ss_chunk.append(simulated)
                    sampled_sweep_simu.append(torch.cat(ss_chunk, dim=2))

                return torch.stack(sampled_sweep_simu).detach()
        
        return sampled_sweep

class USPCReconstruction(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_i = TimeDistributed(Encoder(spatial_dims=2,
            in_channels=1,
            num_channels=[8, 16, 32, 64, 128],
            out_channels=self.hparams.latent_channels,
            num_res_blocks=[2, 2, 2, 2, 2],
            norm_num_groups=8,
            norm_eps=1e-3,
            attention_levels=[False, False, False, False, True],
            with_nonlocal_attn=True), time_dim=2) # [BS, latent_channels, T, H, W]

        self.attn_chunk = AttentionChunk(input_dim=(self.hparams.latent_channels*16*16), hidden_dim=64, chunks=16)

        self.encoder_v = MHAIdxEncoder(input_dim=self.hparams.input_dim, 
                                     output_dim=self.hparams.stages[-1], 
                                     K=self.hparams.K, 
                                     num_heads=self.hparams.num_heads, 
                                     stages=self.hparams.stages, 
                                     dropout=self.hparams.dropout, 
                                     pooling_factor=self.hparams.pooling_factor, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim,
                                     score_pooling=self.hparams.score_pooling,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim, 
                                    #  use_skip_connection=self.hparams.use_skip_connection, 
                                     return_v=True)
        
        self.fc = nn.Linear(self.hparams.stages[-1], self.hparams.output_dim, bias=False)
        
        # self.ff_mu = FeedForward(self.hparams.stages[-1], hidden_dim=self.hparams.stages[-1], dropout=self.hparams.dropout)
        # self.ff_sigma = FeedForward(self.hparams.stages[-1], hidden_dim=self.hparams.stages[-1], dropout=self.hparams.dropout)
        
        # self.decoder_v = MHAIdxDecoder(input_dim=self.hparams.stages[-1], 
        #                              output_dim=self.hparams.output_dim, 
        #                              K=self.hparams.K[::-1], 
        #                              num_heads=self.hparams.num_heads[::-1], 
        #                              stages=self.hparams.stages[::-1], 
        #                              dropout=self.hparams.dropout, 
        #                              pooling_hidden_dim=self.hparams.pooling_hidden_dim[::-1] if self.hparams.pooling_hidden_dim is not None else None,
        #                              feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim[::-1] if self.hparams.feed_forward_hidden_dim is not None else None,
        #                              use_skip_connection=self.hparams.use_skip_connection)

        
        self.vs = VolumeSamplingBlindSweep(mount_point=self.hparams.mount_point)

        us_simulator_cut = MergedLinearLabel11().eval()
        self.us_simulator_cut_td = TimeDistributed(us_simulator_cut, time_dim=2).eval()


        if hasattr(self.hparams, 'n_fixed_samples') and self.hparams.n_fixed_samples is not None and self.hparams.n_fixed_samples > 0:
            x_v_fixed = torch.rand(1, self.hparams.n_fixed_samples, 3)
            x_v_fixed = (self.vs.simulation_fov_end - self.vs.simulation_fov_origin)*x_v_fixed + self.vs.simulation_fov_origin
            self.register_buffer('x_v_fixed', x_v_fixed)
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("USPCReconstruction")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder/Decoder parameters

        group.add_argument("--latent_channels", type=int, default=3, help='Output dimension for the image encoder')
        
        # group.add_argument("--num_samples", type=int, default=4096, help='Number of samples to take from the mesh to start the encoding')
        group.add_argument("--input_dim", type=int, default=6, help='Input dimension for the encoder')
        group.add_argument("--output_dim", type=int, default=3, help='Output dimension of the model')
        group.add_argument("--K", type=int, nargs="*", default=[27, 125], help='Number of K neighbors for each stage')
        group.add_argument("--num_heads", type=int, nargs="*", default=[64, 128], help='Number of attention heads per stage the encoder')
        group.add_argument("--stages", type=int, nargs="*", default=[64, 128], help='Dimension per stage')
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        group.add_argument("--pooling_factor", type=float, nargs="*", default=[0.5, 0.5], help='Pooling factor')
        group.add_argument("--score_pooling", type=int, default=1, help='Use score base pooling')
        group.add_argument("--pooling_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the pooling layer')
        group.add_argument("--feed_forward_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the Residual FeedForward layer')
        # group.add_argument("--use_skip_connection", type=int, default=1, help='Use skip connections, i.e., unet style network')

        group.add_argument("--num_random_sweeps", type=int, default=3, help='How many random sweeps to use')
        group.add_argument("--n_grids", type=int, default=200, help='Number of grids')
        group.add_argument("--target_label", type=int, default=7, help='Target label')

        # group.add_argument("--n_fixed_samples", type=int, default=12288, help='Number of fixed samples')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def on_fit_start(self):

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

        if not os.path.exists(grid_t_file):
            grid_tensor = []
            inverse_grid_t = []
            mask_fan_t = []

            for i in range(self.hparams.n_grids):

                grid_w, grid_h = self.hparams.grid_w, self.hparams.grid_h
                center_x = self.hparams.center_x
                r1 = self.hparams.r1

                center_y = self.hparams.center_y_start + (self.hparams.center_y_end - self.hparams.center_y_start) * (torch.rand(1))
                r2 = self.hparams.r2_start + ((self.hparams.r2_end - self.hparams.r2_start) * torch.rand(1)).item()
                theta = self.hparams.theta_start + ((self.hparams.theta_end - self.hparams.theta_start) * torch.rand(1)).item()
                
                grid, inverse_grid, mask = self.USR.init_grids(grid_w, grid_h, center_x, center_y, r1, r2, theta)

                grid_tensor.append(grid.unsqueeze(dim=0))
                inverse_grid_t.append(inverse_grid.unsqueeze(dim=0))
                mask_fan_t.append(mask.unsqueeze(dim=0))

            self.grid_t = torch.cat(grid_tensor).to(self.device)
            self.inverse_grid_t = torch.cat(inverse_grid_t).to(self.device)
            self.mask_fan_t = torch.cat(mask_fan_t).to(self.device)

            # Save tensors directly to the current directory
            
            torch.save(self.grid_t, grid_t_file)
            torch.save(self.inverse_grid_t, inverse_grid_t_file)
            torch.save(self.mask_fan_t, mask_fan_t_file)

            # print("Grids SAVED!")
            # print(self.grid_t.shape, self.inverse_grid_t.shape, self.mask_fan_t.shape)
        
        elif not hasattr(self, 'grid_t'):
            # Load tensors directly from the current directory
            self.grid_t = torch.load(grid_t_file).to(self.device)
            self.inverse_grid_t = torch.load(inverse_grid_t_file).to(self.device)
            self.mask_fan_t = torch.load(mask_fan_t_file).to(self.device)

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:        
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae
    
    def compute_loss(self, x, Y, step="train", sync_dist=False):
        
        loss_chamfer, _ = chamfer_distance(x, Y, batch_reduction="mean", point_reduction="sum")

        loss = loss_chamfer

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)        
        # self.log(f"{step}_loss_mse", loss_mse, sync_dist=sync_dist)
        # self.log(f"{step}_loss_chamfer", loss_chamfer, sync_dist=sync_dist)
        # self.log(f"{step}_loss_point_mesh_face", loss_point_mesh_face, sync_dist=sync_dist)
        # self.log(f"{step}_loss_point_mesh_edge", loss_point_mesh_edge, sync_dist=sync_dist)

        return loss
    
    def create_mesh(self, V, F):
        return Meshes(verts=V, faces=F)
    
    def get_target(self, X, X_origin, X_end):
        # put the diffusor in the fov
        diffusor_in_fov = self.vs.diffusor_in_fov(X.float(), diffusor_origin=X_origin, diffusor_end=X_end)

        V_fov = self.vs.fov_physical().reshape(-1, 3)
        V_diff = []
        
        # Get only non-background points and their corresponding labels
        for d_fov in diffusor_in_fov:

            d_fov = d_fov.reshape(-1)

            V_diff.append(V_fov[d_fov == self.hparams.target_label])
        
        # Pad them to create tensors
        V_diff = pad_sequence(V_diff, batch_first=True, padding_value=0.0)
        
        return V_diff

    def volume_sampling(self, X, X_origin, X_end, use_random=False):
        with torch.no_grad():
            self.us_simulator_cut_td.eval()

            probe_origin_rand = None
            probe_direction_rand = None
            grid = None
            inverse_grid = None
            mask_fan = None

            if use_random:
                probe_origin_rand = torch.rand(3, device=self.device)*0.0001
                probe_origin_rand = probe_origin_rand
                rotation_ranges = ((-5, 5), (-5, 5), (-10, 10))  # ranges in degrees for x, y, and z rotations
                probe_direction_rand = self.vs.random_affine_matrix(rotation_ranges).to(self.device)

                grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(1,))
            
                grid = self.grid_t[grid_idx]
                inverse_grid = self.inverse_grid_t[grid_idx]
                mask_fan = self.mask_fan_t[grid_idx]

            X_sweeps = []
            X_sweeps_tags = []
            for tag in np.random.choice(self.vs.tags, self.hparams.num_random_sweeps):
                sampled_sweep = self.vs.diffusor_sampling_tag(tag, X.to(torch.float), X_origin.to(torch.float), X_end.to(torch.float), probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)

                sampled_sweep_simu = []

                for ss in sampled_sweep:
                    ss_chunk = []
                    for ss_ch in torch.chunk(ss, chunks=20, dim=1):
                        simulated = self.us_simulator_cut_td(ss_ch.unsqueeze(dim=1), grid, inverse_grid, mask_fan)
                        # simulated = self.us_simulator_cut_td.module.transform_us(simulated)
                        ss_chunk.append(simulated)
                    sampled_sweep_simu.append(torch.cat(ss_chunk, dim=2))
                
                sampled_sweep_simu = torch.stack(sampled_sweep_simu).detach()
                X_sweeps.append(sampled_sweep_simu)
                X_sweeps_tags.append(self.vs.tags_dict[tag])
                # out_fovs = self.vs.simulated_sweep_in_fov(tag, sampled_sweep_simu.detach())
                # X_sweeps.append(out_fovs.unsqueeze(1).unsqueeze(1).detach())
            X_sweeps = torch.cat(X_sweeps, dim=1)
            X_sweeps_tags = torch.tensor(X_sweeps_tags).repeat(X_sweeps.shape[0], 1)
            
            # Y = self.us_simulator_cut_td.module.USR.mean_diffusor_dict[X.to(torch.long)].to(self.device) + torch.randn(X.shape, device=self.device) * self.us_simulator_cut_td.module.USR.variance_diffusor_dict[X.to(torch.long)].to(self.device)
            # Y = self.vs.diffusor_in_fov(Y, X_origin, X_end)

            return X_sweeps, X_sweeps_tags


    def training_step(self, train_batch, batch_idx):
        X, X_origin, X_end = train_batch

        Y = self.get_target(X, X_origin, X_end)

        x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end, use_random=True)

        # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
        # tags shape torch.Size([2, 2])

        Nsweeps = x_sweeps.shape[1]
        
        x = []
        x_v = []

        for n in range(Nsweeps):
            x_sweeps_n = x_sweeps[:, n, :, :, :, :]
            sweeps_tags_n = sweeps_tags[:, n]

            x_sweeps_n, x_sweeps_n_v = self.encode(x_sweeps_n, sweeps_tags_n)
            
            x.append(x_sweeps_n)
            x_v.append(x_sweeps_n_v)

        x = torch.cat(x, dim=1)
        x_v = torch.cat(x_v, dim=1)

        x_hat = self.encode_v(x, x_v)

        Y_v = self.get_target(X, X_origin, X_end)

        return self.compute_loss(x_hat, Y_v, step="train")
        

    def validation_step(self, val_batch, batch_idx):
        
        X, X_origin, X_end = val_batch

        x_sweeps, sweeps_tags = self.volume_sampling(X, X_origin, X_end)

        # x_sweeps shape is B, N, C, T, H, W. N for number of sweeps ex. torch.Size([2, 2, 1, 200, 256, 256]) 
        # tags shape torch.Size([2, 2])

        Nsweeps = x_sweeps.shape[1]
        
        x = []
        x_v = []

        for n in range(Nsweeps):
            x_sweeps_n = x_sweeps[:, n, :, :, :, :]
            sweeps_tags_n = sweeps_tags[:, n]

            x_sweeps_n, x_sweeps_n_v = self.encode(x_sweeps_n, sweeps_tags_n)
            
            x.append(x_sweeps_n)
            x_v.append(x_sweeps_n_v)

        x = torch.cat(x, dim=1)
        x_v = torch.cat(x_v, dim=1)

        x_hat = self.encode_v(x, x_v)

        Y_v = self.get_target(X, X_origin, X_end)

        return self.compute_loss(x_hat, Y_v, step="val", sync_dist=True)

    def encode(self, x, sweeps_tags):

        x = self.encoder_i(x)
        
        x_e = self.attn_chunk(x)
        
        x = x_e.permute(0, 1, 4, 3, 2)
        
        x_v = self.vs.diffusor_tag_resample(x, tag=sweeps_tags)
        x = x.flatten(2).permute(0, 2, 1)

        x = torch.cat([x, x_v], dim=-1)

        return x, x_v

    def encode_v(self, x, x_v):
        
        x_v_fixed = None

        if hasattr(self, 'x_v_fixed'):
            x_v_fixed = self.x_v_fixed
            batch_size = x_v.shape[0]
            x_v_fixed = x_v_fixed.repeat(batch_size, 1, 1)


        x, x_v, unpooling_idx = self.encoder_v(x, x_v, x_v_fixed=x_v_fixed)

        x = self.fc(x)

        # z_mu = self.ff_mu(h)
        # z_sigma = self.ff_sigma(h)
        # z = self.sampling(z_mu, z_sigma)

        return x

    def forward(self, x, sweeps_tags):

        x, x_v = self.encode(x, sweeps_tags)

        x = self.encode_v(x, x_v)

        # z_mu = self.ff_mu(h)
        # z_sigma = self.ff_sigma(h)
        # z = self.sampling(z_mu, z_sigma)

        return x


class USPCSegmentation(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        
        self.encoder = MHAIdxEncoder(input_dim=self.hparams.input_dim, 
                                     output_dim=self.hparams.stages[-1], 
                                     K=self.hparams.K, 
                                     num_heads=self.hparams.num_heads, 
                                     stages=self.hparams.stages, 
                                     dropout=self.hparams.dropout, 
                                     pooling_factor=self.hparams.pooling_factor, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim,
                                     score_pooling=self.hparams.score_pooling,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim, 
                                     use_skip_connection=self.hparams.use_skip_connection)
        
        self.decoder = MHAIdxDecoder(input_dim=self.hparams.stages[-1], 
                                     output_dim=self.hparams.output_dim, 
                                     K=self.hparams.K[::-1], 
                                     num_heads=self.hparams.num_heads[::-1], 
                                     stages=self.hparams.stages[::-1], 
                                     dropout=self.hparams.dropout, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim[::-1] if self.hparams.pooling_hidden_dim is not None else None,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim[::-1] if self.hparams.feed_forward_hidden_dim is not None else None,
                                     use_skip_connection=self.hparams.use_skip_connection)
                                  
        
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        # self.loss_fn = nn.MSELoss()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("USPCSegmentation")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # ENCODER params
        group.add_argument("--input_dim", type=int, default=1, help='Input dimension for the encoder')
        group.add_argument("--output_dim", type=int, default=1, help='Output dimension of the model')
        group.add_argument("--K", type=int, nargs="*", default=[27, 27, 27, 27], help='Number of K neighbors for each stage')
        group.add_argument("--num_heads", type=int, nargs="*", default=[2, 4, 8, 16], help='Number of attention heads per stage the encoder')
        group.add_argument("--stages", type=int, nargs="*", default=[8, 16, 32, 64], help='Dimension per stage')
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        group.add_argument("--pooling_factor", type=float, nargs="*", default=[0.125, 0.5, 0.5, 0.5], help='Pooling factor')
        group.add_argument("--score_pooling", type=int, default=1, help='Use score base pooling')
        group.add_argument("--pooling_hidden_dim", type=int, nargs="*", default=[4, 8, 16, 32], help='Hidden dim for the pooling layer')
        group.add_argument("--feed_forward_hidden_dim", type=int, nargs="*", default=None, help='Hidden dim for the Residual FeedForward layer')
        group.add_argument("--use_skip_connection", type=int, default=0, help='Use skip connections, i.e., unet style network')
        
        group.add_argument('--threshold', help='Threshold value for the simulated US, use to filter background level and reduce the number of points', type=float, default=0.0)       
        group.add_argument('--use_v', help='Use the coordinates as well for the encoder', type=int, default=0)

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def compute_loss(self, x, Y, step="train", sync_dist=False):

        x = x.permute(0, 2, 1)
        Y = Y.permute(0, 2, 1)

        loss = self.loss_fn(x, Y)
        
        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss

    def training_step(self, train_batch, batch_idx):
        X_d = train_batch
        X_v, X_f = X_d["img"]
        Y_v, Y = X_d["seg"]
        
        x = self(X_v, X_f)

        return self.compute_loss(x, Y, step="train")
        

    def validation_step(self, val_batch, batch_idx):
        X_d = val_batch
        X_v, X_f = X_d["img"]
        Y_v, Y = X_d["seg"]

        x = self(X_v, X_f)

        return self.compute_loss(x, Y, step="val", sync_dist=True)

    def forward(self, x_v, x_f):
        # V, VF = self.get_grid_VF(X)
        skip_connections = None
        if self.hparams.use_skip_connection:
            x, unpooling_idxs, skip_connections = self.encoder(x_f, x_v)
        else:
            x, unpooling_idxs = self.encoder(x_f, x_v)
        return self.decoder(x, unpooling_idxs, skip_connections)


class USGAPC(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_i = TimeDistributed(Encoder(spatial_dims=2,
            in_channels=1,
            num_channels=[8, 16, 32, 64, 128],
            out_channels=self.hparams.latent_channels,
            num_res_blocks=[2, 2, 2, 2, 2],
            norm_num_groups=8,
            norm_eps=1e-3,
            attention_levels=[False, False, False, False, True],
            with_nonlocal_attn=True), time_dim=2)
        
        # encoder_i = tv_models.efficientnet_v2_s().features
        # encoder_i[0] = Conv2dNormActivation(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.SiLU)
        # self.encoder_i = TimeDistributed(encoder_i, time_dim=2)

        self.attn_chunk = AttentionChunk(input_dim=(self.hparams.latent_channels*16*16), hidden_dim=64, chunks=16)

        self.encoder_v = MHAIdxEncoder(input_dim=self.hparams.input_dim, 
                                     output_dim=self.hparams.stages[-1], 
                                     K=self.hparams.K, 
                                     num_heads=self.hparams.num_heads, 
                                     stages=self.hparams.stages, 
                                     dropout=self.hparams.dropout, 
                                     pooling_factor=self.hparams.pooling_factor, 
                                     pooling_hidden_dim=self.hparams.pooling_hidden_dim,
                                     score_pooling=self.hparams.score_pooling,
                                     feed_forward_hidden_dim=self.hparams.feed_forward_hidden_dim, 
                                     use_skip_connection=0, 
                                     return_v=True)

        self.mha = Residual(nn.MultiheadAttention(embed_dim=self.hparams.stages[-1], num_heads=self.hparams.stages[-1], dropout=self.hparams.dropout))

        if self.hparams.use_layer_norm:
            self.norm = nn.LayerNorm(self.hparams.stages[-1])
        
        self.attn = SelfAttention(self.hparams.stages[-1], self.hparams.pooling_hidden_dim[-1])
        self.fc = nn.Linear(self.hparams.stages[-1], self.hparams.output_dim)
        
        self.vs = VolumeSamplingBlindSweep(mount_point=self.hparams.mount_point)

        self.loss_fn = nn.L1Loss()
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("USGAPC")

        group.add_argument("--lr", type=float, default=1e-4)
        group.add_argument('--weight_decay', help='Weight decay for optimizer', type=float, default=0.01)
        
        # Encoder/Decoder parameters

        group.add_argument("--latent_channels", type=int, default=3, help='Output dimension for the image encoder')
        
        group.add_argument("--input_dim", type=int, default=6, help='Input dimension for the encoder')
        group.add_argument("--output_dim", type=int, default=1, help='Output dimension of the model')
        group.add_argument("--K", type=int, nargs="*", default=[125, 125], help='Number of K neighbors for each stage')
        group.add_argument("--num_heads", type=int, nargs="*", default=[64, 128], help='Number of attention heads per stage the encoder')
        group.add_argument("--stages", type=int, nargs="*", default=[64, 128], help='Dimension per stage')
        group.add_argument("--dropout", type=float, default=0.1, help='Dropout rate')
        group.add_argument("--pooling_factor", type=float, nargs="*", default=[0.25, 0.25], help='Pooling factor')
        group.add_argument("--score_pooling", type=int, default=1, help='Use score base pooling')
        group.add_argument("--pooling_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the pooling layer')
        group.add_argument("--feed_forward_hidden_dim", type=int, nargs="*", default=[32, 64], help='Hidden dim for the Residual FeedForward layer')
        group.add_argument("--use_layer_norm", type=int, default=1, help='Use layer norm')

        group.add_argument("--max_sweeps", type=int, default=3, help='Max number of sweeps to use')

        return parent_parser
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        return optimizer

    def sampling(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:        
        eps = torch.randn_like(z_sigma)
        z_vae = z_mu + eps * z_sigma
        return z_vae
    
    def compute_loss(self, x, y, step="train", sync_dist=False):
        
        loss = self.loss_fn(x, y)

        self.log(f"{step}_loss", loss, sync_dist=sync_dist)

        return loss
    
    
    def training_step(self, train_batch, batch_idx):
        x_d = train_batch

        x = []
        x_v = []

        keys = x_d.keys()

        for k in keys:
            if isinstance(k, int):
                x_img = x_d[k]
                tags = x_d["tag"][:,k]

                x_, x_v_, _ = self.encode(x_img, tags)

                x.append(x_)
                x_v.append(x_v_)

        x = torch.cat(x, dim=1)
        x_v = torch.cat(x_v, dim=1)
        
        x, x_s = self.attn(x, x)
        x = self.fc(x)

        y = x_d["ga_boe"]

        return self.compute_loss(x, y, step="train")
        

    def validation_step(self, val_batch, batch_idx):
        
        x_d = val_batch
        
        x = []
        x_v = []

        keys = x_d.keys()

        for k in keys:
            if isinstance(k, int):
                x_img = x_d[k]
                tags = x_d["tag"][:,k]

                x_, x_v_, _ = self.encode(x_img, tags)

                x.append(x_)
                x_v.append(x_v_)

        x = torch.cat(x, dim=1)
        x_v = torch.cat(x_v, dim=1)
        
        x, x_s = self.attn(x, x)
        x = self.fc(x)

        y = x_d["ga_boe"]

        return self.compute_loss(x, y, step="val", sync_dist=True)
    
    def encode(self, x, sweeps_tags):

        
        x = self.encoder_i(x)
        
        x_e = self.attn_chunk(x)
        
        x = x_e.permute(0, 1, 4, 3, 2)
        
        x_v = self.vs.diffusor_tag_resample(x, tag=sweeps_tags)
        x = x.flatten(2).permute(0, 2, 1)

        x = torch.cat([x, x_v], dim=-1)

        x, x_v, unpooling_idx = self.encoder_v(x, x_v)

        x, x_w = self.mha(x, x, x)
        
        return x, x_v, x_e
    
    def forward(self, x, sweeps_tags):
        
        x, x_v, _ = self.encode(x, sweeps_tags)

        x, x_s = self.attn(x, x)
        x = self.fc(x)

        return x, x_v