import torch
import numpy as np

import pytorch_lightning as pl
from torchir.networks import DIRNet
from torchir.metrics import NCC
from torchir.transformers import BsplineTransformer

from .us_simulation_jit import MergedLinearCut1
from .us_simu import VolumeSamplingBlindSweep
from .layers import TimeDistributed, MultiHeadAttention3D

class LitDIRNet(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.dirnet = DIRNet(grid_spacing=self.hparams.grid_spacing,
                kernel_size=self.hparams.kernel_size,
                kernels=self.hparams.kernels,
                num_conv_layers=self.hparams.num_conv_layers,
                num_dense_layers=self.hparams.num_dense_layers,
                ndim=self.hparams.ndim,
            )
        self.bspline_transformer = BsplineTransformer(ndim=self.hparams.ndim, upsampling_factors=self.hparams.upsampling_factors)
        self.metric = NCC()

        us_simulator_cut = MergedLinearCut1().eval()
        self.us_simulator_cut_td = TimeDistributed(us_simulator_cut, time_dim=2).eval()
        self.vs = VolumeSamplingBlindSweep(mount_point=self.hparams.mount_point).eval()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitDIRNet")
        parser.add_argument("--grid_spacing", type=int, nargs="+", default=(8, 8))
        parser.add_argument("--kernel_size", type=int, default=3)
        parser.add_argument("--kernels", type=int, default=32)
        parser.add_argument("--num_conv_layers", type=int, default=5)
        parser.add_argument("--num_dense_layers", type=int, default=2)
        parser.add_argument("--ndim", type=int, default=2)
        parser.add_argument("--lr", type=float, default=1e-4)
        return parent_parser
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.dirnet.parameters(), lr=self.hparams.lr, amsgrad=True)
        return optimizer

    def forward(self, fixed, moving):
        params = self.dirnet(fixed, moving)
        warped = self.bspline_transformer(params, fixed, moving)
        return warped
    
    def training_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/training', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        warped = self(batch['fixed'], batch['moving'])
        loss = self.metric(batch['fixed'], warped)
        self.log('NCC/validation', loss)
        return loss  
    
    def volume_sampling(self, X, X_origin, X_end, grid=None, inverse_grid=None, mask_fan=None, use_random=False):
        self.us_simulator_cut_td.eval()

        probe_origin_rand = None
        probe_direction_rand = None
        if use_random:
            probe_origin_rand = torch.rand(3, device=self.device)*0.001
            probe_origin_rand = probe_origin_rand
            rotation_ranges = ((-15, 5), (-15, 15), (-30, 30))  # ranges in degrees for x, y, and z rotations
            probe_direction_rand = self.vs.random_affine_matrix(rotation_ranges).to(self.device)
        
        for tag in np.random.choice(self.vs.tags, 1):
            sampled_sweep = self.vs.diffusor_sampling_tag(tag, X.to(torch.float), X_origin.to(torch.float), X_end.to(torch.float), probe_origin_rand=probe_origin_rand, probe_direction_rand=probe_direction_rand, use_random=use_random)
            sampled_sweep_simu = torch.cat([self.us_simulator_cut_td(ss.unsqueeze(dim=0), grid, inverse_grid, mask_fan) for ss in sampled_sweep], dim=0)

        return sampled_sweep_simu