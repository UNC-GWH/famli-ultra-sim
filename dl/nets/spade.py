import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch import nn
import monai
from monai.networks import nets as MNets
from monai.networks import blocks as MBlocks
import sys
import pytorch_lightning as pl

from generative.networks.nets import PatchDiscriminator, MultiScalePatchDiscriminator

from nets.lotus import UltrasoundRendering, UltrasoundRenderingLinear, UltrasoundRenderingLinearV2, UltrasoundRenderingConv1d
import numpy as np 
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from torchvision import transforms as T

from torch.nn.modules.loss import _Loss
from monai.utils import LossReduction

from generative.networks.nets import SPADEAutoencoderKL, AutoencoderKL
from generative.networks.nets import SPADEDiffusionModelUNet

from torch.cuda.amp import GradScaler, autocast

class NCELoss(_Loss):
    """
    Calculates the NCELoss

    Args:
        reduction: {``"none"``, ``"mean"``, ``"sum"``} Specifies the reduction to apply to the output.
        Defaults to ``"none"``.
        - ``"none"``: no reduction will be applied.
        - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
        - ``"sum"``: the output will be summed.
    """

    def __init__(
        self,
        reduction: LossReduction,
    ) -> None:
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self, features_source: list, features_target: list
    ) -> torch.Tensor:
        """

        Args:
            source: output of a projection head
            target: output of a projection head
        """

        total_nce_loss = 0.0
        for feat_s, feat_t in zip(features_source, features_target):
            loss = self.patch_nce_loss(feat_s, feat_t)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5.0    

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
        loss = self.criterion(out, torch.arange(0, out.size(0), dtype=torch.long).to(feat_q.device))
        return loss

class ProjectionHeads(nn.Module):
    def __init__(self, blocks, blocks_ids=[0, 4, 7, 10, 14, 17], in_shape=(1, 1, 256, 256)):
        super().__init__()
        
        self.blocks_ids = blocks_ids
        
        x = torch.rand(in_shape)
        
        for i, layer in enumerate(blocks):
            x = layer(x)
            if i in blocks_ids:

                mlp = nn.Sequential(*[
                    nn.Linear(x.shape[1], 256),
                    nn.ReLU(),
                    nn.Linear(256, 256)
                ])
                
                # print("ProjectionHeads {i}, {shape}".format(i=i, shape=x.shape))

                setattr(self, 'mlp_%d' % i, mlp)
                
        
    def forward(self, feats):
        
        return_feats = []
        
        for feat_id, feat in zip(self.blocks_ids, feats):
            mlp = getattr(self, 'mlp_%d' % feat_id)
            feat = mlp(feat)
            norm = feat.pow(2).sum(1, keepdim=True).pow(1. / 2)
            feat = feat.div(norm + 1e-7)
            return_feats.append(feat)
        return return_feats


class SPADELotus(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        # self.D = MNets.Discriminator(in_shape=(1, 128, 128))
        self.D = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=16, in_channels=1, out_channels=1)
        

        # self.USR = UltrasoundRendering(**kwargs)
        self.USR = UltrasoundRenderingLinear(**kwargs)
        # self.USR = UltrasoundRenderingLinearV2(**kwargs)
        # self.USR = UltrasoundRenderingConv1d(**kwargs)
        

        self.G = SPADEAutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=(2, 2, 2, 2),
                num_channels=(8, 16, 32, 64),
                attention_levels=[False, False, False, False],
                latent_channels=8,
                norm_num_groups=8,
                label_nc=self.hparams.num_labels,
            )
        
        self.H = ProjectionHeads(self.G.encoder.blocks)

        # self.PH = nn.Sequential(
        #         nn.Linear(8*32*32, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, 256)
        #     )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.nce_loss = NCELoss(reduction='none')
        
        self.l1 = nn.L1Loss()
        # self.mse = nn.MSELoss()
        # self.cos_sim = nn.CosineSimilarity(dim=1)

        self.automatic_optimization = False
        
        height = 256
        if hasattr(self.hparams, 'height'):
            height = self.hparams.height
        self.transform_us = T.Compose([T.Pad((0, int(40*height/128), 0, 0)), T.CenterCrop(height)])
        
        # self.scaler_g = GradScaler()
        # self.scaler_d = GradScaler()

    def configure_optimizers(self):
        opt_gen = optim.AdamW(
            list(self.USR.parameters()) + list(self.G.parameters()),
            # self.USR.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay            
        )
        opt_disc = optim.AdamW(
            self.D.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )        
        opt_head = optim.AdamW(
            # list(self.H.parameters()) + list(self.PH.parameters()),
            self.H.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )

        return [opt_gen, opt_disc, opt_head]
    
    # This is only called during inference time to set a custom grid
    def init_grid(self, w, h, center_x, center_y, r1, r2, theta):
        grid = self.USR.compute_grid(w, h, center_x, center_y, r1, r2, theta)
        inverse_grid, mask = self.USR.compute_grid_inverse(grid)
        
        self.USR.grid = self.USR.normalize_grid(grid)
        self.USR.inverse_grid = self.USR.normalize_grid(inverse_grid)
        self.USR.mask_fan = mask

    def forward(self, X, grid = None, inverse_grid = None, mask_fan = None):
        if mask_fan is None:
            mask_fan = self.USR.mask_fan
        
        S = self.one_hot(self.transform_us(X))
        X_ = self.transform_us(self.USR(X, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan))        
        X, z_mu, z_sigma = self.G(X_, S)        
        return X*self.transform_us(mask_fan), X_, z_mu, z_sigma

    # def scheduler_step(self):
    #     self.scheduler_disc.step()
    #     self.scheduler_gen.step()
    #     self.scheduler_mlp.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def on_fit_start(self):

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t_{h}.pt'.format(h=self.hparams.grid_w)
        inverse_grid_t_file = 'inverse_grid_t_{h}.pt'.format(h=self.hparams.grid_w)
        mask_fan_t_file = 'mask_fan_t_{h}.pt'.format(h=self.hparams.grid_w)

        if self.hparams.create_grids or not os.path.exists(grid_t_file):
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

    def training_step(self, train_batch, batch_idx):

        # Y is the real ultrasound
        labeled, Y = train_batch
        X_x = labeled['img']
        X_s = labeled['seg']

        opt_gen, opt_disc, opt_head = self.optimizers()

        grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(X_s.shape[0],))
        
        grid = self.grid_t[grid_idx]
        inverse_grid = self.inverse_grid_t[grid_idx]
        mask_fan = self.mask_fan_t[grid_idx]

        Y_fake, X_, z_mu, z_sigma = self(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)

        # update D
        loss_d = 0.0
        if self.current_epoch >= self.hparams.warm_up_epochs:            
            self.set_requires_grad(self.D, True)
            opt_disc.zero_grad()
            loss_d = self.compute_D_loss(Y, Y_fake)
            self.manual_backward(loss_d)            
            opt_disc.step()
        

        self.set_requires_grad(self.D, False)
        opt_gen.zero_grad()
        opt_head.zero_grad()
        loss_g = self.compute_G_loss(Y, Y_fake, X_, z_mu, z_sigma)
        self.manual_backward(loss_g)
        opt_gen.step()
        opt_head.step()

        # update G
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

    def validation_step(self, val_batch, batch_idx):

        labeled, Y = val_batch
        X_x = labeled['img']
        X_s = labeled['seg']
        
        grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(X_s.shape[0],))
        
        grid = self.grid_t[grid_idx]
        inverse_grid = self.inverse_grid_t[grid_idx]
        mask_fan = self.mask_fan_t[grid_idx]

        Y_fake, X_, z_mu, z_sigma = self(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
        loss_g = self.compute_G_loss(Y, Y_fake, X_, z_mu, z_sigma)

        self.log("val_loss", loss_g, sync_dist=True)
    
    def compute_G_loss(self, Y, Y_fake, X_, z_mu, z_sigma):

        recons_loss = self.l1(Y_fake, X_.detach())
        p_loss = self.perceptual_loss(Y_fake, Y.float())
        
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        a_loss = 0.0
        if self.current_epoch >= self.hparams.warm_up_epochs:
            logits_fake = self.D(Y_fake.contiguous().float())[-1]
            a_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)        
        
        nce_loss = 0.0
        if self.hparams.nce_weight > 0:
            nce_loss = self.calculate_NCE_loss(Y_fake, Y)
        
        sim_loss = 0.0
        # if self.hparams.sim_weight > 0:
        #     sim_loss = self.calculate_sim_loss(Y_fake, Y)        

        loss_g = self.hparams.recons_weight*recons_loss + self.hparams.nce_weight*nce_loss + self.hparams.adversarial_weight * a_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss) + self.hparams.sim_weight*sim_loss

        return loss_g

    def compute_D_loss(self, Y, Y_fake):
        logits_fake = self.D(Y_fake.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.D(Y.contiguous())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = self.hparams.adversarial_weight * discriminator_loss
        
        return loss_d
    
    def calculate_sim_loss(self, X, Y):
        
        batch_size = X.shape[0]
        feat_f = self.G.encoder(X).flatten(1)
        feat_r = self.G.encoder(Y).flatten(1)

        feat_f = self.PH(feat_f)
        feat_r = self.PH(feat_r)

        loss_proj_c = self.cos_sim(feat_f, feat_r) # the cosine similarity between the features, the higher the better

        loss_proj_c_sorted_i = torch.argsort(loss_proj_c) # this gives the indices sorted in ascending order, i.e., the most dissimilar ones first
        loss_proj_c = 1.0 - loss_proj_c[loss_proj_c_sorted_i] # we want the most dissimilar ones to have the highest loss
        
        w = torch.pow(torch.arange(batch_size, device=self.device)/batch_size + 0.01, 2) #With the weighting scheme we want to reduce the importance of the most dissimilar ones and increase the importance of the most similar ones

        loss_proj_c = torch.sum(w*loss_proj_c)

        return loss_proj_c
    
    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.encode_nce(tgt, self.G.encoder.blocks)
        feat_k, _, = self.encode_nce(src, self.G.encoder.blocks, patch_ids=patch_ids_q)
        
        feat_q_pool = self.H(feat_q)
        feat_k_pool = self.H(feat_k)

        return self.nce_loss(feat_q_pool, feat_k_pool)
    
    def encode_nce(self, x, blocks, patch_ids=None, blocks_ids=[0, 4, 7, 10, 14, 17], num_patches = 256):
        feat = x
        return_ids = []
        return_feats = []
        p_id = 0

        for block_id, block in enumerate(blocks):
            feat = block(feat)
            if block_id in blocks_ids:
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
                if patch_ids is not None:
                    patch_id = patch_ids[p_id]
                    p_id += 1
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1]) 
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                    return_ids.append(patch_id)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)

                return_feats.append(x_sample)

        return return_feats, return_ids
    
    def one_hot(self, input_label):
        # One hot encoding function for the labels
        shape_ = list(input_label.shape)
        shape_[1] = self.hparams.num_labels
        label_out = torch.zeros(shape_).to(self.device)
        for channel in range(self.hparams.num_labels):
            label_out[:, channel, ...] = input_label[:, 0, ...] == channel
        return label_out


class SPADELotusMulti(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.D = MultiScalePatchDiscriminator(
            num_d=2,
            num_layers_d=3,
            spatial_dims=2,
            num_channels=32,
            in_channels=1,
            out_channels=1,
            minimum_size_im=128,
            norm="INSTANCE",
            kernel_size=3)
        

        # self.USR = UltrasoundRendering(**kwargs)
        self.USR = UltrasoundRenderingLinear(**kwargs)
        # self.USR = UltrasoundRenderingLinearV2(**kwargs)
        # self.USR = UltrasoundRenderingConv1d(**kwargs)
        

        self.G = SPADEAutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=(2, 2, 2, 2),
                num_channels=(8, 16, 32, 64),
                attention_levels=[False, False, False, False],
                latent_channels=8,
                norm_num_groups=8,
                label_nc=self.hparams.num_labels,
            )
        
        # self.H = ProjectionHeads(self.G.encoder.blocks)
        self.H = nn.Sequential(
                nn.Linear(8*64*64, 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.nce_loss = NCELoss(reduction='none')
        
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.l1 = nn.L1Loss()
        # self.mse = nn.MSELoss()

        self.automatic_optimization = False
        
        self.transform_us = T.Compose([T.Pad((0, int(40*self.hparams.height/128), 0, 0)), T.CenterCrop(self.hparams.height)])
        
        # self.scaler_g = GradScaler()
        # self.scaler_d = GradScaler()

    def configure_optimizers(self):
        opt_gen = optim.AdamW(
            list(self.USR.parameters()) + list(self.G.parameters()),
            # self.USR.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay            
        )
        opt_disc = optim.AdamW(
            self.D.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )        
        opt_head = optim.AdamW(
            self.H.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )

        return [opt_gen, opt_disc, opt_head]
    
    # This is only called during inference time to set a custom grid
    def init_grid(self, w, h, center_x, center_y, r1, r2, theta):
        grid = self.USR.compute_grid(w, h, center_x, center_y, r1, r2, theta)
        inverse_grid, mask = self.USR.compute_grid_inverse(grid)
        
        self.USR.grid = self.USR.normalize_grid(grid)
        self.USR.inverse_grid = self.USR.normalize_grid(inverse_grid)
        self.USR.mask_fan = mask

    def forward(self, X, grid = None, inverse_grid = None, mask_fan = None):
        if mask_fan is None:
            mask_fan = self.USR.mask_fan
        
        S = self.one_hot(self.transform_us(X))
        X_ = self.transform_us(self.USR(X, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan))        
        X, z_mu, z_sigma = self.G(X_, S)        
        return X*self.transform_us(mask_fan), X_, z_mu, z_sigma

    # def scheduler_step(self):
    #     self.scheduler_disc.step()
    #     self.scheduler_gen.step()
    #     self.scheduler_mlp.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def on_fit_start(self):

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t_{h}.pt'.format(h=self.hparams.grid_w)
        inverse_grid_t_file = 'inverse_grid_t_{h}.pt'.format(h=self.hparams.grid_w)
        mask_fan_t_file = 'mask_fan_t_{h}.pt'.format(h=self.hparams.grid_w)

        if self.hparams.create_grids or not os.path.exists(grid_t_file):
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

    def training_step(self, train_batch, batch_idx):

        # Y is the real ultrasound
        labeled, Y = train_batch
        X_x = labeled['img']
        X_s = labeled['seg']

        opt_gen, opt_disc, opt_head = self.optimizers()

        grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(X_s.shape[0],))
        
        grid = self.grid_t[grid_idx]
        inverse_grid = self.inverse_grid_t[grid_idx]
        mask_fan = self.mask_fan_t[grid_idx]

        Y_fake, X_, z_mu, z_sigma = self(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)

        # update D
        loss_d = 0.0
        if self.current_epoch >= self.hparams.warm_up_epochs:            
            self.set_requires_grad(self.D, True)
            opt_disc.zero_grad()
            loss_d = self.compute_D_loss(Y, Y_fake)
            self.manual_backward(loss_d)            
            opt_disc.step()            
            self.untoggle_optimizer(opt_disc)
        self.set_requires_grad(self.D, False)

        opt_gen.zero_grad()
        opt_head.zero_grad()
        loss_g = self.compute_G_loss(Y, Y_fake, X_, z_mu, z_sigma)
        self.manual_backward(loss_g)
        opt_gen.step()
        opt_head.step()

        # update G
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

    def validation_step(self, val_batch, batch_idx):

        labeled, Y = val_batch
        X_x = labeled['img']
        X_s = labeled['seg']
        
        grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(X_s.shape[0],))
        
        grid = self.grid_t[grid_idx]
        inverse_grid = self.inverse_grid_t[grid_idx]
        mask_fan = self.mask_fan_t[grid_idx]

        Y_fake, X_, z_mu, z_sigma = self(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
        loss_g = self.compute_G_loss(Y, Y_fake, X_, z_mu, z_sigma)

        self.log("val_loss", loss_g, sync_dist=True)
    
    def compute_G_loss(self, Y, Y_fake, X_, z_mu, z_sigma):

        recons_loss = self.l1(Y_fake, X_.detach())
        p_loss = self.perceptual_loss(Y_fake, Y.float())
        
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        a_loss = 0.0
        if self.current_epoch >= self.hparams.warm_up_epochs:
            
            logits_fake, features_fakes = self.D(Y_fake.contiguous())
            a_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)        

            _, features_reals = self.D(Y.contiguous())
            loss_feat = self.feature_loss(features_fakes, features_reals)*self.hparams.feature_weight
            a_loss = a_loss + loss_feat
        
        sim_loss = self.calculate_sim_loss(Y_fake.detach(), Y)

        loss_g = self.hparams.recons_weight*recons_loss + self.hparams.sim_weight*sim_loss + self.hparams.adversarial_weight * a_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        return loss_g

    def compute_D_loss(self, Y, Y_fake):
        logits_fake, _ = self.D(Y_fake.contiguous().detach())
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real, _ = self.D(Y.contiguous())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = self.hparams.adversarial_weight * discriminator_loss
        
        return loss_d
    
    def feature_loss(self, input_features_disc_fake, input_features_disc_real):
        num_D = len(input_features_disc_fake)
        GAN_Feat_loss = torch.zeros(1).to(self.device)
        for i in range(num_D):  # for each discriminator
            num_intermediate_outputs = len(input_features_disc_fake[i])
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.l1(input_features_disc_fake[i][j], input_features_disc_real[i][j].detach())
                GAN_Feat_loss += unweighted_loss / num_D
        return GAN_Feat_loss
    
    def calculate_sim_loss(self, X, Y):
        
        batch_size = X.shape[0]
        feat_f = self.G.encoder(X).flatten(1)
        feat_r = self.G.encoder(Y).flatten(1)

        loss_proj_c = self.cos_sim(feat_f, feat_r) # the cosine similarity between the features, the higher the better

        loss_proj_c_sorted_i = torch.argsort(loss_proj_c) # this gives the indices sorted in ascending order, i.e., the most dissimilar ones first
        loss_proj_c = 1.0 - loss_proj_c[loss_proj_c_sorted_i] # we want the most dissimilar ones to have the highest loss
        
        w = torch.pow(torch.arange(batch_size, device=self.device)/batch_size + 0.01, 2) #With the weighting scheme we want to reduce the importance of the most dissimilar ones and increase the importance of the most similar ones

        loss_proj_c = torch.sum(w*loss_proj_c)

        return loss_proj_c

    
    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.encode_nce(tgt, self.G.encoder.blocks)
        feat_k, _, = self.encode_nce(src, self.G.encoder.blocks, patch_ids=patch_ids_q)
        
        feat_q_pool = self.H(feat_q)
        feat_k_pool = self.H(feat_k)

        return self.nce_loss(feat_q_pool, feat_k_pool)
    
    def encode_nce(self, x, blocks, patch_ids=None, blocks_ids=[0, 4, 7, 10, 14, 17], num_patches = 256):
        feat = x
        return_ids = []
        return_feats = []
        p_id = 0

        for block_id, block in enumerate(blocks):
            feat = block(feat)
            if block_id in blocks_ids:
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
                if patch_ids is not None:
                    patch_id = patch_ids[p_id]
                    p_id += 1
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1]) 
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                    return_ids.append(patch_id)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)

                return_feats.append(x_sample)

        return return_feats, return_ids
    
    def one_hot(self, input_label):
        # One hot encoding function for the labels
        shape_ = list(input_label.shape)
        shape_[1] = self.hparams.num_labels
        label_out = torch.zeros(shape_).to(self.device)
        for channel in range(self.hparams.num_labels):
            label_out[:, channel, ...] = input_label[:, 0, ...] == channel
        return label_out


class SPADELotusGAN(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        # self.D = MNets.Discriminator(in_shape=(1, 128, 128))
        self.D = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=16, in_channels=1, out_channels=1)

        # self.USR = UltrasoundRendering(**kwargs)
        self.USR = UltrasoundRenderingLinear(**kwargs)

        self.G = SPADEAutoencoderKL(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                num_res_blocks=(2, 2, 2, 2),
                num_channels=(8, 16, 32, 64),
                attention_levels=[False, False, False, False],
                latent_channels=8,
                norm_num_groups=8,
                label_nc=self.hparams.num_labels,
            )
        
        self.gen = MNets.Generator(latent_shape=(128,), start_shape=(1, 4, 4), channels=(64, 32, 8), strides=(2, 2, 1))
        
        self.H = ProjectionHeads(self.G.encoder.blocks)

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.nce_loss = NCELoss(reduction='none')
        
        self.l1 = nn.L1Loss()
        # self.mse = nn.MSELoss()

        self.automatic_optimization = False
        
        self.transform_us = T.Compose([T.Pad((0, int(40*self.hparams.height/128), 0, 0)), T.CenterCrop(self.hparams.height)])
        
        # self.scaler_g = GradScaler()
        # self.scaler_d = GradScaler()

    def configure_optimizers(self):
        opt_gen = optim.AdamW(
            list(self.USR.parameters()) + list(self.G.parameters()) + list(self.gen.parameters()),
            # self.USR.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay            
        )
        opt_disc = optim.AdamW(
            self.D.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )        
        opt_head = optim.AdamW(
            self.H.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )

        return [opt_gen, opt_disc, opt_head]
    
    # This is only called during inference time to set a custom grid
    def init_grid(self, w, h, center_x, center_y, r1, r2, theta):
        grid = self.USR.compute_grid(w, h, center_x, center_y, r1, r2, theta)
        inverse_grid, mask = self.USR.compute_grid_inverse(grid)
        
        self.USR.grid = self.USR.normalize_grid(grid)
        self.USR.inverse_grid = self.USR.normalize_grid(inverse_grid)
        self.USR.mask_fan = mask

    def forward(self, X):
        M = self.transform_us(self.USR.mask_fan)
        S = self.one_hot(self.transform_us(X)*M)
        X = self.transform_us(self.USR(X))
            
        h = self.encode(X)
        X, _, _ = self.decode(h, S)
        
        return X*M

    # def scheduler_step(self):
    #     self.scheduler_disc.step()
    #     self.scheduler_gen.step()
    #     self.scheduler_mlp.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def on_fit_start(self):

        # Define the file names directly without using out_dir

        grid_t_file = 'grid_t_{h}.pt'.format(h=self.hparams.grid_w)
        inverse_grid_t_file = 'inverse_grid_t_{h}.pt'.format(h=self.hparams.grid_w)
        mask_fan_t_file = 'mask_fan_t_{h}.pt'.format(h=self.hparams.grid_w)

        if self.hparams.create_grids or not os.path.exists(grid_t_file):
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

    def training_step(self, train_batch, batch_idx):

        # Y is the real ultrasound
        labeled, Y = train_batch
        X_x = labeled['img']
        X_s = labeled['seg']

        opt_gen, opt_disc, opt_head = self.optimizers()

        opt_gen.zero_grad(set_to_none=True)
        opt_head.zero_grad(set_to_none=True)

        grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(X_s.shape[0],))
        
        grid = self.grid_t[grid_idx]
        inverse_grid = self.inverse_grid_t[grid_idx]
        mask_fan = self.mask_fan_t[grid_idx]

        X_ = self.USR(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
        X_s = X_s*mask_fan
        X_ = self.transform_us(X_)
        X = X_
        X_s = self.transform_us(X_s)
        
        labels = self.one_hot(X_s)
            
        
        h = h + self.gen(torch.rand(h.shape[0], 128).to(self.device))
        Y_fake, z_mu, z_sigma = self.decode(h, labels)
        Y_fake = Y_fake*mask_fan
        
        
        recons_loss = 0.0
        if self.current_epoch < self.hparams.warm_up_epochs:
            recons_loss = self.l1(Y_fake, X_)
        
        p_loss = self.perceptual_loss(Y_fake.float(), Y.float())
        
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        a_loss = 0.0
        if self.current_epoch >= self.hparams.warm_up_epochs:
            logits_fake = self.D(Y_fake.contiguous().float())[-1]
            a_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)        
        
        nce_loss = self.calculate_NCE_loss(feat_q, feat_k)
        
        loss_g = self.hparams.recons_weight*recons_loss + self.hparams.nce_weight*nce_loss + self.hparams.adversarial_weight * a_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        loss_g.backward()
        opt_gen.step()
        opt_head.step()
            
        # update D
        loss_d = 0.0
        if self.current_epoch >= self.hparams.warm_up_epochs:
            # self.set_requires_grad(self.D, True)
            opt_disc.zero_grad(set_to_none=True)
            loss_d = self.compute_D_loss(Y, Y_fake)
            loss_d.backward()
            opt_disc.step()

        # update G
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

    def validation_step(self, val_batch, batch_idx):

        labeled, Y = val_batch
        X_x = labeled['img']
        X_s = labeled['seg']
        
        grid_idx = torch.randint(low=0, high=self.hparams.n_grids - 1, size=(X_s.shape[0],))
        
        grid = self.grid_t[grid_idx]
        inverse_grid = self.inverse_grid_t[grid_idx]
        mask_fan = self.mask_fan_t[grid_idx]

        X_ = self.USR(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
        X_s = X_s*mask_fan
        X = self.transform_us(X_)
        X_s = self.transform_us(X_s)
        
        labels = self.one_hot(X_s)        
        
        feat_q, patch_ids_q, h = self.encode_nce(X, self.G.encoder.blocks)        
        h = h + self.gen(torch.rand(h.shape[0], 128).to(self.device))
        Y_fake, z_mu, z_sigma = self.decode(h, labels)
        Y_fake = Y_fake*mask_fan

        feat_k, _, _ = self.encode_nce(Y, self.G.encoder.blocks, patch_ids=patch_ids_q)
        
        recons_loss = 0.0
        if self.current_epoch < self.hparams.warm_up_epochs:
            recons_loss = self.l1(Y_fake, X_)
        
        p_loss = self.perceptual_loss(Y_fake.float(), Y.float())
        
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        a_loss = 0.0
        if self.current_epoch >= self.hparams.warm_up_epochs:
            logits_fake = self.D(Y_fake.contiguous().float())[-1]
            a_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
        
        nce_loss = self.calculate_NCE_loss(feat_q, feat_k)
        
        loss_g = self.hparams.recons_weight*recons_loss + self.hparams.nce_weight*nce_loss + self.hparams.adversarial_weight * a_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        self.log("val_loss", loss_g, sync_dist=True)
    
    def compute_D_loss(self, Y, Y_fake):
        logits_fake = self.D(Y_fake.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.D(Y.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = self.hparams.adversarial_weight * discriminator_loss
        
        return loss_d
    
    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.encode_nce(tgt, self.G.encoder.blocks)
        feat_k, _, _ = self.encode_nce(src, self.G.encoder.blocks, patch_ids=patch_ids_q)
        
        feat_q_pool = self.H(feat_q)
        feat_k_pool = self.H(feat_k)

        return self.nce_loss(feat_q_pool, feat_k_pool)        
    
    def encode_nce(self, x, blocks, patch_ids=None, blocks_ids=[0, 4, 7, 10, 14, 17], num_patches = 256):
        feat = x
        return_ids = []
        return_feats = []
        p_id = 0

        for block_id, block in enumerate(blocks):
            feat = block(feat)
            if block_id in blocks_ids:
                feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
                if patch_ids is not None:
                    patch_id = patch_ids[p_id]
                    p_id += 1
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1]) #, device=config.DEVICE
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))] # .to(patch_ids.device)
                    return_ids.append(patch_id)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])

                return_feats.append(x_sample)

        return return_feats, return_ids
    
    def one_hot(self, input_label):
        # One hot encoding function for the labels
        shape_ = list(input_label.shape)
        shape_[1] = self.hparams.num_labels
        label_out = torch.zeros(shape_).to(self.device)
        for channel in range(self.hparams.num_labels):
            label_out[:, channel, ...] = input_label[:, 0, ...] == channel
        return label_out
    