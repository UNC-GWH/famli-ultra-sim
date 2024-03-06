import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch import nn

from generative.networks import nets as MNets
from nets.cut_D import Discriminator
from nets.cut_G import Generator
from nets.cut_P import Head
from nets.lotus import UltrasoundRendering, UltrasoundRenderingLinear, UltrasoundRenderingConv1d

import pytorch_lightning as pl
import os

from torchvision import transforms as T


class Cut(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.D_Y = Discriminator()

        self.USR = UltrasoundRendering(**kwargs)

        if hasattr(self.hparams, 'use_pre_trained_lotus') and self.hparams.use_pre_trained_lotus and os.path.exists('train_output/ultra-sim/rendering/v0.1/epoch=199-val_loss=0.04.ckpt'):
            usr = UltrasoundRendering.load_from_checkpoint('train_output/ultra-sim/rendering/v0.1/epoch=199-val_loss=0.04.ckpt')        
            
            
            self.USR.acoustic_impedance_dict = usr.acoustic_impedance_dict
            self.USR.attenuation_dict = usr.attenuation_dict
            self.USR.mu_0_dict = usr.mu_0_dict
            self.USR.mu_1_dict = usr.mu_1_dict
            self.USR.sigma_0_dict = usr.sigma_0_dict

        self.G = Generator()
        self.H = Head()

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # lambda_lr = lambda epoch: 1.0 - max(0, epoch - config.NUM_EPOCHS / 2) / (config.NUM_EPOCHS / 2)
        # self.scheduler_disc = lr_scheduler.LambdaLR(self.opt_disc, lr_lambda=lambda_lr)
        # self.scheduler_gen = lr_scheduler.LambdaLR(self.opt_gen, lr_lambda=lambda_lr)
        # self.scheduler_mlp = lr_scheduler.LambdaLR(self.opt_head, lr_lambda=lambda_lr)

        self.automatic_optimization = False

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])

    def configure_optimizers(self):
        opt_gen = optim.AdamW(
            list(self.USR.parameters()) + list(self.G.parameters()),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay            
        )
        opt_disc = optim.AdamW(
            self.D_Y.parameters(),
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
        return self.G(self.transform_us(self.USR(X)))*self.transform_us(self.USR.mask_fan).to(self.device)

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

    def on_train_start(self):

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

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
        
        else:
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

        X_ = self.USR(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
        X = self.transform_us(X_)
        Y_fake = self.G(X)

        Y_idt = None
        if self.hparams.lambda_y:
            Y_idt = self.G(Y)           

        # update D
        self.set_requires_grad(self.D_Y, True)
        opt_disc.zero_grad()
        loss_d = self.compute_D_loss(Y, Y_fake)
        loss_d.backward()
        opt_disc.step()

        # update G
        self.set_requires_grad(self.D_Y, False)
        opt_gen.zero_grad()
        opt_head.zero_grad()
        loss_g = self.compute_G_loss(X, Y, Y_fake, Y_idt)

        if self.current_epoch < self.hparams.warm_up_epochs_diffusor:
            loss_g = loss_g + self.hparams.diffusor_w*self.mse(X_, self.USR.add_speckle_noise(X_x)*mask_fan)

        loss_g.backward()
        opt_gen.step()
        opt_head.step()

        
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

    def validation_step(self, val_batch, batch_idx):

        labeled, Y = val_batch
        X_x = labeled['img']
        X_s = labeled['seg']

        X_ = self.USR(X_s)
        X = self.transform_us(X_)
        Y_fake = self.G(X)

        Y_idt = None
        if self.hparams.lambda_y:
            Y_idt = self.G(Y)
        
        loss_G = self.compute_G_loss(X, Y, Y_fake, Y_idt)

        self.log("val_loss", loss_G, sync_dist=True)
        

    def compute_D_loss(self, Y, Y_fake):
        # Fake
        fake = Y_fake.detach()
        pred_fake = self.D_Y(fake)
        loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        # Real
        pred_real = self.D_Y(Y)
        loss_D_real = self.mse(pred_real, torch.ones_like(pred_real))

        loss_D_Y = (loss_D_fake + loss_D_real) / 2
        return loss_D_Y

    def compute_G_loss(self, X, Y, Y_fake, Y_idt = None):
        fake = Y_fake
        pred_fake = self.D_Y(fake)
        loss_G_adv = self.mse(pred_fake, torch.ones_like(pred_fake))

        loss_NCE = self.calculate_NCE_loss(X, Y_fake)
        if self.hparams.lambda_y > 0:
            loss_NCE_Y = self.calculate_NCE_loss(Y, Y_idt)
            loss_NCE = (loss_NCE + loss_NCE_Y) * 0.5

        loss_G = loss_G_adv + loss_NCE
        return loss_G

    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.G(tgt, encode_only=True)
        feat_k, _ = self.G(src, encode_only=True, patch_ids=patch_ids_q)

        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
        loss = self.cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=self.device))
        return loss
    

class CutLinear(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.D_Y = Discriminator()

        self.USR = UltrasoundRenderingLinear(**kwargs)

        self.G = Generator()
        self.H = Head()

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # lambda_lr = lambda epoch: 1.0 - max(0, epoch - config.NUM_EPOCHS / 2) / (config.NUM_EPOCHS / 2)
        # self.scheduler_disc = lr_scheduler.LambdaLR(self.opt_disc, lr_lambda=lambda_lr)
        # self.scheduler_gen = lr_scheduler.LambdaLR(self.opt_gen, lr_lambda=lambda_lr)
        # self.scheduler_mlp = lr_scheduler.LambdaLR(self.opt_head, lr_lambda=lambda_lr)

        self.automatic_optimization = False

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])

    def configure_optimizers(self):
        opt_gen = optim.AdamW(
            self.G.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay            
        )
        opt_disc = optim.AdamW(
            self.D_Y.parameters(),
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
        return self.G(self.transform_us(self.USR(X)))*self.transform_us(self.USR.mask_fan).to(self.device)

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

    def on_train_start(self):

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

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
        
        else:
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

        X_ = self.USR(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
        X = self.transform_us(X_)
        Y_fake = self.G(X)

        Y_idt = None
        if self.hparams.lambda_y:
            Y_idt = self.G(Y)           

        # update D
        self.set_requires_grad(self.D_Y, True)
        opt_disc.zero_grad()
        loss_d = self.compute_D_loss(Y, Y_fake)
        loss_d.backward()
        opt_disc.step()

        # update G
        self.set_requires_grad(self.D_Y, False)
        opt_gen.zero_grad()
        opt_head.zero_grad()
        loss_g = self.compute_G_loss(X, Y, Y_fake, Y_idt)

        loss_g.backward()
        opt_gen.step()
        opt_head.step()
        
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

    def validation_step(self, val_batch, batch_idx):

        labeled, Y = val_batch
        X_x = labeled['img']
        X_s = labeled['seg']

        X_ = self.USR(X_s)
        X = self.transform_us(X_)
        Y_fake = self.G(X)

        Y_idt = None
        if self.hparams.lambda_y:
            Y_idt = self.G(Y)
        
        loss_G = self.compute_G_loss(X, Y, Y_fake, Y_idt)

        self.log("val_loss", loss_G, sync_dist=True)
        

    def compute_D_loss(self, Y, Y_fake):
        # Fake
        fake = Y_fake.detach()
        pred_fake = self.D_Y(fake)
        loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        # Real
        pred_real = self.D_Y(Y)
        loss_D_real = self.mse(pred_real, torch.ones_like(pred_real))

        loss_D_Y = (loss_D_fake + loss_D_real) / 2
        return loss_D_Y

    def compute_G_loss(self, X, Y, Y_fake, Y_idt = None):
        fake = Y_fake
        pred_fake = self.D_Y(fake)
        loss_G_adv = self.mse(pred_fake, torch.ones_like(pred_fake))

        loss_NCE = self.calculate_NCE_loss(X, Y_fake)
        if self.hparams.lambda_y > 0:
            loss_NCE_Y = self.calculate_NCE_loss(Y, Y_idt)
            loss_NCE = (loss_NCE + loss_NCE_Y) * 0.5

        loss_G = loss_G_adv + loss_NCE
        return loss_G

    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.G(tgt, encode_only=True)
        feat_k, _ = self.G(src, encode_only=True, patch_ids=patch_ids_q)

        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
        loss = self.cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=self.device))
        return loss


class CutAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.autoencoderkl = MNets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 384),
            latent_channels=3,
            num_res_blocks=1,
            norm_num_groups=32,
            attention_levels=(False, False, True),
        )
        
        self.D_Y = Discriminator(in_channels=3)

        self.USR = UltrasoundRenderingLinear(**kwargs)

        self.G = Generator(in_channels=3)
        self.H = Head(in_channels=3)

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # lambda_lr = lambda epoch: 1.0 - max(0, epoch - config.NUM_EPOCHS / 2) / (config.NUM_EPOCHS / 2)
        # self.scheduler_disc = lr_scheduler.LambdaLR(self.opt_disc, lr_lambda=lambda_lr)
        # self.scheduler_gen = lr_scheduler.LambdaLR(self.opt_gen, lr_lambda=lambda_lr)
        # self.scheduler_mlp = lr_scheduler.LambdaLR(self.opt_head, lr_lambda=lambda_lr)

        self.automatic_optimization = False

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])

    def configure_optimizers(self):
        opt_gen = optim.AdamW(
            list(self.USR.parameters()) + list(self.G.parameters()),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay            
        )
        opt_disc = optim.AdamW(
            self.D_Y.parameters(),
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
        X = self.transform_us(self.USR(X))
        X = self.autoencoderkl.encoder(X)
        return self.G(X)

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

    def on_train_start(self):

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

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
        
        else:
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

        X_ = self.USR(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
        X = self.transform_us(X_)

        with torch.no_grad():
            Y = self.autoencoderkl.encoder(Y)
            X = self.autoencoderkl.encoder(X)

        Y_fake = self.G(X)

        Y_idt = None
        if self.hparams.lambda_y:
            Y_idt = self.G(Y)           

        # update D
        self.set_requires_grad(self.D_Y, True)
        opt_disc.zero_grad()
        loss_d = self.compute_D_loss(Y, Y_fake)
        loss_d.backward()
        opt_disc.step()

        # update G
        self.set_requires_grad(self.D_Y, False)
        opt_gen.zero_grad()
        opt_head.zero_grad()
        loss_g = self.compute_G_loss(X, Y, Y_fake, Y_idt)

        loss_g.backward()
        opt_gen.step()
        opt_head.step()
        
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

    def validation_step(self, val_batch, batch_idx):

        labeled, Y = val_batch
        X_x = labeled['img']
        X_s = labeled['seg']

        X_ = self.USR(X_s)
        X = self.transform_us(X_)

        Y = self.autoencoderkl.encoder(Y)
        X = self.autoencoderkl.encoder(X)

        Y_fake = self.G(X)

        Y_idt = None
        if self.hparams.lambda_y:
            Y_idt = self.G(Y)
        
        loss_G = self.compute_G_loss(X, Y, Y_fake, Y_idt)

        self.log("val_loss", loss_G, sync_dist=True)
        

    def compute_D_loss(self, Y, Y_fake):
        # Fake
        fake = Y_fake.detach()
        pred_fake = self.D_Y(fake)
        loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        # Real
        pred_real = self.D_Y(Y)
        loss_D_real = self.mse(pred_real, torch.ones_like(pred_real))

        loss_D_Y = (loss_D_fake + loss_D_real) / 2
        return loss_D_Y

    def compute_G_loss(self, X, Y, Y_fake, Y_idt = None):
        fake = Y_fake
        pred_fake = self.D_Y(fake)
        loss_G_adv = self.mse(pred_fake, torch.ones_like(pred_fake))

        loss_NCE = self.calculate_NCE_loss(X, Y_fake)
        if self.hparams.lambda_y > 0:
            loss_NCE_Y = self.calculate_NCE_loss(Y, Y_idt)
            loss_NCE = (loss_NCE + loss_NCE_Y) * 0.5

        loss_G = loss_G_adv + loss_NCE
        return loss_G

    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.G(tgt, encode_only=True)
        feat_k, _ = self.G(src, encode_only=True, patch_ids=patch_ids_q)

        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
        loss = self.cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=self.device))
        return loss



class CutLotus(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.D_Y = Discriminator()

        self.USR = UltrasoundRendering(**kwargs)

        self.G = Generator()
        self.H = Head()

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # lambda_lr = lambda epoch: 1.0 - max(0, epoch - config.NUM_EPOCHS / 2) / (config.NUM_EPOCHS / 2)
        # self.scheduler_disc = lr_scheduler.LambdaLR(self.opt_disc, lr_lambda=lambda_lr)
        # self.scheduler_gen = lr_scheduler.LambdaLR(self.opt_gen, lr_lambda=lambda_lr)
        # self.scheduler_mlp = lr_scheduler.LambdaLR(self.opt_head, lr_lambda=lambda_lr)

        self.automatic_optimization = False

        self.transform_us = T.Compose([T.Pad((0, 80, 0, 0)), T.CenterCrop(256)])

    def configure_optimizers(self):
        opt_gen = optim.AdamW(
            # list(self.USR.parameters()) + list(self.G.parameters()),
            self.USR.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay            
        )
        opt_disc = optim.AdamW(
            self.D_Y.parameters(),
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
        return self.G(self.transform_us(self.USR(X)))*self.transform_us(self.USR.mask_fan)

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

    def on_train_start(self):

        # Define the file names directly without using out_dir
        grid_t_file = 'grid_t.pt'
        inverse_grid_t_file = 'inverse_grid_t.pt'
        mask_fan_t_file = 'mask_fan_t.pt'

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
        
        else:
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

        X_ = self.USR(X_s, grid=grid, inverse_grid=inverse_grid, mask_fan=mask_fan)
        X = self.transform_us(X_)
        Y_fake = self.G(X)

        Y_idt = None
        if self.hparams.lambda_y:
            Y_idt = self.G(Y)           

        # update D
        self.set_requires_grad(self.D_Y, True)
        opt_disc.zero_grad()
        loss_d = self.compute_D_loss(Y, Y_fake)
        loss_d.backward()
        opt_disc.step()

        # update G
        self.set_requires_grad(self.D_Y, False)
        opt_gen.zero_grad()
        opt_head.zero_grad()
        loss_g = self.compute_G_loss(X, Y, Y_fake, Y_idt)

        loss_g.backward()
        opt_gen.step()
        opt_head.step()
        
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

    def validation_step(self, val_batch, batch_idx):

        labeled, Y = val_batch
        X_x = labeled['img']
        X_s = labeled['seg']

        X_ = self.USR(X_s)
        X = self.transform_us(X_)
        Y_fake = self.G(X)

        Y_idt = None
        if self.hparams.lambda_y:
            Y_idt = self.G(Y)
        
        loss_G = self.compute_G_loss(X, Y, Y_fake, Y_idt)

        self.log("val_loss", loss_G, sync_dist=True)
        

    def compute_D_loss(self, Y, Y_fake):
        # Fake
        fake = Y_fake.detach()
        pred_fake = self.D_Y(fake)
        loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        # Real
        pred_real = self.D_Y(Y)
        loss_D_real = self.mse(pred_real, torch.ones_like(pred_real))

        loss_D_Y = (loss_D_fake + loss_D_real) / 2
        return loss_D_Y

    def compute_G_loss(self, X, Y, Y_fake, Y_idt = None):
        fake = Y_fake
        pred_fake = self.D_Y(fake)
        loss_G_adv = self.mse(pred_fake, torch.ones_like(pred_fake))

        loss_NCE = self.calculate_NCE_loss(X, Y_fake)
        if self.hparams.lambda_y > 0:
            loss_NCE_Y = self.calculate_NCE_loss(Y, Y_idt)
            loss_NCE = (loss_NCE + loss_NCE_Y) * 0.5

        loss_G = loss_G_adv + loss_NCE
        return loss_G

    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.G(tgt, encode_only=True)
        feat_k, _ = self.G(src, encode_only=True, patch_ids=patch_ids_q)

        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
        loss = self.cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=self.device))
        return loss

class CUTModelLightning(pl.LightningModule):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        super().__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate gradients for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Unpack the batch data
        self.set_input(batch)

        # Forward pass through the networks
        self.forward()

        # Compute the loss for the generator
        if optimizer_idx == 0:
            loss_G = self.compute_G_loss()
            self.log('loss_G', loss_G, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss_G

        # Compute the loss for the discriminator
        if optimizer_idx == 1:
            loss_D = self.compute_D_loss()
            self.log('loss_D', loss_D, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss_D
