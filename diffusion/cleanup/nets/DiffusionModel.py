import torch
import lightning as l
import torch.nn as nn
import torch.optim as optim
from diffusers import DDPMScheduler, UNet2DModel
from Resize2D import Resize2D

class DiffusionModel(l.LightningModule):
    def __init__(self, lr=1e-5, num_train_timesteps=1000):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet2DModel(
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(256, 256, 512, 512, 1024, 1024),  # the number of output channes for each UNet block
            down_block_types=( 
                "DownBlock2D", "DownBlock2D", "DownBlock2D", 
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
                ), 
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", 
                "UpBlock2D", "UpBlock2D", "UpBlock2D"  
            ),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="linear")
        self.loss_fn = nn.MSELoss()

    def forward(self, x, timesteps):
        return self.net(x, timesteps).sample

    def training_step(self, batch, batch_idx):
        x= batch  # Ignore labels if they exist
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(x).to(self.device)
        noisy_images = self.scheduler.add_noise(x, noise, timesteps)

        pred = self(noisy_images, timesteps)  # Model forward pass
        loss = self.loss_fn(pred, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        outputs = self(x, timesteps)
        loss = self.loss_fn(outputs, x)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer
    
class DiffusionModel128(l.LightningModule):
    def __init__(self, lr=1e-5, num_train_timesteps=1000):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet2DModel(
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(256, 256, 512, 512, 1024, 1024),  # the number of output channes for each UNet block
            down_block_types=( 
                "DownBlock2D", "DownBlock2D", "DownBlock2D", 
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
                ), 
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", 
                "UpBlock2D", "UpBlock2D", "UpBlock2D"  
            ),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="linear")
        self.loss_fn = nn.MSELoss()
        self.resize = Resize2D((128,128))

    def forward(self, x, timesteps):
        return self.net(x, timesteps).sample

    def training_step(self, batch, batch_idx):
        x= self.resize(batch)  # Ignore labels if they exist
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(x).to(self.device)
        noisy_images = self.scheduler.add_noise(x, noise, timesteps)

        pred = self(noisy_images, timesteps)  # Model forward pass
        loss = self.loss_fn(pred, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device)
        outputs = self(x, timesteps)
        loss = self.loss_fn(outputs, x)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer