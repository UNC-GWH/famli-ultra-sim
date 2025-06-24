import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import functools

from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks import nets
from generative.networks.nets import PatchDiscriminator
from generative.losses import PatchAdversarialLoss, PerceptualLoss


from diffusers import DDPMScheduler, UNet2DModel



import lightning as L
from lightning.pytorch.core import LightningModule
from transforms import ultrasound_transforms as ust 


class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.05):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        if self.training:
            return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)
        return x
    
class RandCoarseShuffle(nn.Module):    
    def __init__(self, prob=0.75, holes=16, spatial_size=32):
        super(RandCoarseShuffle, self).__init__()
        self.t = transforms.RandCoarseShuffle(prob=prob, holes=holes, spatial_size=spatial_size)
    def forward(self, x):
        if self.training:
            return self.t(x)
        return x

class SaltAndPepper(nn.Module):    
    def __init__(self, prob=0.05):
        super(SaltAndPepper, self).__init__()
        self.prob = prob
    def __call__(self, x):
        noise_tensor = torch.rand(x.shape)
        salt = torch.max(x)
        pepper = torch.min(x)
        x[noise_tensor < self.prob/2] = salt
        x[noise_tensor > 1-self.prob/2] = pepper
        return x

class AutoEncoderKL(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        latent_channels = 3
        if hasattr(self.hparams, "latent_channels"):
            latent_channels = self.hparams.latent_channels

        self.autoencoderkl = nets.AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 384),
            latent_channels=latent_channels,
            num_res_blocks=1,
            norm_num_groups=32,
            attention_levels=(False, False, True),
        )

        # self.autoencoderkl = nets.AutoencoderKL(
        #     spatial_dims=2,
        #     in_channels=1,
        #     out_channels=1,
        #     num_channels=(128, 128, 256, 512),
        #     latent_channels=latent_channels,
        #     num_res_blocks=2,
        #     attention_levels=(False, False, False, False),
        #     with_encoder_nonlocal_attn=False,
        #     with_decoder_nonlocal_attn=False,
        # )

        # self.autoencoderkl = nets.AutoencoderKL(spatial_dims=2,
        #     in_channels=1,
        #     out_channels=1,
        #     num_channels=(128, 256, 512, 512),
        #     latent_channels=latent_channels,
        #     num_res_blocks=2,
        #     norm_num_groups=32,
        #     attention_levels=(False, False, False, True))

        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")

        self.automatic_optimization = False

        self.discriminator = nets.PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)        

        self.adversarial_loss = PatchAdversarialLoss(criterion="least_squares")        

        # For mixed precision training
        # self.scaler_g = GradScaler()
        # self.scaler_d = GradScaler()

        if hasattr(self.hparams, "denoise") and self.hparams.denoise: 
            self.noise_transform = torch.nn.Sequential(
                GaussianNoise(0.0, 0.05),
                RandCoarseShuffle(),
                SaltAndPepper()
            )
        else:
            self.noise_transform = nn.Identity()

        if hasattr(self.hparams, "smooth") and self.hparams.smooth: 
            self.smooth_transform = transforms.RandSimulateLowResolution(prob=1.0, zoom_range=(0.15, 0.3))
        else:
            self.smooth_transform = nn.Identity()
        
        

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.autoencoderkl.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        optimizer_d = optim.AdamW(self.discriminator.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer_g, optimizer_d]
    
    def compute_loss_generator(self, x, reconstruction, z_mu, z_sigma):
        recons_loss = F.l1_loss(reconstruction.float(), x.float())
        p_loss = self.perceptual_loss(reconstruction.float(), x.float())
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss_g = recons_loss + (self.hparams.kl_weight * kl_loss) + (self.hparams.perceptual_weight * p_loss)

        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_adv_loss = self.adversarial_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.hparams.adversarial_weight * generator_adv_loss

        return loss_g, recons_loss
    
    def compute_loss_discriminator(self, x, reconstruction):
        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adversarial_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(x.contiguous().detach())[-1]
        loss_d_real = self.adversarial_loss(logits_real, target_is_real=True, for_discriminator=True)
        return (loss_d_fake + loss_d_real) * 0.5


    def training_step(self, train_batch, batch_idx):
        x = train_batch

        optimizer_g, optimizer_d = self.optimizers()

        reconstruction, z_mu, z_sigma = self.autoencoderkl(self.smooth_transform(self.noise_transform(x)))

        loss_g, recons_loss = self.compute_loss_generator(x, reconstruction, z_mu, z_sigma)

        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)
        
        loss_d = 0.0
        if self.trainer.current_epoch > self.hparams.autoencoder_warm_up_n_epochs:
            loss_d = self.compute_loss_discriminator(x, reconstruction)

            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)

        self.log("train_loss_recons", recons_loss)
        self.log("train_loss_g", loss_g)
        self.log("train_loss_d", loss_d)

        return {"train_loss_g": loss_g, "train_loss_d": loss_d}
        

    def validation_step(self, val_batch, batch_idx):
        x = val_batch

        reconstruction, z_mu, z_sigma = self.autoencoderkl(x)
        recon_loss = F.l1_loss(x.float(), reconstruction.float())

        self.log("val_loss", recon_loss, sync_dist=True)

    def forward(self, images):        
        return self.autoencoderkl(images)



class DiffusionModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet2DModel(
            in_channels=self.hparams.in_channels,  # the number of input channels, 3 for RGB images
            out_channels=self.hparams.out_channels,  # the number of output channels
            layers_per_block=self.hparams.layers_per_block,  # how many ResNet layers to use per UNet block
            block_out_channels=self.hparams.block_out_channels,  # the number of output channes for each UNet block
            down_block_types=self.hparams.down_block_types,  # the types of blocks to use in the downsampling part of the UNet
            up_block_types=self.hparams.up_block_types,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=self.hparams.num_train_timesteps, beta_schedule="linear")
        self.loss_fn = nn.MSELoss()

        self.resize = ust.Resize2D(self.hparams.image_size)

    @staticmethod
    def add_model_specific_args(parent_parser):

        hparams_group = parent_parser.add_argument_group('DiffusionModel Hugging Face')
        hparams_group.add_argument('--lr', default=1e-4, type=float, help='Learning rate generator')
        hparams_group.add_argument('--num_train_timesteps', default=1000, type=int, help='Number of training timesteps')
        hparams_group.add_argument('--in_channels', default=1, type=int, help='Number of input channels')
        hparams_group.add_argument('--out_channels', default=1, type=int, help='Number of output channels')
        hparams_group.add_argument('--layers_per_block', default=2, type=int, help='Number of layers per block')
        hparams_group.add_argument('--block_out_channels', default=(128, 128, 256, 256, 512, 512), nargs='+', type=int, help='Block out channels')
        hparams_group.add_argument('--down_block_types', default=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"), nargs='+', type=str, help='Down block types')
        hparams_group.add_argument('--up_block_types', default=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"), nargs='+', type=str, help='Up block types')
        hparams_group.add_argument('--image_size', default=(128, 128), type=int, nargs='+', help='Image size')

        return parent_parser
        

    def forward(self, x, timesteps):
        return self.net(x, timesteps).sample

    def training_step(self, batch, batch_idx):
        x = batch  

        x = self.resize(x)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(x).to(self.device)
        noisy_images = self.scheduler.add_noise(x, noise, timesteps)

        pred = self(noisy_images, timesteps)  # Model forward pass
        loss = self.loss_fn(pred, noise)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch  
        x = self.resize(x)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.size(0),), device=self.device).long()
        noise = torch.randn_like(x).to(self.device)
        noisy_images = self.scheduler.add_noise(x, noise, timesteps)

        pred = self(noisy_images, timesteps)  # Model forward pass
        loss = self.loss_fn(pred, noise)

        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def sample(self, num_samples=1):

        size = list(self.hparams.image_size)
        size = [num_samples, self.hparams.in_channels] + size

        noisy_image = torch.randn(size).to(self.device)
        # Sample timesteps
        timesteps = self.scheduler.timesteps.to(self.device)

        # Reverse process (denoising)
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                # Predict noise
                noise_pred = self.net(noisy_image, t.unsqueeze(0)).sample
                
                # Remove noise using scheduler
                noisy_image = self.scheduler.step(noise_pred, t, noisy_image).prev_sample

        return noisy_image
