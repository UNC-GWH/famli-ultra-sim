import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm
import nrrd
import SimpleITK as sitk
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import json
import cv2
import os

device = torch.device("cuda")

class DiffusionModel(pl.LightningModule):
    def __init__(self, lr=1e-5, num_train_timesteps=1000):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet2DModel(
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(256, 256, 512, 512, 1024, 1024),  # the number of output channes for each UNet block
            down_block_types=( 
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ), 
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"  
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

# checkpoint_path = "/mnt/raid/home/ajarry/data/outputs_lightning/final53epoch/model.pth"
# model = DiffusionModel()
# state_dict = torch.load(checkpoint_path)
# model.load_state_dict(state_dict, strict=False)
# model = model.to(device)
# scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# # Set up sampling parameters
# image_size = (1, 1, 256, 256)
# num_inference_steps =200  


# scheduler.set_timesteps(num_inference_steps)
# # Start with pure noise
# device = "cuda" if torch.cuda.is_available() else "cpu"
# noisy_image = torch.randn(image_size).to(device)

# # Sample timesteps
# timesteps = scheduler.timesteps.to(device)

# data = sitk.GetArrayFromImage(sitk.ReadImage('/mnt/raid/home/ajarry/data/cephalic_sweeps/frame_0001/C3_us.nrrd'))
# print(data.shape)
# totensor = transforms.ToTensor()
# targets=[]
# for frame in data:
#     targets.append(totensor(np.uint8(frame)).to(device))

# def guidance_loss(image, target):
#     return torch.abs(image - target).mean()


# guidance_loss_scale = 175 # Explore changing this to 5, or 100

# noise = torch.randn(1, 1, 256, 256).to(device)
# stack = []
# for target in targets:
#     x = noise
#     for i, t in tqdm(enumerate(scheduler.timesteps)):
#         with torch.no_grad():
#             noise_pred = model(x, t)
#         x = x.detach().requires_grad_()
#         x0 = scheduler.step(noise_pred, t, x).pred_original_sample
#         loss = guidance_loss(x0, target) * guidance_loss_scale
#         cond_grad = -torch.autograd.grad(loss, x)[0]
#         x = x.detach() + cond_grad
#         x = scheduler.step(noise_pred, t, x).prev_sample
#     stack.append(x.squeeze(0).cpu().permute(2,1,0))



def load_model_and_scheduler(checkpoint_path):
    model = DiffusionModel()
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict, strict=False)
    model = model.to("cuda")
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    return model, scheduler

def inference_worker(nrrd_path, checkpoint_path, guidance_loss_scale):
    torch.cuda.set_device(0)  # Use GPU 0; adjust if running multi-GPU setup

    model, scheduler = load_model_and_scheduler(checkpoint_path)
    scheduler.set_timesteps(200)

    data = sitk.GetArrayFromImage(sitk.ReadImage(nrrd_path))
    totensor = transforms.ToTensor()
    targets = [totensor(np.uint8(frame)).to("cuda") for frame in data]

    noise = torch.randn(1, 1, 256, 256).to("cuda")
    stack = []

    for target in targets:
        x = noise.clone()
        for i, t in tqdm(enumerate(scheduler.timesteps), total=len(scheduler.timesteps), leave=False):
            with torch.no_grad():
                noise_pred = model(x, t)
            x = x.detach().requires_grad_()
            x0 = scheduler.step(noise_pred, t, x).pred_original_sample
            loss = torch.abs(x0 - target).mean() * guidance_loss_scale
            cond_grad = -torch.autograd.grad(loss, x)[0]
            x = x.detach() + cond_grad
            x = scheduler.step(noise_pred, t, x).prev_sample
        stack.append(x.squeeze(0).cpu().permute(2, 1, 0))

    out_array = torch.stack(stack).numpy()
    out_img = sitk.GetImageFromArray(out_array)
    out_path = os.path('/mnt/raid/home/ajarry/data/cephalic_output/')
    sitk.WriteImage(out_img, out_path)
    print(f"✓ {nrrd_path} → {out_path}")

def run_parallel_inference(nrrd_paths, checkpoint_path, guidance_loss_scale=175):
    ctx = mp.get_context("spawn")  # CUDA-safe process spawn
    with ctx.Pool(processes=2) as pool:
        pool.starmap(inference_worker, [(p, checkpoint_path, 175) for p in nrrd_paths])
    processes = []

    for nrrd_path in nrrd_paths:
        p = ctx.Process(target=inference_worker, args=(nrrd_path, checkpoint_path, guidance_loss_scale))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    import glob
    checkpoint_path = "/mnt/raid/home/ajarry/data/outputs_lightning/final53epoch/model.pth"
    nrrd_paths = [
        path for path in glob.glob("/mnt/raid/home/ajarry/data/cephalic_sweeps/frame_*/*")
        if path.endswith(".nrrd") and "us" in os.path.basename(path).lower()
    ]

    run_parallel_inference(nrrd_paths, checkpoint_path)
