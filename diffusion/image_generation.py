import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import nrrd
import SimpleITK as sitk
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
import os
import argparse

def init_world(data_path, num_inference_steps):
    # Set up device (GPU)
    device = torch.device("cuda")

    class DiffusionModel(pl.LightningModule):
        def __init__(self, lr=1e-5, num_train_timesteps=1000):
            super().__init__()
            self.save_hyperparameters()  # log hyperparameters automatically
            # Initialize the UNet denoising model
            self.net = UNet2DModel(
                in_channels=1, out_channels=1,
                layers_per_block=2,
                block_out_channels=(256, 256, 512, 512, 1024, 1024),
                down_block_types=(
                    "DownBlock2D", "DownBlock2D", "DownBlock2D", 
                    "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
                    "UpBlock2D", "UpBlock2D", "UpBlock2D"
                ),
            )
            # Scheduler for training noise schedule
            self.scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps,
                                           beta_schedule="linear")
            self.loss_fn = nn.MSELoss()

        def forward(self, x, timesteps):
            # Return only the denoised sample
            return self.net(x, timesteps).sample

        def training_step(self, batch, batch_idx):
            # Standard training step for diffusion model
            x = batch
            steps = self.scheduler.config.num_train_timesteps
            timesteps = torch.randint(0, steps, (x.size(0),),
                                      device=self.device).long()
            noise = torch.randn_like(x).to(self.device)
            noisy_images = self.scheduler.add_noise(x, noise, timesteps)
            pred = self(noisy_images, timesteps)
            loss = self.loss_fn(pred, noise)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x = batch
            steps = self.scheduler.config.num_train_timesteps
            timesteps = torch.randint(0, steps, (x.size(0),),
                                      device=self.device)
            outputs = self(x, timesteps)
            loss = self.loss_fn(outputs, x)
            self.log("val_loss", loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return optim.Adam(self.net.parameters(), lr=self.hparams.lr)

    # Load checkpoint into LightningModule
    checkpoint_path = "/mnt/raid/home/ajarry/data/outputs_lightning/final53epoch/model.pth"
    model = DiffusionModel()
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # Set up inference scheduler
    scheduler = DDIMScheduler(beta_start=0.0001, beta_end=0.02,
                              beta_schedule="linear")
    scheduler.set_timesteps(num_inference_steps)

    # Load target image from disk and normalize to [0,1], add channel dim
    data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
    targets = torch.tensor(data / 255.0).unsqueeze(1)

    return device, model, scheduler, targets

def guidance_loss(image, target):
    # Use L1 difference as guidance loss
    return torch.abs(image - target).mean()

def inference(device, model, scheduler, targets, guidance_scale):
    print("Generating image...")
    noise = torch.randn(1, 1, 256, 256, device=device)
    stack = []

    for guide in torch.chunk(targets.to(device), 3):
        x = noise.repeat(guide.shape[0], 1, 1, 1)
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            with torch.no_grad():
                noise_pred = model(x, t)

            x = x.detach().requires_grad_()
            x0 = scheduler.step(noise_pred, t, x).pred_original_sample

            # Compute tweak using guidance loss gradient
            loss = guidance_loss(x0, guide) * guidance_scale
            cond_grad = -torch.autograd.grad(loss, x)[0]
            x = x.detach() + cond_grad
            x = scheduler.step(noise_pred, t, x).prev_sample

        stack.append(x.cpu())

    # Combine chunks and remove channel dimension
    cat = torch.cat(stack).squeeze(1).cpu().numpy()
    print("Image generated!")
    return cat

def image_generation(data_path, num_inference_steps,
                            guidance_scale, output_folder_path):
    print("Using", data_path, "as guidance")
    device, model, scheduler, targets = init_world(
        data_path=data_path,
        num_inference_steps=num_inference_steps
    )
    generated_output = inference(device, model, scheduler,
                                 targets, guidance_scale)

    # Extract frame number and basename from input path
    split_input = data_path.split('/')
    frame_number = split_input[-2]  # second last element
    basename = split_input[-1]      # last element
    out_dir = os.path.join(output_folder_path, frame_number)
    save_path = os.path.join(out_dir, basename)

    os.makedirs(out_dir, exist_ok=True)
    # Write NRRD file (expects NumPy array)
    nrrd.write(save_path, generated_output, index_order='C')
    print("Image saved at:", save_path)

def main():
    # Set up CLI interface
    parser = argparse.ArgumentParser(
        description="Generate ultrasound images via diffusion"
    )

    # Required positional args
    parser.add_argument(
        'input_data_path',
        help='Path to guidance image'
    )
    parser.add_argument(
        'output_folder_path',
        help='Folder to save generated images'
    )

    # Optional args with defaults
    parser.add_argument(
        '-n', '--num_inference_steps',
        type=int, default=20,
        help='Number of inference steps (default: %(default)s)'
    )
    parser.add_argument(
        '-g', '--guidance_scale',
        type=int, default=30000,
        help='Strength of guidance (default: %(default)s)'
    )

    args = parser.parse_args()  # parse command-line input

    print("Data path: ", args.input_data_path)
    print("Output folder path: ", args.output_folder_path)
    print("Number of inference steps:", args.num_inference_steps)
    print("Guidance scale:", args.guidance_scale)

    image_generation(
        data_path=args.input_data_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_folder_path=args.output_folder_path
    )

if __name__ == "__main__":
    main()
