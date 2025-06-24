import torch
from tqdm.auto import tqdm
from diffusers import DDPMScheduler
from nets.DiffusionModel import DiffusionModel
import os
import nrrd

device = torch.device("cuda")

checkpoint_path = "/mnt/raid/home/ajarry/data/outputs_lightning/CSVoutput_20250620_173053/model.pth"
model = DiffusionModel()
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)

# Define the scheduler (make sure it matches the one used in training)
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

# Set up sampling parameters
image_size = (1, 1, 128, 128)  # Adjust shape (batch, channels, height, width)
num_inference_steps =200  # Can be reduced for faster generation
scheduler.set_timesteps(num_inference_steps)
# Start with pure noise
device = "cuda" if torch.cuda.is_available() else "cpu"
noisy_image = torch.randn(image_size).to(device)

# Sample timesteps
timesteps = scheduler.timesteps.to(device)

# Reverse process (denoising)
with torch.no_grad():
    for i, t in tqdm(enumerate(timesteps)):
        # Predict noise
        noise_pred = model(noisy_image, t.unsqueeze(0))
        
        # Remove noise using scheduler
        noisy_image = scheduler.step(noise_pred, t, noisy_image).prev_sample
# Convert to image format

generated_image = noisy_image.cpu().numpy().squeeze(0).permute(1,2,0)
nrrd.write(file=".", data=generated_image)
print(generated_image.shape)
