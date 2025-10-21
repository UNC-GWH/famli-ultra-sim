
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import os

from nets import diffusion
from transforms import ultrasound_transforms as ust 

from diffusers import DDIMScheduler

from nets.us_simu import SweepSampling

import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure

import torch.multiprocessing as mp

import pandas as pd

import argparse

import SimpleITK as sitk

def image_generation(gpu_id, args, df_split):

    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    mount_point = args.mount_point
    model = diffusion.DiffusionModel.load_from_checkpoint(os.path.join(mount_point, 'train_output/diffusion/0.1/epoch=76-val_loss=0.01.ckpt'))
    model.eval()
    model = model.to(device)

    AE = diffusion.AutoEncoderKL.load_from_checkpoint(os.path.join(mount_point, "train_output/diffusionAE/extract_frames_Dataset_C_masked_resampled_256_spc075_wscores_meta_BPD01_MACFL025-7mo-9mo/v0.4/epoch=72-val_loss=0.01.ckpt"))
    AE.eval()
    AE = AE.to(device)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def ssim_loss(image, target):
        # SSIM returns a similarity (1 = identical)
        return 1.0 - ssim_metric(image, target)

    perceptual_metric = lpips.LPIPS(net='vgg').to(device)

    def perceptual_loss(image, target):
        # LPIPS expects 3-channel images in [-1, 1]
        image_3c = image.repeat(1, 3, 1, 1) * 2 - 1
        target_3c = target.repeat(1, 3, 1, 1) * 2 - 1
        return perceptual_metric(image_3c, target_3c).mean()


    def guidance_loss(image, target, weights=(1.0, 1.0, 0.1)):
        l1 = torch.abs(image - target).mean()
        ssim = ssim_loss(image, target)
        lpips_val = perceptual_loss(image, target)
        return weights[0] * l1 + weights[1] * ssim + weights[2] * lpips_val

    def inference(model, scheduler, targets, guidance_scale, chunks=1, ae_chunks=4, weights=(1.0, 1.0, 0.1), noise_epsilon = 0.05 ):
        # print("Generating image...")
        # noise = torch.randn(1, 1, model.hparams.image_size[1], model.hparams.image_size[0], device=targets.device)
        base_noise = torch.randn(1, 1, model.hparams.image_size[1], model.hparams.image_size[0], device=targets.device)    

        stack = []

        resize_128 = ust.Resize2D((128, 128))
        resize_256 = ust.Resize2D((256, 256))

        for guide in torch.chunk(targets, chunks=chunks, dim=1):
            guide = guide.permute(1, 0, 2, 3)  # Change to (B, C, H, W)
            guide = resize_128(guide)
            
            x = base_noise + noise_epsilon * torch.randn_like(guide)
            
            for i, t in enumerate(scheduler.timesteps):
                with torch.no_grad():
                    noise_pred = model(x, t)

                x = x.detach().requires_grad_()
                x0 = scheduler.step(model_output=noise_pred, timestep=t, sample=x).pred_original_sample

                # Compute tweak using guidance loss gradient
                loss = guidance_loss(x0, guide, weights) * guidance_scale
                cond_grad = -torch.autograd.grad(loss, x, retain_graph=False)[0]
                cond_grad = cond_grad / (cond_grad.norm() + 1e-8)
                x = x.detach() + guidance_scale * cond_grad
                
                x = scheduler.step(noise_pred, t, x).prev_sample
            
            with torch.no_grad():
                x_chunk = []
                for x_c in torch.chunk(x, chunks=ae_chunks, dim=0):
                    x_c = AE(resize_256(x_c))[0]
                    x_chunk.append(x_c)
                x = torch.cat(x_chunk, dim=0)
            stack.append(x.cpu())

        # Combine chunks and remove channel dimension
        cat = torch.cat(stack).squeeze(1)
        # print("Image generated!")
        return cat

    for i, row in df_split[gpu_id].iterrows():

        diffusor_fn = row['img']
        probe_paths = row['probe_paths']

        ss = SweepSampling(diffusor_fn=diffusor_fn, probe_paths=probe_paths, **vars(args)).to(device)
        sweeps, sweeps_diff, tags = ss.volume_sampling()

        num_inference_steps = args.num_inference_steps
        guidance_scale = args.guidance_scale

        scheduler = DDIMScheduler()
        # scheduler = LMSDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps)
        
        for tag, sweep, sweep_diff in zip(tags, sweeps[0], sweeps_diff[0]):

            tag_name = ss.vs.tags[tag]

            out_path = os.path.join(args.out, os.path.splitext(diffusor_fn)[0], f'{tag_name}_label.nrrd')
            out_path_us = os.path.join(args.out, os.path.splitext(diffusor_fn)[0], f'{tag_name}.nrrd')            

            if not os.path.exists(os.path.dirname(out_path_us)):
                    os.makedirs(os.path.dirname(out_path_us))

            if not os.path.exists(out_path_us) or args.ow == 1:

                # sweep_us = inference(model, scheduler, sweep_diff, guidance_scale, chunks=args.g_chunks, ae_chunks=args.ae_chunks, weights=(1.0, 10.0, 1.0))

                print(f"Writing sampled labeled weep {tag_name} to {out_path}")
                sweep_np = sweep.squeeze().cpu().numpy().astype(np.uint8)
                sweep_img = sitk.GetImageFromArray(sweep_np)
                sitk.WriteImage(sweep_img, out_path)
            
                # print(f"Writing sampled simulated sweep {tag_name} to {out_path_us}")
                # sweep_us_np = sweep_us.squeeze().cpu().numpy()* 255.0
                # sweep_us_np = np.clip(sweep_us_np, 0, 255).astype(np.uint8)
                # sweep_us_img = sitk.GetImageFromArray(sweep_us_np)
                # sitk.WriteImage(sweep_us_img, out_path_us)

def split_dataframe(df, world_size):
    # Evenly split the DataFrame into `world_size` chunks
    return [df.iloc[i::world_size].reset_index(drop=True) for i in range(world_size)]

def main(args):
    
    world_size = torch.cuda.device_count()  # or any other number of processes

    df = pd.read_csv(args.csv)
    df_split = split_dataframe(df, world_size)

    mp.spawn(image_generation, args=(args, df_split), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultrasound Simulation Argument Parser")
    
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to the csv file with the diffusors')
    parser.add_argument('--params_csv', type=str, default='/mnt/raid/C1_ML_Analysis/simulated_data_export/animation_export/shapes_intensity_map_nrrd_speckel.csv',
                        help='Path to the CSV file with shape and intensity mappings.')
    
    parser.add_argument('--mount_point', type=str, default='/mnt/raid/C1_ML_Analysis', help='Mount point for the dataset.')

    parser.add_argument('--grid_w', type=int, default=256,
                        help='Grid width for the simulation.')
    parser.add_argument('--grid_h', type=int, default=256,
                        help='Grid height for the simulation.')
    parser.add_argument('--center_x', type=float, default=128.0,
                        help='X position of the circle creating the transducer.')
    parser.add_argument('--center_y', type=float, default=-40.0,
                        help='Y position of the circle creating the transducer.')
    parser.add_argument('--r1', type=float, default=20.0,
                        help='Radius of the first circle (inner).')
    parser.add_argument('--r2', type=float, default=255.0,
                        help='Radius of the second circle (outer).')
    parser.add_argument('--theta', type=float, default=np.pi / 4.25,
                        help='Aperture angle of the transducer (in radians).')
    parser.add_argument('--padding', type=int, default=55,
                        help='Padding around the simulated ultrasound image.')
    
    parser.add_argument('--g_chunks', type=int, default=1, help='Guidance number of chunks')
    parser.add_argument('--ae_chunks', type=int, default=1, help='Number of chunks for autoencoder')
    parser.add_argument('--num_inference_steps', type=int, default=100,
                        help='Number of inference steps for the diffusion model.')
    parser.add_argument('--guidance_scale', type=float, default=15.0, 
                        help='Guidance scale for the diffusion model.')
    parser.add_argument('--out', type=str, required=True,
                        help='Output directory for the generated images.')
    parser.add_argument('--ow', type=int, default=0, help='Overwrite existing files: 0 for no, 1 for yes')
    
    main(parser.parse_args())