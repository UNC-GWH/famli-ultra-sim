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
import multiprocessing as mp
from image_generation import image_generation
import pandas as pd

# rsync -a

def multi_gpu_image_gen(gpu_id, csv_file):
    torch.cuda.set_device(gpu_id)

    # Load the list of paths from CSV
    with open(csv_file, 'r') as f:
        paths = [line.strip() for line in f.readlines() if line.strip()]

    print(f"GPU {gpu_id} handling {len(paths)} paths from {csv_file}")
    


    # Process each path (replace this block with your logic)
    for path in paths:
        print(f"GPU {gpu_id}: Processing {path}")
        image_generation(path, 20, 30000, "/mnt/raid/home/ajarry/data/cephalic_output")

if __name__ == "__main__":
    # List of CSVs, each for a different GPU
    csv_files = [
        "/mnt/raid/home/ajarry/data/all_paths.csv",
        "/mnt/raid/home/ajarry/data/all_paths.csv",
        "/mnt/raid/home/ajarry/data/all_paths.csv",
        "/mnt/raid/home/ajarry/data/all_paths.csv"
    ]

    processes = []
    for gpu_id in range(4):
        p = mp.Process(target=multi_gpu_image_gen, args=(gpu_id, csv_files[gpu_id]))
        p.start()
        processes.append(p)

    # Wait for all to complete
    for p in processes:
        p.join()