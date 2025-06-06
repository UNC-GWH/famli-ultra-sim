# Ultrasound Image Generation with Guided Diffusion

This script generates synthetic ultrasound images using a diffusion model guided by a target image.  
It uses PyTorch Lightning, Hugging Face’s Diffusers, SimpleITK, and NRRD for data handling and inference.

---

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Packages:
  - `diffusers`
  - `pytorch_lightning`
  - `simpleitk`
  - `pynrrd`
  - `tqdm`

Install via:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers pytorch_lightning simpleitk pynrrd tqdm

## Input Format

The input image must be an NRRD file (*.nrrd) representing a 2D grayscale ultrasound frame.

## Usage 
python generate_ultrasound.py \
  <input_data_path> \
  <output_folder_path> \
  [-n NUM_INFERENCE_STEPS] \
  [-g GUIDANCE_SCALE]
