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
```

## Input Format

The input image must be an NRRD file (*.nrrd) representing a 2D grayscale ultrasound frame.

## Usage 

Run the script via the command line:

```python
python generate_ultrasound.py input_data_path output_folder_path [-n NUM_INFERENCE_STEPS] [-g GUIDANCE_SCALE]
```

## Required Arguments

- `input_data_path`: Path to the NRRD image used as guidance.
- `output_folder_path`: Directory where the generated image will be saved.

## Optional Arguments

- `-n`, `--num_inference_steps`: Number of denoising steps during generation (default: 20).
- `-g`, `--guidance_scale`: Scale of guidance signal strength (default: 30000).

## Example

```bash
python generate_ultrasound.py /path/to/C3_us.nrrd /path/to/output -n 20 -g 30000
```
## Model Details

- Backbone: `UNet2DModel` from `diffusers`
- Scheduler: DDIM for inference, DDPM for training
- Trained weights are loaded from:
`/mnt/raid/home/ajarry/data/outputs_lightning/final53epoch/model.pth`

Note: This script assumes the model checkpoint and image dimensions (256×256) are compatible.

## Output

The output image is saved as an NRRD file in:

```php
<output_folder_path>/<frame_number>/<basename>
```

Where:

 frame_number is derived from the input path.
- basename is the original filename.
